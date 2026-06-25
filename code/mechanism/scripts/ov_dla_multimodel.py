"""
OV DLA Flip Analysis — Multi-Model Version
==========================================
Two-pass design (same as ov_dla_flip.py):
  Pass 1: base model DLA for all cases
  Pass 2: merge CER-DPO adapter, DLA for all cases

Supports: starcoder2_3b, starcoder2_15b, deepseek_6_7b
(starcoder2_7b results already exist in ov_dla_results.json)

Usage:
    CUDA_VISIBLE_DEVICES=0 python ov_dla_multimodel.py --model starcoder2_3b
    CUDA_VISIBLE_DEVICES=1 python ov_dla_multimodel.py --model starcoder2_15b
    CUDA_VISIBLE_DEVICES=4 python ov_dla_multimodel.py --model deepseek_6_7b
"""

import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR  = Path(__file__).parent
MECH_DIR    = SCRIPT_DIR.parent                        # 06_mechanism/
PE_OUT_DIR  = MECH_DIR.parent / "05_positive_engineering" / "output"

MODEL_REGISTRY = {
    "starcoder2_3b": {
        "base_model": "/data/models/StarCoder/starcoder2-3b",
        "adapter_dir": str(PE_OUT_DIR / "dpo_anchor_full01_20260423" / "starcoder2_3b"),
        # StarCoder2 tokenizer: svd=33607, linalg=21254, qr=13652, up=436, interpolate=33546
        "cases": [
            {"name": "torch.svd → torch.linalg.svd",
             "prompt": "import torch\nA = torch.randn(4, 4)\nU, S, V = torch.",
             "dep_tok_str": "svd", "rep_tok_str": "linalg"},
            {"name": "torch.qr → torch.linalg.qr",
             "prompt": "import torch\nA = torch.randn(4, 4)\nQ, R = torch.",
             "dep_tok_str": "qr",  "rep_tok_str": "linalg"},
            {"name": "F.upsample → F.interpolate",
             "prompt": "import torch.nn.functional as F\nout = F.",
             "dep_tok_str": "up",  "rep_tok_str": "interpolate"},
        ],
    },
    "starcoder2_15b": {
        "base_model": "/data/models/StarCoder/starcoder2-15b",
        "adapter_dir": str(PE_OUT_DIR / "dpo_anchor_full01_20260423" / "starcoder2_15b"),
        "cases": [
            {"name": "torch.svd → torch.linalg.svd",
             "prompt": "import torch\nA = torch.randn(4, 4)\nU, S, V = torch.",
             "dep_tok_str": "svd", "rep_tok_str": "linalg"},
            {"name": "torch.qr → torch.linalg.qr",
             "prompt": "import torch\nA = torch.randn(4, 4)\nQ, R = torch.",
             "dep_tok_str": "qr",  "rep_tok_str": "linalg"},
            {"name": "F.upsample → F.interpolate",
             "prompt": "import torch.nn.functional as F\nout = F.",
             "dep_tok_str": "up",  "rep_tok_str": "interpolate"},
        ],
    },
    "deepseek_6_7b": {
        "base_model": "/data/models/deepseek-ai/deepseek-coder-6.7b-instruct",
        "adapter_dir": str(PE_OUT_DIR / "dpo_anchor_full01_20260423" / "deepseek_coder_6_7b_instruct"),
        # DeepSeek tokenizer: sv=10477, l=75, q=80, up=393, inter=2263
        "cases": [
            {"name": "torch.svd → torch.linalg.svd",
             "prompt": "import torch\nA = torch.randn(4, 4)\nU, S, V = torch.",
             "dep_tok_str": "sv",    "rep_tok_str": "l"},
            {"name": "torch.qr → torch.linalg.qr",
             "prompt": "import torch\nA = torch.randn(4, 4)\nQ, R = torch.",
             "dep_tok_str": "q",     "rep_tok_str": "l"},
            {"name": "F.upsample → F.interpolate",
             "prompt": "import torch.nn.functional as F\nout = F.",
             "dep_tok_str": "up",    "rep_tok_str": "inter"},
        ],
    },
}


def get_final_logit_diff(model, tok, device, case):
    dep_id = tok.encode(case["dep_tok_str"], add_special_tokens=False)[0]
    rep_id = tok.encode(case["rep_tok_str"], add_special_tokens=False)[0]
    inputs = tok(case["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return (logits[rep_id] - logits[dep_id]).item(), dep_id, rep_id


def compute_dla_all_cases(model, tok, device, cases, top_k=6):
    cfg       = model.config
    n_layers  = cfg.num_hidden_layers
    n_q_heads = cfg.num_attention_heads
    d_model   = cfg.hidden_size
    d_head    = d_model // n_q_heads

    W_U = model.model.embed_tokens.weight.detach().float().to(device)

    all_results = {}

    for case in cases:
        dep_id = tok.encode(case["dep_tok_str"], add_special_tokens=False)[0]
        rep_id = tok.encode(case["rep_tok_str"], add_special_tokens=False)[0]
        inputs  = tok(case["prompt"], return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        dec_pos = seq_len - 1
        diff_dir = (W_U[rep_id] - W_U[dep_id])

        head_outs = {}

        def make_hook(layer_idx):
            def hook(module, inp, out):
                if not inp:
                    return
                concat = inp[0]
                o_w = module.weight.detach().float()
                with torch.no_grad():
                    for h in range(n_q_heads):
                        h_out = concat[0, dec_pos, h*d_head:(h+1)*d_head].float()
                        o_h   = o_w[:, h*d_head:(h+1)*d_head]
                        head_outs[(layer_idx, h)] = (o_h @ h_out).cpu()
            return hook

        handles = [
            model.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l))
            for l in range(n_layers)
        ]
        with torch.no_grad():
            _ = model(**inputs)
        for h in handles:
            h.remove()

        rows = []
        for l in range(n_layers):
            for h in range(n_q_heads):
                contrib = head_outs.get((l, h))
                if contrib is None:
                    continue
                cv        = contrib.to(device)
                dla       = (diff_dir @ cv).item()
                dep_logit = (W_U[dep_id] @ cv).item()
                rep_logit = (W_U[rep_id] @ cv).item()
                # Relative depth: 0.0 (first layer) to 1.0 (last layer)
                rel_depth = l / (n_layers - 1)
                rows.append({
                    "layer": l, "head": h,
                    "rel_depth": round(rel_depth, 4),
                    "dla": dla, "dep_logit": dep_logit, "rep_logit": rep_logit,
                })
        all_results[case["name"]] = rows

    return all_results


def analyze(args):
    device   = "cuda"
    reg      = MODEL_REGISTRY[args.model]
    cases    = reg["cases"]
    out_json = MECH_DIR / "output" / f"ov_dla_{args.model}.json"

    print(f"Model: {args.model}")
    tok = AutoTokenizer.from_pretrained(reg["base_model"])

    # Verify tokens
    for case in cases:
        for key in ("dep_tok_str", "rep_tok_str"):
            ids = tok.encode(case[key], add_special_tokens=False)
            assert len(ids) == 1, f"Multi-token {case[key]!r}: {ids}"
    print("  Token check passed.")

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        reg["base_model"], dtype=torch.float16, device_map=device
    )
    model.eval()

    print("Pass 1: base model DLA...")
    base_dla = compute_dla_all_cases(model, tok, device, cases)
    base_diffs = {}
    for c in cases:
        diff, dep_id, rep_id = get_final_logit_diff(model, tok, device, c)
        base_diffs[c["name"]] = {"diff": diff, "dep_id": dep_id, "rep_id": rep_id}
    print("  Base logit diffs:", {k: f"{v['diff']:+.2f}" for k, v in base_diffs.items()})

    print("\nLoading CER-DPO adapter and merging...")
    model = PeftModel.from_pretrained(model, reg["adapter_dir"])
    model = model.merge_and_unload()
    model.eval()

    print("Pass 2: CER-DPO DLA...")
    cerdpo_dla = compute_dla_all_cases(model, tok, device, cases)
    cerdpo_diffs = {}
    for c in cases:
        diff, dep_id, rep_id = get_final_logit_diff(model, tok, device, c)
        cerdpo_diffs[c["name"]] = {"diff": diff, "dep_id": dep_id, "rep_id": rep_id}
    print("  CER-DPO logit diffs:", {k: f"{v['diff']:+.2f}" for k, v in cerdpo_diffs.items()})

    # Compute flip scores and report
    all_results = {}
    cfg       = model.config
    n_layers  = cfg.num_hidden_layers
    n_q_heads = cfg.num_attention_heads

    for case in cases:
        pname = case["name"]
        bmap = {(r["layer"], r["head"]): r for r in base_dla[pname]}
        cmap = {(r["layer"], r["head"]): r for r in cerdpo_dla[pname]}

        flips = []
        for (l, h), br in bmap.items():
            cr = cmap.get((l, h))
            if cr is None:
                continue
            flip_delta = cr["dla"] - br["dla"]
            flips.append({
                "layer": l, "head": h,
                "rel_depth": br["rel_depth"],
                "base_dla": br["dla"],   "cerdpo_dla": cr["dla"],
                "flip_delta": flip_delta,
                "base_dep_logit": br["dep_logit"], "base_rep_logit": br["rep_logit"],
                "cerdpo_dep_logit": cr["dep_logit"], "cerdpo_rep_logit": cr["rep_logit"],
            })
        flips.sort(key=lambda x: x["flip_delta"], reverse=True)

        print(f"\n{'='*60}")
        print(f"  {pname}")
        print(f"  logit diff: base={base_diffs[pname]['diff']:+.2f}  "
              f"cerdpo={cerdpo_diffs[pname]['diff']:+.2f}")
        print(f"  Top flipped heads:")
        print(f"  {'Head':8s} {'rel_d':>6s} {'base_DLA':>9s} {'cerdpo_DLA':>11s} {'Δ':>8s}  dep B→C  rep B→C")
        for f in flips[:6]:
            print(
                f"  L{f['layer']:02d}H{f['head']:02d}  "
                f"{f['rel_depth']:>6.3f}  "
                f"{f['base_dla']:>+9.3f}  {f['cerdpo_dla']:>+11.3f}  "
                f"{f['flip_delta']:>+8.3f}  "
                f"{f['base_dep_logit']:+.3f}→{f['cerdpo_dep_logit']:+.3f}  "
                f"{f['base_rep_logit']:+.3f}→{f['cerdpo_rep_logit']:+.3f}"
            )

        all_results[pname] = {
            "case":              case,
            "model_key":         args.model,
            "n_layers":          n_layers,
            "n_q_heads":         n_q_heads,
            "base_logit_diff":   base_diffs[pname]["diff"],
            "cerdpo_logit_diff": cerdpo_diffs[pname]["diff"],
            "top_flipped":       flips[:20],
            "bottom_flipped":    flips[-10:],
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)

    def ser(obj):
        if isinstance(obj, dict):  return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [ser(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        return float(obj) if hasattr(obj, "__float__") else str(obj)

    with open(out_json, "w") as f:
        json.dump(ser(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    analyze(args)
