"""
OV Direct-Logit-Attribution (DLA) Flip Analysis  (v2 — correct two-pass design)
=================================================================================
Pass 1: compute base model DLA for all cases.
Pass 2: merge CER-DPO adapter once, compute CER-DPO DLA for all cases.

For each head (l, h) at the decision position:
    contribution(l,h) = W_O_h @ (attn_weights_h @ V_h)   [d_model vector]
    DLA(l,h)          = dot(contribution, W_U_rep - W_U_dep)   [scalar]

"Flip head" = head where DLA switches sign or grows substantially from base → CER-DPO.

Usage:
    CUDA_VISIBLE_DEVICES=6 python ov_dla_flip.py
"""

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Three API pairs with single-token decision point
CASES = [
    {
        "name":    "torch.svd → torch.linalg.svd",
        "prompt":  "import torch\nA = torch.randn(4, 4)\nU, S, V = torch.",
        "dep_tok": "svd",
        "rep_tok": "linalg",
    },
    {
        "name":    "torch.qr → torch.linalg.qr",
        "prompt":  "import torch\nA = torch.randn(4, 4)\nQ, R = torch.",
        "dep_tok": "qr",
        "rep_tok": "linalg",
    },
    {
        "name":    "F.upsample → F.interpolate",
        "prompt":  "import torch.nn.functional as F\nout = F.",
        "dep_tok": "up",
        "rep_tok": "interpolate",
    },
]


def get_final_logit_diff(model, tok, device, case):
    inputs = tok(case["prompt"], return_tensors="pt").to(device)
    dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
    rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return (logits[rep_id] - logits[dep_id]).item()


def compute_dla_all_cases(model, tok, device, top_k_tokens=6):
    """
    Run DLA for every case. Returns dict: case_name -> list of head results.
    """
    cfg = model.config
    n_layers   = cfg.num_hidden_layers
    n_q_heads  = cfg.num_attention_heads
    d_model    = cfg.hidden_size
    d_head     = d_model // n_q_heads

    W_U = model.model.embed_tokens.weight.detach().float().to(device)  # [vocab, d_model]

    case_results = {}

    for case in CASES:
        prompt  = case["prompt"]
        dep_id  = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
        rep_id  = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
        inputs  = tok(prompt, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        dec_pos = seq_len - 1

        diff_dir = (W_U[rep_id] - W_U[dep_id])  # [d_model]

        head_outs = {}  # (l, h) -> [d_model]

        def make_hook(layer_idx):
            def hook(module, inp, out):
                if not inp:
                    return
                concat = inp[0]  # [1, seq, nQ*d_head]
                o_w = module.weight.detach().float()  # [d_model, nQ*d_head]
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
                cv = contrib.to(device)
                dla        = (diff_dir @ cv).item()
                dep_logit  = (W_U[dep_id] @ cv).item()
                rep_logit  = (W_U[rep_id] @ cv).item()
                logit_vec  = W_U @ cv
                top_ids    = torch.topk(logit_vec, top_k_tokens).indices.tolist()
                top_tokens = [tok.decode([i]) for i in top_ids]
                rows.append({
                    "layer": l, "head": h,
                    "dla": dla,
                    "dep_logit": dep_logit,
                    "rep_logit": rep_logit,
                    "top_promoted": top_tokens,
                })
        case_results[case["name"]] = rows

    return case_results


def analyze(args):
    device = "cuda"

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    for case in CASES:
        for key in ("dep_tok", "rep_tok"):
            ids = tok.encode(case[key], add_special_tokens=False)
            assert len(ids) == 1, f"Multi-token {case[key]!r}: {ids}"

    # ── Pass 1: base model ──────────────────────────────────────────────────
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, device_map=device
    )
    model.eval()

    print("Pass 1: base model DLA for all cases...")
    base_dla   = compute_dla_all_cases(model, tok, device)
    base_diffs = {c["name"]: get_final_logit_diff(model, tok, device, c) for c in CASES}
    print("  Base logit diffs:", {k: f"{v:+.2f}" for k, v in base_diffs.items()})

    # ── Pass 2: CER-DPO model ───────────────────────────────────────────────
    print("\nLoading CER-DPO adapter and merging...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    print("Pass 2: CER-DPO model DLA for all cases...")
    cerdpo_dla   = compute_dla_all_cases(model, tok, device)
    cerdpo_diffs = {c["name"]: get_final_logit_diff(model, tok, device, c) for c in CASES}
    print("  CER-DPO logit diffs:", {k: f"{v:+.2f}" for k, v in cerdpo_diffs.items()})

    # ── Compute flip scores and report ──────────────────────────────────────
    all_results = {}
    for case in CASES:
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
                "base_dla":   br["dla"],
                "cerdpo_dla": cr["dla"],
                "flip_delta": flip_delta,
                "base_dep_logit":   br["dep_logit"],
                "base_rep_logit":   br["rep_logit"],
                "cerdpo_dep_logit": cr["dep_logit"],
                "cerdpo_rep_logit": cr["rep_logit"],
                "base_top3":   br["top_promoted"][:3],
                "cerdpo_top3": cr["top_promoted"][:3],
            })

        flips.sort(key=lambda x: x["flip_delta"], reverse=True)

        print(f"\n{'='*65}")
        print(f"  {pname}")
        print(f"  Final logit diff (rep-dep): base={base_diffs[pname]:+.2f}  "
              f"cerdpo={cerdpo_diffs[pname]:+.2f}")
        print(f"\n  Top flipped heads (Δ = cerdpo_DLA - base_DLA):")
        print(f"  {'Head':8s} {'base_DLA':>9s} {'cerdpo_DLA':>11s} {'Δ':>8s}  "
              f"dep_logit B→C   rep_logit B→C")
        for f in flips[:8]:
            print(
                f"  L{f['layer']:02d}H{f['head']:02d}  "
                f"{f['base_dla']:>+9.3f}  {f['cerdpo_dla']:>+11.3f}  "
                f"{f['flip_delta']:>+8.3f}  "
                f"{f['base_dep_logit']:+.3f}→{f['cerdpo_dep_logit']:+.3f}   "
                f"{f['base_rep_logit']:+.3f}→{f['cerdpo_rep_logit']:+.3f}"
            )
            print(f"           base_top3={f['base_top3']}  cerdpo_top3={f['cerdpo_top3']}")

        print(f"\n  Bottom (became more pro-deprecated):")
        for f in flips[-5:]:
            print(
                f"  L{f['layer']:02d}H{f['head']:02d}  "
                f"Δ={f['flip_delta']:+.3f}   "
                f"dep: {f['base_dep_logit']:+.3f}→{f['cerdpo_dep_logit']:+.3f}   "
                f"rep: {f['base_rep_logit']:+.3f}→{f['cerdpo_rep_logit']:+.3f}"
            )

        all_results[pname] = {
            "case": case,
            "base_logit_diff":   base_diffs[pname],
            "cerdpo_logit_diff": cerdpo_diffs[pname],
            "top_flipped":  flips[:15],
            "bottom_flipped": flips[-10:],
        }

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def ser(obj):
        if isinstance(obj, dict):  return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [ser(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        return float(obj) if hasattr(obj, '__float__') else str(obj)

    with open(out_path, "w") as f:
        json.dump(ser(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default="/data/models/StarCoder/starcoder2-7b")
    parser.add_argument("--adapter-dir", default="../05_positive_engineering/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b")
    parser.add_argument("--output-json", default="output/ov_dla_results.json")
    args = parser.parse_args()
    analyze(args)
