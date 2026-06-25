#!/usr/bin/env python3
"""
OV DLA Analysis: Base vs CER-DPO vs CER-PT (StarCoder2-7B)
===========================================================
Three-pass design for comparing mechanisms across parameterisation choices.

  Pass 1: base model DLA
  Pass 2: CER-DPO (LoRA merged) DLA
  Pass 3+: CER-PT DLA for each prefix config

DLA formula (same as ov_dla_flip.py):
    contribution(l, h) = W_O_h @ (attn_weights_h @ V_h)   [d_model vector]
    DLA(l, h)          = dot(contribution, W_U_rep - W_U_dep)

For CER-PT, V_h includes prefix KV vectors passed as past_key_values.
Hooks are placed on base_model.model.layers[l].self_attn.o_proj.

Usage:
    CUDA_VISIBLE_DEVICES=0 python ov_dla_prefix.py
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PE_OUT = PROJECT_ROOT / "05_positive_engineering" / "output"

sys.path.insert(0, str(PROJECT_ROOT / "05_positive_engineering"))
from scripts.train_dpo_prefix import PrefixCausalLM  # noqa: E402

BASE_MODEL = "/data/models/StarCoder/starcoder2-7b"
ADAPTER_DIR = str(PE_OUT / "dpo7b_screen_full_anchor01_20260423" / "starcoder2_7b")
CERPT_BASE = PE_OUT / "cerpt_20260605"

PREFIX_CONFIGS = [
    {"name": "cerpt_m16",    "dir": CERPT_BASE / "cerpt_m16"},
    {"name": "cerpt_m64",    "dir": CERPT_BASE / "cerpt_m64"},
    {"name": "cerpt_m256",   "dir": CERPT_BASE / "cerpt_m256"},
    {"name": "pt_only_m64",  "dir": CERPT_BASE / "pt_only_m64"},
]

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

OUT_JSON = SCRIPT_DIR.parent / "output" / "ov_dla_prefix_7b.json"


def get_logit_diff(model, tok, inputs, dep_id, rep_id):
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return (logits[rep_id] - logits[dep_id]).item()


def compute_dla(model, tok, device, top_k=6):
    """
    Compute per-head DLA for all CASES.
    model: any nn.Module with .model.layers[l].self_attn.o_proj
    Returns: {case_name: [{"layer":l, "head":h, "dla":..., "dep_logit":..., "rep_logit":...}]}
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_q_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_head = d_model // n_q_heads

    W_U = model.model.embed_tokens.weight.detach().float().to(device)

    case_results = {}

    for case in CASES:
        dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
        rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
        inputs = tok(case["prompt"], return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        dec_pos = seq_len - 1
        diff_dir = (W_U[rep_id] - W_U[dep_id]).float()

        head_outs = {}

        def make_hook(layer_idx):
            def hook(module, inp, out):
                if not inp:
                    return
                concat = inp[0]  # [bsz, seq, nQ*d_head]
                o_w = module.weight.detach().float()
                with torch.no_grad():
                    for h in range(n_q_heads):
                        h_out = concat[0, dec_pos, h * d_head:(h + 1) * d_head].float()
                        o_h = o_w[:, h * d_head:(h + 1) * d_head]
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
                dla = (diff_dir @ cv).item()
                dep_logit = (W_U[dep_id] @ cv).item()
                rep_logit = (W_U[rep_id] @ cv).item()
                logit_vec = W_U @ cv
                top_ids = torch.topk(logit_vec, top_k).indices.tolist()
                rows.append({
                    "layer": l, "head": h,
                    "rel_depth": round(l / (n_layers - 1), 4),
                    "dla": dla,
                    "dep_logit": dep_logit,
                    "rep_logit": rep_logit,
                    "top_promoted": [tok.decode([i]) for i in top_ids],
                })
        case_results[case["name"]] = rows

    return case_results


def compute_dla_prefix(prefix_model, tok, device, top_k=6):
    """
    Compute DLA for a PrefixCausalLM model.

    The prefix is injected as past_key_values before the prompt tokens.
    Hooks are registered on prefix_model.base_model.model.layers[l].self_attn.o_proj.
    The attention over prefix positions is transparently captured in the head output
    because attn_weights @ V already includes prefix KV contributions.
    """
    base = prefix_model.base_model
    cfg = base.config
    n_layers = cfg.num_hidden_layers
    n_q_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_head = d_model // n_q_heads

    W_U = base.model.embed_tokens.weight.detach().float().to(device)

    case_results = {}

    for case in CASES:
        dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
        rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
        inputs = tok(case["prompt"], return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        dec_pos = seq_len - 1  # last prompt position; prefix is in past_kv, not input_ids
        diff_dir = (W_U[rep_id] - W_U[dep_id]).float()

        head_outs = {}

        def make_hook(layer_idx):
            def hook(module, inp, out):
                if not inp:
                    return
                concat = inp[0]  # [bsz, seq_len, nQ*d_head] — prompt positions only
                o_w = module.weight.detach().float()
                with torch.no_grad():
                    for h in range(n_q_heads):
                        h_out = concat[0, dec_pos, h * d_head:(h + 1) * d_head].float()
                        o_h = o_w[:, h * d_head:(h + 1) * d_head]
                        head_outs[(layer_idx, h)] = (o_h @ h_out).cpu()
            return hook

        handles = [
            base.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l))
            for l in range(n_layers)
        ]
        with torch.no_grad():
            _ = prefix_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        for h in handles:
            h.remove()

        rows = []
        for l in range(n_layers):
            for h in range(n_q_heads):
                contrib = head_outs.get((l, h))
                if contrib is None:
                    continue
                cv = contrib.to(device)
                dla = (diff_dir @ cv).item()
                dep_logit = (W_U[dep_id] @ cv).item()
                rep_logit = (W_U[rep_id] @ cv).item()
                logit_vec = W_U @ cv
                top_ids = torch.topk(logit_vec, top_k).indices.tolist()
                rows.append({
                    "layer": l, "head": h,
                    "rel_depth": round(l / (n_layers - 1), 4),
                    "dla": dla,
                    "dep_logit": dep_logit,
                    "rep_logit": rep_logit,
                    "top_promoted": [tok.decode([i]) for i in top_ids],
                })
        case_results[case["name"]] = rows

    return case_results


def get_logit_diff_prefix(prefix_model, tok, device, case):
    dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
    rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
    inputs = tok(case["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = prefix_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return (out.logits[0, -1, rep_id] - out.logits[0, -1, dep_id]).item()


def flip_analysis(base_dla, query_dla, base_diffs, query_diffs, label, n_layers, top_k=8):
    results = {}
    for case in CASES:
        pname = case["name"]
        bmap = {(r["layer"], r["head"]): r for r in base_dla[pname]}
        qmap = {(r["layer"], r["head"]): r for r in query_dla[pname]}

        flips = []
        for (l, h), br in bmap.items():
            qr = qmap.get((l, h))
            if qr is None:
                continue
            flip_delta = qr["dla"] - br["dla"]
            flips.append({
                "layer": l, "head": h,
                "rel_depth": br["rel_depth"],
                "base_dla": br["dla"],
                f"{label}_dla": qr["dla"],
                "flip_delta": flip_delta,
                "base_dep_logit": br["dep_logit"],
                "base_rep_logit": br["rep_logit"],
                f"{label}_dep_logit": qr["dep_logit"],
                f"{label}_rep_logit": qr["rep_logit"],
                "base_top3": br["top_promoted"][:3],
                f"{label}_top3": qr["top_promoted"][:3],
            })
        flips.sort(key=lambda x: x["flip_delta"], reverse=True)

        print(f"\n{'='*65}")
        print(f"  {pname}  [{label}]")
        print(f"  logit diff: base={base_diffs[pname]:+.2f}  {label}={query_diffs[pname]:+.2f}")
        print(f"  Top flipped heads:")
        print(f"  {'Head':8s} {'rel_d':>6s} {'base_DLA':>9s} {label+'_DLA':>12s} {'Δ':>8s}")
        for f in flips[:top_k]:
            print(
                f"  L{f['layer']:02d}H{f['head']:02d}  "
                f"{f['rel_depth']:>6.3f}  "
                f"{f['base_dla']:>+9.3f}  {f[label+'_dla']:>+12.3f}  "
                f"{f['flip_delta']:>+8.3f}"
            )

        results[pname] = {
            "case": {"name": pname, "dep_tok": case["dep_tok"], "rep_tok": case["rep_tok"]},
            "base_logit_diff": base_diffs[pname],
            f"{label}_logit_diff": query_diffs[pname],
            "top_flipped": flips[:15],
            "bottom_flipped": flips[-8:],
        }
    return results


def ser(obj):
    if isinstance(obj, dict):
        return {k: ser(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ser(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return float(obj) if hasattr(obj, "__float__") else str(obj)


def main():
    device = "cuda"

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    for case in CASES:
        for key in ("dep_tok", "rep_tok"):
            ids = tok.encode(case[key], add_special_tokens=False)
            assert len(ids) == 1, f"Multi-token {case[key]!r}: {ids}"
    print("  Token check passed.")

    # ── Pass 1: base model ──────────────────────────────────────────────────
    print("\nLoading base model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    print("Pass 1: base model DLA...")
    base_dla = compute_dla(model, tok, device)
    base_diffs = {}
    for case in CASES:
        dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
        rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
        inputs = tok(case["prompt"], return_tensors="pt").to(device)
        base_diffs[case["name"]] = get_logit_diff(model, tok, inputs, dep_id, rep_id)
    print("  Base logit diffs:", {k: f"{v:+.2f}" for k, v in base_diffs.items()})

    # ── Pass 2: CER-DPO ─────────────────────────────────────────────────────
    print("\nLoading CER-DPO LoRA adapter and merging...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.eval()

    print("Pass 2: CER-DPO DLA...")
    cerdpo_dla = compute_dla(model, tok, device)
    cerdpo_diffs = {}
    for case in CASES:
        dep_id = tok.encode(case["dep_tok"], add_special_tokens=False)[0]
        rep_id = tok.encode(case["rep_tok"], add_special_tokens=False)[0]
        inputs = tok(case["prompt"], return_tensors="pt").to(device)
        cerdpo_diffs[case["name"]] = get_logit_diff(model, tok, inputs, dep_id, rep_id)
    print("  CER-DPO logit diffs:", {k: f"{v:+.2f}" for k, v in cerdpo_diffs.items()})

    all_results = {"base": {}, "cerdpo": {}}
    cerdpo_res = flip_analysis(base_dla, cerdpo_dla, base_diffs, cerdpo_diffs, "cerdpo",
                               model.config.num_hidden_layers)
    all_results["cerdpo"] = cerdpo_res

    # ── Pass 3+: CER-PT configs ──────────────────────────────────────────────
    print("\nReloading clean base model for CER-PT passes...")
    del model
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    base_model.eval()

    for cfg_entry in PREFIX_CONFIGS:
        pname = cfg_entry["name"]
        pt_file = cfg_entry["dir"] / "prefix_params.pt"
        if not pt_file.exists():
            print(f"\n[SKIP] {pname}: prefix_params.pt not found at {pt_file}")
            continue

        print(f"\nLoading prefix: {pname}...")
        state = torch.load(pt_file, map_location="cpu")
        prefix_len = state["prefix_length"]
        prefix_model = PrefixCausalLM(base_model, prefix_length=prefix_len)
        prefix_model.prefix_k.data.copy_(state["prefix_k"].to(dtype=prefix_model.prefix_k.dtype))
        prefix_model.prefix_v.data.copy_(state["prefix_v"].to(dtype=prefix_model.prefix_v.dtype))
        prefix_model.eval()
        print(f"  prefix_length={prefix_len}, params={prefix_len * state['n_layers'] * state['n_kv_heads'] * state['d_head'] * 2 / 1e6:.2f}M")

        print(f"Pass: {pname} DLA...")
        pt_dla = compute_dla_prefix(prefix_model, tok, device)
        pt_diffs = {case["name"]: get_logit_diff_prefix(prefix_model, tok, device, case)
                    for case in CASES}
        print(f"  {pname} logit diffs:", {k: f"{v:+.2f}" for k, v in pt_diffs.items()})

        pt_res = flip_analysis(base_dla, pt_dla, base_diffs, pt_diffs, pname,
                               base_model.config.num_hidden_layers)
        all_results[pname] = pt_res

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(ser(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()
