"""
OV Circuit Token-Level Flip Analysis for CER-DPO
=================================================
For each (deprecated, replacement) API pair, compute for every attention head (l, h):
  ov_logits(t) = E[t] @ W_V_kv @ W_O_h @ E^T   (promotion scores over vocab)
where t is the first-diverging token of the deprecated API.

"Flip score" = rank_base(dep) - rank_cerdpo(dep) + rank_cerdpo(rep) - rank_base(rep)
(higher = head more strongly flipped toward replacement after CER-DPO)

Reports top-5 flipped heads per API pair with concrete before/after token rankings.

Usage:
    CUDA_VISIBLE_DEVICES=6 python ov_circuit_flip.py \
        --base-model /data/models/StarCoder/starcoder2-7b \
        --adapter-dir ../05_positive_engineering/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b \
        --output-json output/ov_flip_results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# API pairs to analyze: (deprecated, replacement, shared_prefix)
# Only keep pairs where the diverging token is a single clean token in StarCoder2 tokenizer.
API_PAIRS = [
    {
        "name":        "torch.svd → torch.linalg.svd",
        "deprecated":  "torch.svd",
        "replacement": "torch.linalg.svd",
        "dep_token":   "svd",       # first diverging token (deprecated side)
        "rep_token":   "linalg",    # first diverging token (replacement side)
    },
    {
        "name":        "torch.qr → torch.linalg.qr",
        "deprecated":  "torch.qr",
        "replacement": "torch.linalg.qr",
        "dep_token":   "qr",
        "rep_token":   "linalg",
    },
    {
        "name":        "torch.symeig → torch.linalg.eigh",
        "deprecated":  "torch.symeig",
        "replacement": "torch.linalg.eigh",
        "dep_token":   "sy",       # 'symeig' tokenizes as ['sy','me','ig']
        "rep_token":   "linalg",
    },
    {
        "name":        "torch.nn.functional.upsample → ...interpolate",
        "deprecated":  "torch.nn.functional.upsample",
        "replacement": "torch.nn.functional.interpolate",
        "dep_token":   "up",           # 'upsample' -> ['up','sample']
        "rep_token":   "interpolate",
    },
]


def get_ov_logits(embed, v_weight, o_weight, dep_tok_id, top_k=20, device="cuda"):
    """
    Compute OV-circuit promotion logits for a single attention head.

    Args:
        embed:    [vocab, d_model] embedding matrix (float32 or float16)
        v_weight: [d_head, d_model] V-projection weight for this KV head
        o_weight: [d_model, d_head] O-projection weight for this Q head
        dep_tok_id: int, token id whose OV contribution we measure

    Returns:
        logits: [vocab] promotion scores
        top_ids: top_k token ids
    """
    # embed[dep_tok_id] @ W_V^T -> value vector of shape [d_head]
    e_dep = embed[dep_tok_id].to(device)  # [d_model]
    val = v_weight @ e_dep                  # [d_head]
    # map through O projection -> residual stream contribution [d_model]
    res = o_weight @ val                    # [d_model]
    # project back to vocab
    logits = embed @ res                    # [vocab]
    top_ids = torch.topk(logits, top_k).indices
    return logits.cpu(), top_ids.cpu()


def rank_of(logits, tok_id):
    """Return the rank (0-indexed, lower = better) of tok_id in logits."""
    return (logits > logits[tok_id]).sum().item()


def analyze(args):
    device = "cuda"

    print("Loading tokenizer and base model...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    # Load embedding matrix once
    embed = model.model.embed_tokens.weight.detach().float()  # [vocab, d_model]
    vocab_size = embed.shape[0]

    cfg = model.config
    n_layers      = cfg.num_hidden_layers       # 32
    n_q_heads     = cfg.num_attention_heads      # 36
    n_kv_heads    = cfg.num_key_value_heads      # 4
    d_model       = cfg.hidden_size              # 4608
    d_head        = d_model // n_q_heads         # 128
    kv_groups     = n_q_heads // n_kv_heads      # 9

    print(f"Architecture: {n_layers} layers, {n_q_heads} Q heads, {n_kv_heads} KV heads, d_head={d_head}")

    # Precompute token ids for all pairs
    for pair in API_PAIRS:
        for key in ("dep_token", "rep_token"):
            ids = tok.encode(pair[key], add_special_tokens=False)
            assert len(ids) == 1, f"Token '{pair[key]}' is multi-token: {ids}"
            pair[f"{key}_id"] = ids[0]
        print(f"  {pair['name']}: dep_id={pair['dep_token_id']} ({pair['dep_token']!r}), "
              f"rep_id={pair['rep_token_id']} ({pair['rep_token']!r})")

    def extract_wv_wo(layer_idx, q_head_idx):
        """Extract V-weight for this Q head's KV group and O-weight for this Q head."""
        attn = model.model.layers[layer_idx].self_attn
        kv_h = q_head_idx // kv_groups
        v_w  = attn.v_proj.weight.detach().float()  # [n_kv_heads*d_head, d_model]
        o_w  = attn.o_proj.weight.detach().float()  # [d_model, n_q_heads*d_head]
        v_slice = v_w[kv_h * d_head : (kv_h + 1) * d_head, :]   # [d_head, d_model]
        o_slice = o_w[:, q_head_idx * d_head : (q_head_idx + 1) * d_head]  # [d_model, d_head]
        return v_slice.to(device), o_slice.to(device)

    # ---- Phase 1: base model OV scores ----
    print("\n[Phase 1] Computing base model OV scores...")
    base_scores = {}   # (layer, head, pair_name) -> (dep_rank, rep_rank, top5)

    for l in range(n_layers):
        for h in range(n_q_heads):
            v_w, o_w = extract_wv_wo(l, h)
            for pair in API_PAIRS:
                dep_id = pair["dep_token_id"]
                rep_id = pair["rep_token_id"]
                logits, top_ids = get_ov_logits(embed.to(device), v_w, o_w, dep_id)
                dep_rank = rank_of(logits, dep_id)
                rep_rank = rank_of(logits, rep_id)
                top5     = [tok.decode([i.item()]) for i in top_ids[:5]]
                base_scores[(l, h, pair["name"])] = {
                    "dep_rank": dep_rank,
                    "rep_rank": rep_rank,
                    "top5": top5,
                    "dep_logit": logits[dep_id].item(),
                    "rep_logit": logits[rep_id].item(),
                }
        if (l + 1) % 8 == 0:
            print(f"  Layer {l+1}/{n_layers} done")

    # ---- Phase 2: load CER-DPO adapter ----
    print("\n[Phase 2] Loading CER-DPO adapter and computing deltas...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.eval()
    print("  Adapter merged.")

    # ---- Phase 3: CER-DPO OV scores ----
    print("\n[Phase 3] Computing CER-DPO OV scores...")
    cerdpo_scores = {}

    for l in range(n_layers):
        for h in range(n_q_heads):
            v_w, o_w = extract_wv_wo(l, h)
            for pair in API_PAIRS:
                dep_id = pair["dep_token_id"]
                rep_id = pair["rep_token_id"]
                logits, top_ids = get_ov_logits(embed.to(device), v_w, o_w, dep_id)
                dep_rank = rank_of(logits, dep_id)
                rep_rank = rank_of(logits, rep_id)
                top5     = [tok.decode([i.item()]) for i in top_ids[:5]]
                cerdpo_scores[(l, h, pair["name"])] = {
                    "dep_rank": dep_rank,
                    "rep_rank": rep_rank,
                    "top5": top5,
                    "dep_logit": logits[dep_id].item(),
                    "rep_logit": logits[rep_id].item(),
                }
        if (l + 1) % 8 == 0:
            print(f"  Layer {l+1}/{n_layers} done")

    # ---- Phase 4: compute flip scores ----
    print("\n[Phase 4] Computing flip scores...")
    results = {}

    for pair in API_PAIRS:
        pname = pair["name"]
        flips = []
        for l in range(n_layers):
            for h in range(n_q_heads):
                key = (l, h, pname)
                b = base_scores[key]
                c = cerdpo_scores[key]
                # Flip score: base should rank dep low and rep high → after CER-DPO rep should rank high
                # We want: large positive = head strongly promotes replacement after CER-DPO
                # Score = (rep rises in rank) + (dep falls in rank)
                rep_gain = b["rep_rank"] - c["rep_rank"]   # positive = rep got better rank
                dep_drop = c["dep_rank"] - b["dep_rank"]   # positive = dep got worse rank
                flip_score = rep_gain + dep_drop
                flips.append({
                    "layer": l, "head": h,
                    "flip_score": flip_score,
                    "rep_gain": rep_gain,
                    "dep_drop": dep_drop,
                    "base": b,
                    "cerdpo": c,
                })
        flips.sort(key=lambda x: x["flip_score"], reverse=True)
        results[pname] = {
            "pair": pair,
            "top_flipped_heads": flips[:10],
            "bottom_flipped_heads": flips[-5:],   # heads that moved opposite direction
        }
        print(f"\n  {pname}")
        print(f"  {'Head':8s} {'flip':>6s} {'rep_gain':>9s} {'dep_drop':>9s}  base_top5  →  cerdpo_top5")
        for f in flips[:5]:
            print(f"  L{f['layer']:02d}H{f['head']:02d}  "
                  f"{f['flip_score']:>6d}  {f['rep_gain']:>9d}  {f['dep_drop']:>9d}  "
                  f"{f['base']['top5'][:3]} → {f['cerdpo']['top5'][:3]}")
            print(f"          base: dep_rank={f['base']['dep_rank']}, rep_rank={f['base']['rep_rank']}, "
                  f"dep_logit={f['base']['dep_logit']:.2f}, rep_logit={f['base']['rep_logit']:.2f}")
            print(f"          cerdpo: dep_rank={f['cerdpo']['dep_rank']}, rep_rank={f['cerdpo']['rep_rank']}, "
                  f"dep_logit={f['cerdpo']['dep_logit']:.2f}, rep_logit={f['cerdpo']['rep_logit']:.2f}")

    # Save results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",   default="/data/models/StarCoder/starcoder2-7b")
    parser.add_argument("--adapter-dir",  default="../05_positive_engineering/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b")
    parser.add_argument("--output-json",  default="output/ov_flip_results.json")
    args = parser.parse_args()
    analyze(args)
