#!/usr/bin/env python3
"""
Compute gradient attribution scores for each LoRA module in CER-DPO.

For each (layer, module) pair, accumulates the RMS gradient norms of
lora_A and lora_B over the training set, then combines them into a
per-module attribution score that can be compared with the Frobenius
norm heatmap from §6.1.

Attribution score = sqrt(mean_sq_grad_A + mean_sq_grad_B)
This is the RMS gradient norm of the full (A, B) LoRA parameterization.

Usage:
    python compute_gradient_attribution.py \
        --model-name-or-path /data/models/StarCoder/starcoder2-7b \
        --adapter-dir output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b \
        --train-file data/mixed_sft_v1/mixed_sft_train.jsonl \
        --output-dir output/gradient_attribution_20260519/starcoder2_7b \
        --api-anchor-weight 0.1 \
        --max-samples 0
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_compare_lora import load_jsonl
from src.dpo_training import DPOCollator, VersionAwareDPODataset, VersionAwareDPOTrainer

LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+layers\.(?P<layer>\d+)\.self_attn\.(?P<module>q_proj|k_proj|v_proj|o_proj))"
    r"\.lora_(?P<ab>A|B)\.(?:default\.)?weight$"
)


def parse_grad_key(name: str) -> Optional[Tuple[int, str, str]]:
    """Returns (layer_idx, module_name, 'A'|'B') or None."""
    m = LORA_KEY_RE.match(name)
    if not m:
        return None
    return int(m.group("layer")), m.group("module"), m.group("ab")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient attribution for LoRA modules.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 = use all training pairs")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--api-anchor-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 to save memory; gradients are still numerically useful
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir), is_trainable=True)
    model.train()

    # Verify LoRA params have requires_grad
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable LoRA parameters: {len(lora_params)}")
    if not lora_params:
        raise RuntimeError("No trainable parameters found — check adapter loading.")

    # Build data
    rows = load_jsonl(args.train_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    dataset = VersionAwareDPODataset(rows, tokenizer, args.max_length)
    collator = DPOCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=collator, shuffle=False)
    print(f"DPO pairs: {len(dataset)}")

    trainer = VersionAwareDPOTrainer(
        beta=args.beta,
        api_anchor_weight=args.api_anchor_weight,
    )

    # Accumulate: sq_sum[key] = sum of squared gradient norms over batches
    sq_sum_A: Dict[Tuple[int, str], float] = {}
    sq_sum_B: Dict[Tuple[int, str], float] = {}
    n_batches = 0

    for batch in tqdm(loader, desc="Computing gradients"):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad()

        loss = trainer.compute_loss(model, batch)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            parsed = parse_grad_key(name)
            if parsed is None:
                continue
            layer, module, ab = parsed
            key = (layer, module)
            sq_norm = param.grad.float().pow(2).sum().item()
            if ab == "A":
                sq_sum_A[key] = sq_sum_A.get(key, 0.0) + sq_norm
            else:
                sq_sum_B[key] = sq_sum_B.get(key, 0.0) + sq_norm

        n_batches += 1

    print(f"Processed {n_batches} batches.")

    # Build result records
    keys = set(sq_sum_A) | set(sq_sum_B)
    results = []
    for key in sorted(keys):
        layer, module = key
        mean_sq_A = sq_sum_A.get(key, 0.0) / n_batches
        mean_sq_B = sq_sum_B.get(key, 0.0) / n_batches
        attribution = (mean_sq_A + mean_sq_B) ** 0.5
        results.append({
            "layer": layer,
            "module": module,
            "grad_rms_A": mean_sq_A ** 0.5,
            "grad_rms_B": mean_sq_B ** 0.5,
            "grad_attribution": attribution,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gradient_attribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer", "module", "grad_rms_A", "grad_rms_B", "grad_attribution"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {csv_path}")

    # Quick summary: top-10 by attribution
    top = sorted(results, key=lambda r: -r["grad_attribution"])[:10]
    print("\nTop-10 modules by gradient attribution:")
    for r in top:
        print(f"  layer {r['layer']:2d}  {r['module']:<8s}  "
              f"attr={r['grad_attribution']:.5f}  "
              f"(A={r['grad_rms_A']:.5f}, B={r['grad_rms_B']:.5f})")

    # Save manifest
    manifest = {
        "model": args.model_name_or_path,
        "adapter": str(args.adapter_dir),
        "train_file": str(args.train_file),
        "n_dpo_pairs": len(dataset),
        "n_batches": n_batches,
        "beta": args.beta,
        "api_anchor_weight": args.api_anchor_weight,
        "csv": str(csv_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
