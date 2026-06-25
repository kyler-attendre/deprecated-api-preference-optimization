#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys
import torch
from safetensors.torch import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_lora_delta import compute_effective_delta, parse_lora_module_key, read_adapter_config
from scripts.eval_compare_lora import build_model, evaluate_model, load_jsonl, resolve_existing_path


def prune_delta_keep_fraction(delta: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    if keep_fraction >= 1.0:
        return delta
    flat = delta.abs().reshape(-1)
    keep_count = max(1, int(flat.numel() * keep_fraction))
    threshold = torch.topk(flat, keep_count, largest=True).values[-1]
    mask = delta.abs() >= threshold
    return delta * mask


def apply_pruned_delta_to_model(model, *, adapter_dir: Path, keep_fraction: float) -> None:
    config = read_adapter_config(adapter_dir)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))

    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        parsed = parse_lora_module_key(key)
        if parsed is None:
            continue
        _, _, prefix = parsed
        slot = grouped.setdefault(prefix, {})
        if ".lora_A." in key:
            slot["lora_A"] = value
        else:
            slot["lora_B"] = value

    for prefix, tensors in grouped.items():
        module_path = prefix.replace("base_model.model.", "", 1)
        module = model.get_submodule(module_path)
        delta = compute_effective_delta(
            lora_a=tensors["lora_A"],
            lora_b=tensors["lora_B"],
            lora_alpha=config["lora_alpha"],
            lora_r=config["r"],
        )
        pruned = prune_delta_keep_fraction(delta, keep_fraction=keep_fraction)
        module.weight.data.add_(pruned.to(device=module.weight.device, dtype=module.weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sparse effective-delta variants without saving merged checkpoints.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, action="append", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
    test_file = resolve_existing_path(args.test_file, label="test file")
    rows = load_jsonl(test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict] = {}

    for keep_fraction in args.keep_fraction:
        label = f"keep_{int(round(keep_fraction * 100)):02d}"
        model, tokenizer = build_model(args.model_name_or_path)
        apply_pruned_delta_to_model(model, adapter_dir=adapter_dir, keep_fraction=keep_fraction)
        summary = evaluate_model(
            label=label,
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_path=output_dir / f"{label}_predictions.jsonl",
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            prompt_field="version_prompt",
        )
        summaries[label] = {"keep_fraction": keep_fraction, "summary": summary}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(adapter_dir),
        "test_file": str(test_file),
        "num_rows": len(rows),
        "variants": summaries,
    }
    with (output_dir / "sparse_delta_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
