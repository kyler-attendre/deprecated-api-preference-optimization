#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MECH_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MECH_ROOT.parent
SRC_ROOT = MECH_ROOT
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.lens_analysis import (  # noqa: E402
    MODEL_REGISTRY,
    aggregate_layerwise,
    build_focus_examples,
    build_model,
    compare_focus_example,
    focus_example_to_dict,
    get_final_norm,
    get_output_projection,
    load_many_jsonl,
    load_tuned_lens,
    resolve_existing_path,
    safe_slug,
    write_json,
    write_jsonl,
)


DEFAULT_TEST_FILES = [
    "05_positive_engineering/data/processed_clean/repair_sft_test.jsonl",
    "05_positive_engineering/data/processed_clean/consistency_sft_test.jsonl",
]


def resolve_many(paths: List[str], label: str) -> List[Path]:
    return [resolve_existing_path(Path(path), label=label) for path in paths]


def run_variant(
    *,
    label: str,
    model_name_or_path: str,
    adapter_dir: Optional[Path],
    tuned_lens_path: Optional[Path],
    prompt_field: str,
    test_rows: List[Dict],
    output_dir: Path,
    max_length: int,
    max_samples: int,
) -> Dict:
    model, tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
    final_norm = get_final_norm(model)
    output_projection = get_output_projection(model)
    tuned_lens = None
    tuned_metadata = None
    if tuned_lens_path is not None:
        tuned_lens, tuned_metadata = load_tuned_lens(
            tuned_lens_path,
            device=next(model.parameters()).device,
        )

    examples = build_focus_examples(test_rows, prompt_field=prompt_field)
    if max_samples > 0:
        examples = examples[:max_samples]

    write_jsonl(output_dir / f"{label}_focus_examples.jsonl", (focus_example_to_dict(example) for example in examples))

    all_results = {}
    for lens_label, lens_obj in [("logit_lens", None), ("tuned_lens", tuned_lens)]:
        if lens_label == "tuned_lens" and tuned_lens is None:
            continue
        sample_results = []
        for example in examples:
            sample_results.append(
                compare_focus_example(
                    model=model,
                    tokenizer=tokenizer,
                    final_norm=final_norm,
                    output_projection=output_projection,
                    example=example,
                    max_length=max_length,
                    tuned_lens=lens_obj,
                )
            )
        layer_summary = aggregate_layerwise(sample_results)
        write_jsonl(output_dir / f"{label}_{lens_label}_samples.jsonl", sample_results)
        write_json(
            output_dir / f"{label}_{lens_label}_summary.json",
            {
                "label": label,
                "lens": lens_label,
                "samples": len(sample_results),
                "layer_summary": layer_summary,
                "tuned_lens_metadata": tuned_metadata,
            },
        )
        all_results[lens_label] = {
            "samples": len(sample_results),
            "layer_summary": layer_summary,
            "sample_file": str((output_dir / f"{label}_{lens_label}_samples.jsonl").resolve()),
            "summary_file": str((output_dir / f"{label}_{lens_label}_summary.json").resolve()),
        }
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare official base and Anchored DPO with Logit Lens / Tuned Lens.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-tuned-lens", type=Path, default=None)
    parser.add_argument("--adapter-tuned-lens", type=Path, default=None)
    parser.add_argument("--test-files", nargs="+", default=DEFAULT_TEST_FILES)
    parser.add_argument("--prompt-field", type=str, default="version_prompt", choices=["version_prompt", "probing_input"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    adapter_dir = resolve_existing_path(args.adapter_dir, label="adapter dir")
    test_files = resolve_many(args.test_files, "test file")
    base_tuned_lens = resolve_existing_path(args.base_tuned_lens, label="base tuned lens") if args.base_tuned_lens else None
    adapter_tuned_lens = (
        resolve_existing_path(args.adapter_tuned_lens, label="adapter tuned lens")
        if args.adapter_tuned_lens
        else None
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]
    test_rows = load_many_jsonl(test_files)

    base_results = run_variant(
        label="official_base",
        model_name_or_path=model_name_or_path,
        adapter_dir=None,
        tuned_lens_path=base_tuned_lens,
        prompt_field=args.prompt_field,
        test_rows=test_rows,
        output_dir=output_dir,
        max_length=max_length,
        max_samples=args.max_samples,
    )
    adapter_results = run_variant(
        label="anchored_dpo",
        model_name_or_path=model_name_or_path,
        adapter_dir=adapter_dir,
        tuned_lens_path=adapter_tuned_lens,
        prompt_field=args.prompt_field,
        test_rows=test_rows,
        output_dir=output_dir,
        max_length=max_length,
        max_samples=args.max_samples,
    )

    summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "adapter_dir": str(adapter_dir),
        "prompt_field": args.prompt_field,
        "test_files": [str(path) for path in test_files],
        "max_length": max_length,
        "max_samples": args.max_samples,
        "official_base": base_results,
        "anchored_dpo": adapter_results,
    }
    write_json(output_dir / "run_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
