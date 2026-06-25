#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

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
    write_json,
    write_jsonl,
)
from src.variant_compare import VariantSpec, build_default_variant_specs, normalize_variant_specs  # noqa: E402


DEFAULT_TEST_FILES = [
    "05_positive_engineering/data/processed_clean/repair_sft_test.jsonl",
    "05_positive_engineering/data/processed_clean/consistency_sft_test.jsonl",
]


def resolve_many(paths: List[str], label: str) -> List[Path]:
    return [resolve_existing_path(Path(path), label=label) for path in paths]


def run_variant(
    *,
    spec: VariantSpec,
    model_name_or_path: str,
    prompt_field: str,
    test_rows: List[Dict],
    output_dir: Path,
    max_length: int,
    max_samples: int,
) -> Dict:
    adapter_dir = resolve_existing_path(Path(spec.adapter_dir), label=f"{spec.label} adapter dir") if spec.adapter_dir else None
    tuned_lens_path = (
        resolve_existing_path(Path(spec.tuned_lens_path), label=f"{spec.label} tuned lens")
        if spec.tuned_lens_path
        else None
    )

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

    write_jsonl(output_dir / f"{spec.label}_focus_examples.jsonl", (focus_example_to_dict(example) for example in examples))

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
        sample_file = output_dir / f"{spec.label}_{lens_label}_samples.jsonl"
        summary_file = output_dir / f"{spec.label}_{lens_label}_summary.json"
        write_jsonl(sample_file, sample_results)
        write_json(
            summary_file,
            {
                "label": spec.label,
                "lens": lens_label,
                "samples": len(sample_results),
                "layer_summary": layer_summary,
                "tuned_lens_metadata": tuned_metadata,
            },
        )
        all_results[lens_label] = {
            "samples": len(sample_results),
            "layer_summary": layer_summary,
            "sample_file": str(sample_file.resolve()),
            "summary_file": str(summary_file.resolve()),
            "tuned_lens_metadata": tuned_metadata,
        }
    return all_results


def load_variant_specs(args) -> List[VariantSpec]:
    if args.variant_specs_json:
        specs_path = resolve_existing_path(Path(args.variant_specs_json), label="variant specs json")
        raw_specs = json.loads(specs_path.read_text(encoding="utf-8"))
        return normalize_variant_specs(raw_specs)
    return build_default_variant_specs(
        model_key=args.model_key,
        mechanism_root=args.mechanism_root,
        positive_engineering_root=args.positive_engineering_root,
    )


def main():
    parser = argparse.ArgumentParser(description="Run variant-aware Logit Lens / Tuned Lens comparisons.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--mechanism-root", type=Path, default=PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427")
    parser.add_argument("--positive-engineering-root", type=Path, default=PROJECT_ROOT / "05_positive_engineering")
    parser.add_argument("--variant-specs-json", type=str, default=None)
    parser.add_argument("--test-files", nargs="+", default=DEFAULT_TEST_FILES)
    parser.add_argument("--prompt-field", type=str, default="version_prompt", choices=["version_prompt", "probing_input"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    args.mechanism_root = resolve_existing_path(args.mechanism_root, label="mechanism root")
    args.positive_engineering_root = resolve_existing_path(args.positive_engineering_root, label="positive engineering root")
    test_files = resolve_many(args.test_files, "test file")
    test_rows = load_many_jsonl(test_files)

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = load_variant_specs(args)
    if not variant_specs:
        raise ValueError(f"No variant specs resolved for {args.model_key}")

    variant_results = {}
    for spec in variant_specs:
        variant_results[spec.label] = run_variant(
            spec=spec,
            model_name_or_path=model_name_or_path,
            prompt_field=args.prompt_field,
            test_rows=test_rows,
            output_dir=output_dir,
            max_length=max_length,
            max_samples=args.max_samples,
        )

    summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "prompt_field": args.prompt_field,
        "test_files": [str(path) for path in test_files],
        "max_length": max_length,
        "max_samples": args.max_samples,
        "variant_specs": [spec.__dict__ for spec in variant_specs],
        "variants": variant_results,
    }
    write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
