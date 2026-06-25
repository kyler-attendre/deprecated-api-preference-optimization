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

from src.adl_compare import (  # noqa: E402
    build_prompt_sets,
    build_token_family_token_ids,
    compare_variant_against_base,
    prompt_to_dict,
    trace_last_token_hidden_and_logits,
)
from src.lens_analysis import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
    get_final_norm,
    get_output_projection,
    load_many_jsonl,
    resolve_existing_path,
    write_json,
    write_jsonl,
)
from src.variant_compare import build_default_variant_specs  # noqa: E402


DEFAULT_DATA_FILES = [
    "05_positive_engineering/data/processed_clean/repair_sft_test.jsonl",
    "05_positive_engineering/data/processed_clean/consistency_sft_test.jsonl",
    "05_positive_engineering/data/processed_clean/reference_sft_test.jsonl",
]

LABEL_MAP = {
    "official_base": "base",
    "plain_dpo": "dpo",
    "anchored_dpo": "anchored_dpo",
}


def resolve_many(paths: List[str], label: str) -> List[Path]:
    return [resolve_existing_path(Path(path), label=label) for path in paths]


def normalize_variant_specs(raw_specs):
    normalized = []
    for spec in raw_specs:
        label = LABEL_MAP.get(spec.label, spec.label)
        normalized.append(
            {
                "label": label,
                "adapter_dir": spec.adapter_dir,
            }
        )
    return normalized


def compute_prompt_traces(
    *,
    model,
    tokenizer,
    final_norm,
    output_projection,
    prompts,
    max_length: int,
) -> Dict[str, Dict]:
    traces = {}
    for prompt in prompts:
        traces[prompt.prompt_id] = trace_last_token_hidden_and_logits(
            model=model,
            tokenizer=tokenizer,
            final_norm=final_norm,
            output_projection=output_projection,
            prompt_text=prompt.prompt_text,
            max_length=max_length,
        )
    return traces


def main():
    parser = argparse.ArgumentParser(description="Run an ADL-style comparison on random and neutral prompts.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument(
        "--positive-engineering-root",
        type=Path,
        default=PROJECT_ROOT / "05_positive_engineering",
    )
    parser.add_argument("--mechanism-root", type=Path, default=PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427")
    parser.add_argument("--data-files", nargs="+", default=DEFAULT_DATA_FILES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-prompts-per-type", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    args.positive_engineering_root = resolve_existing_path(args.positive_engineering_root, label="positive engineering root")
    args.mechanism_root = resolve_existing_path(args.mechanism_root, label="mechanism root")
    data_files = resolve_many(args.data_files, "data file")
    rows = load_many_jsonl(data_files)

    raw_variant_specs = build_default_variant_specs(
        model_key=args.model_key,
        mechanism_root=args.mechanism_root,
        positive_engineering_root=args.positive_engineering_root,
    )
    variant_specs = normalize_variant_specs(raw_variant_specs)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_prompt_sets(rows, max_prompts_per_type=args.max_prompts_per_type)
    write_jsonl(output_dir / "prompt_manifest.jsonl", (prompt_to_dict(prompt) for prompt in prompts))
    prompt_counts = {}
    for prompt in prompts:
        prompt_counts[prompt.prompt_type] = prompt_counts.get(prompt.prompt_type, 0) + 1

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]

    base_model, tokenizer = build_model(model_name_or_path, adapter_dir=None)
    base_final_norm = get_final_norm(base_model)
    base_output_projection = get_output_projection(base_model)
    family_token_ids = build_token_family_token_ids(tokenizer, rows)
    base_prompt_traces = compute_prompt_traces(
        model=base_model,
        tokenizer=tokenizer,
        final_norm=base_final_norm,
        output_projection=base_output_projection,
        prompts=prompts,
        max_length=max_length,
    )
    del base_model
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    variants = {}
    for spec in variant_specs:
        if spec["label"] == "base":
            continue
        adapter_dir = resolve_existing_path(Path(spec["adapter_dir"]), label=f"{spec['label']} adapter dir")
        model, variant_tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
        final_norm = get_final_norm(model)
        output_projection = get_output_projection(model)
        variant_prompt_traces = compute_prompt_traces(
            model=model,
            tokenizer=variant_tokenizer,
            final_norm=final_norm,
            output_projection=output_projection,
            prompts=prompts,
            max_length=max_length,
        )
        comparison = compare_variant_against_base(
            tokenizer=variant_tokenizer,
            base_prompt_traces=base_prompt_traces,
            variant_prompt_traces=variant_prompt_traces,
            prompts=prompts,
            family_token_ids=family_token_ids,
            top_k=args.top_k,
        )
        write_jsonl(output_dir / f"{spec['label']}_prompt_rows.jsonl", comparison["prompt_rows"])
        write_json(output_dir / f"{spec['label']}_summary.json", comparison["prompt_type_summary"])
        variants[spec["label"]] = {
            "adapter_dir": str(adapter_dir),
            "prompt_type_summary": comparison["prompt_type_summary"],
            "prompt_rows_file": str((output_dir / f"{spec['label']}_prompt_rows.jsonl").resolve()),
            "summary_file": str((output_dir / f"{spec['label']}_summary.json").resolve()),
        }
        del model
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run_summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "data_files": [str(path) for path in data_files],
        "max_length": max_length,
        "max_prompts_per_type": args.max_prompts_per_type,
        "prompt_counts": prompt_counts,
        "variants": variants,
        "family_token_counts": {key: len(value) for key, value in family_token_ids.items()},
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
