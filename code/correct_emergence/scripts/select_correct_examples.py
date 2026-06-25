#!/usr/bin/env python3
"""Step 0 entry point: build candidate examples from repair/consistency test sets
and EDAPIBench, run them through a base backbone with teacher forcing, and split
them into "base model predicts the ground-truth API correctly" subsets
(first-token-correct / full-span-correct).

Usage example (smoke test, 10 samples per source):
    python scripts/select_correct_examples.py \
        --model-key starcoder2_7b \
        --sources repair consistency edapibench \
        --max-samples-per-source 10 \
        --output-dir output/smoke_starcoder2_7b
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_07 = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_ROOT_07.parent
# `06_mechanism/src` and `07_correct_emergence/src` are both packages literally
# named `src`; add `06_mechanism/src` itself (not its parent) to sys.path and
# import `lens_analysis` as a top-level module so the two `src` packages never
# collide in sys.modules.
MECH_SRC_DIR = PROJECT_ROOT / "06_mechanism" / "src"

for path in (MECH_SRC_DIR, PROJECT_ROOT_07):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.correct_selection import (  # noqa: E402
    CorrectExample,
    build_edapibench_examples,
    build_repair_consistency_examples,
    evaluate_example_correctness,
    example_to_dict,
    load_edapibench_rows,
    load_jsonl_rows,
)
from lens_analysis import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
    write_json,
    write_jsonl,
)

REPAIR_CONSISTENCY_FILES = {
    "repair": PROJECT_ROOT / "05_positive_engineering/data/processed_clean/repair_sft_test.jsonl",
    "consistency": PROJECT_ROOT / "05_positive_engineering/data/processed_clean/consistency_sft_test.jsonl",
}

EDAPIBENCH_FILES = [
    PROJECT_ROOT / "adalora/edapibench-src/EDAPI-Bench-main/data/EditDeprecatedAPI/codegemma-2b/all.json",
    PROJECT_ROOT / "adalora/edapibench-src/EDAPI-Bench-main/data/EditDeprecatedAPI/qwencoder-3b/all.json",
    PROJECT_ROOT / "adalora/edapibench-src/EDAPI-Bench-main/data/EditDeprecatedAPI/deepseek-1.3b/all.json",
]

ALL_SOURCES = ["repair", "consistency", "edapibench"]


def build_examples_for_source(source: str, *, prompt_field: str) -> List[CorrectExample]:
    if source in ("repair", "consistency"):
        path = REPAIR_CONSISTENCY_FILES[source]
        if not path.exists():
            raise FileNotFoundError(f"{source} test file not found: {path}")
        rows = load_jsonl_rows(path)
        return build_repair_consistency_examples(rows, source=source, prompt_field=prompt_field)
    if source == "edapibench":
        missing = [p for p in EDAPIBENCH_FILES if not p.exists()]
        if missing:
            raise FileNotFoundError(f"EDAPIBench file(s) not found: {missing}")
        rows = load_edapibench_rows(EDAPIBENCH_FILES)
        return build_edapibench_examples(rows)
    raise ValueError(f"Unknown source: {source}")


def summarize_group(records: List[Dict], group_key: str) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(record)
    summary = {}
    for key, items in sorted(groups.items()):
        n = len(items)
        first_hits = sum(1 for item in items if item["first_token_correct"])
        span_hits = sum(1 for item in items if item["full_span_correct"])
        summary[key] = {
            "evaluated": n,
            "first_token_correct": first_hits,
            "first_token_correct_rate": first_hits / n if n else 0.0,
            "full_span_correct": span_hits,
            "full_span_correct_rate": span_hits / n if n else 0.0,
        }
    return summary


def run_source(
    *,
    source: str,
    model,
    tokenizer,
    prompt_field: str,
    max_length: int,
    max_samples: int,
    output_dir: Path,
) -> Dict:
    t0 = time.time()
    examples = build_examples_for_source(source, prompt_field=prompt_field)
    candidate_count = len(examples)
    if max_samples > 0:
        examples = examples[:max_samples]

    evaluated: List[Dict] = []
    skipped = 0
    for example in examples:
        result = evaluate_example_correctness(
            model=model,
            tokenizer=tokenizer,
            example=example,
            max_length=max_length,
        )
        if result is None:
            skipped += 1
            continue
        record = example_to_dict(example)
        record.update(result)
        evaluated.append(record)

    correct = [r for r in evaluated if r["full_span_correct"]]
    first_token_only = [r for r in evaluated if r["first_token_correct"] and not r["full_span_correct"]]

    write_jsonl(output_dir / f"{source}_evaluated_examples.jsonl", evaluated)
    write_jsonl(output_dir / f"{source}_correct_examples.jsonl", correct)
    write_jsonl(output_dir / f"{source}_first_token_only_examples.jsonl", first_token_only)

    n = len(evaluated)
    first_hits = sum(1 for r in evaluated if r["first_token_correct"])
    span_hits = len(correct)

    elapsed = time.time() - t0
    return {
        "source": source,
        "candidate_examples": candidate_count,
        "evaluated_examples": n,
        "skipped_examples": skipped,
        "first_token_correct": first_hits,
        "first_token_correct_rate": first_hits / n if n else 0.0,
        "full_span_correct": span_hits,
        "full_span_correct_rate": span_hits / n if n else 0.0,
        "by_library": summarize_group(evaluated, "library"),
        "by_category": summarize_group(evaluated, "category"),
        "by_task_family": summarize_group(evaluated, "task_family"),
        "elapsed_seconds": round(elapsed, 1),
        "output_files": {
            "evaluated": str((output_dir / f"{source}_evaluated_examples.jsonl").resolve()),
            "correct": str((output_dir / f"{source}_correct_examples.jsonl").resolve()),
            "first_token_only": str((output_dir / f"{source}_first_token_only_examples.jsonl").resolve()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Step 0: select samples the base model predicts correctly.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default="starcoder2_7b")
    parser.add_argument("--sources", nargs="+", default=ALL_SOURCES, choices=ALL_SOURCES)
    parser.add_argument("--prompt-field", type=str, default="version_prompt", choices=["version_prompt", "probing_input"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-samples-per-source", type=int, default=0, help="0 = no cap (full run)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]

    print(f"[select_correct_examples] loading model: {model_name_or_path}", flush=True)
    model, tokenizer = build_model(model_name_or_path)

    per_source_summary = {}
    for source in args.sources:
        print(f"[select_correct_examples] running source={source} ...", flush=True)
        per_source_summary[source] = run_source(
            source=source,
            model=model,
            tokenizer=tokenizer,
            prompt_field=args.prompt_field,
            max_length=max_length,
            max_samples=args.max_samples_per_source,
            output_dir=output_dir,
        )
        s = per_source_summary[source]
        print(
            f"[select_correct_examples] source={source} "
            f"evaluated={s['evaluated_examples']} "
            f"first_token_correct={s['first_token_correct']} ({s['first_token_correct_rate']:.1%}) "
            f"full_span_correct={s['full_span_correct']} ({s['full_span_correct_rate']:.1%}) "
            f"elapsed={s['elapsed_seconds']}s",
            flush=True,
        )

    summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "prompt_field": args.prompt_field,
        "max_length": max_length,
        "max_samples_per_source": args.max_samples_per_source,
        "sources": per_source_summary,
    }
    write_json(output_dir / "selection_summary.json", summary)
    print(f"[select_correct_examples] wrote summary to {output_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()
