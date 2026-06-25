#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_compare_lora import (
    build_model,
    evaluate_model,
    load_jsonl,
    resolve_existing_path,
)


COARSE_OLD_VERSION_HEADERS = {
    "numpy": "# numpy 1.24.0\nimport numpy\n",
    "pandas": "# pandas 1.5.3\nimport pandas\n",
    "pytorch": "# pytorch 1.13.1\nimport torch\n",
    "scipy": "# scipy 1.10.1\nimport scipy\n",
    "seaborn": "# seaborn 0.12.2\nimport seaborn\n",
    "sklearn": "# scikit-learn 1.3.2\nimport sklearn\n",
    "tensorflow": "# tensorflow 1.15.0\nimport tensorflow\n",
    "transformers": "# transformers 4.30.0\nimport transformers\n",
}


def rewrite_row_with_coarse_old_version(row: Dict) -> Dict:
    library = row["library"]
    header = COARSE_OLD_VERSION_HEADERS[library]
    rewritten = dict(row)
    rewritten["original_version_prompt"] = row["version_prompt"]
    rewritten["version_prompt"] = header + row["probing_input"]
    rewritten["reverse_version_kind"] = "coarse_old_version"
    rewritten["coarse_old_version_header"] = header.strip()
    return rewritten


def build_reverse_subset(rows: Iterable[Dict], base_predictions: Iterable[Dict]) -> List[Dict]:
    base_by_id = {str(row["id"]): row for row in base_predictions}
    subset: List[Dict] = []
    for row in rows:
        base_row = base_by_id.get(str(row["id"]))
        if not base_row or not base_row.get("has_replacement"):
            continue
        if row["library"] not in COARSE_OLD_VERSION_HEADERS:
            continue
        subset.append(rewrite_row_with_coarse_old_version(row))
    return subset


def evaluate_variant(
    *,
    label: str,
    model_name_or_path: str,
    adapter_dir: Optional[Path],
    rows: List[Dict],
    output_dir: Path,
    max_length: int,
    max_new_tokens: int,
) -> Dict:
    model, tokenizer = build_model(model_name_or_path, adapter_dir=adapter_dir)
    summary = evaluate_model(
        label=label,
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        output_path=output_dir / f"{label}_predictions.jsonl",
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        prompt_field="version_prompt",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_adapter_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coarse old-version negative-control prompts.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter in LABEL=PATH format.")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    test_file = resolve_existing_path(args.test_file, label="test file")
    base_predictions_file = resolve_existing_path(args.base_predictions, label="base predictions")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_file)
    base_predictions = load_jsonl(base_predictions_file)
    reverse_rows = build_reverse_subset(rows, base_predictions)
    if args.max_samples > 0:
        reverse_rows = reverse_rows[: args.max_samples]

    reverse_test_file = output_dir / "reverse_version_subset.jsonl"
    with reverse_test_file.open("w", encoding="utf-8") as handle:
        for row in reverse_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {}
    summaries["base"] = evaluate_variant(
        label="base",
        model_name_or_path=args.model_name_or_path,
        adapter_dir=None,
        rows=reverse_rows,
        output_dir=output_dir,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    for label, raw_path in [parse_adapter_arg(item) for item in args.adapter]:
        adapter_dir = resolve_existing_path(raw_path, label=f"{label} adapter dir")
        summaries[label] = evaluate_variant(
            label=label,
            model_name_or_path=args.model_name_or_path,
            adapter_dir=adapter_dir,
            rows=reverse_rows,
            output_dir=output_dir,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "test_file": str(test_file),
        "base_predictions": str(base_predictions_file),
        "reverse_subset_file": str(reverse_test_file),
        "reverse_version_kind": "coarse_old_version",
        "coarse_old_version_headers": COARSE_OLD_VERSION_HEADERS,
        "samples": len(reverse_rows),
        "variants": summaries,
    }
    with (output_dir / "reverse_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
