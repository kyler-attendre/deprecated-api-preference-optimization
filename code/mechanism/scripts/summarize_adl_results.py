#!/usr/bin/env python3
import argparse
import csv
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

from src.adl_compare import build_adl_rows, mean  # noqa: E402
from src.variant_compare import model_label_for  # noqa: E402


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize ADL experiment outputs across models.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for run_summary_path in sorted(input_root.glob("*/run_summary.json")):
        payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
        rows.extend(build_adl_rows(payload["model_key"], payload))

    write_csv(output_dir / "adl_summary_rows.csv", rows)

    grouped = {}
    for group_name, predicate in {
        "starcoder_scale": lambda row: row["model_group_starcoder_scale"],
        "cross_family_same_scale": lambda row: row["model_group_cross_family_same_scale"],
    }.items():
        grouped[group_name] = {}
        for prompt_type in sorted({row["prompt_type"] for row in rows}):
            grouped[group_name][prompt_type] = {}
            for variant in sorted({row["variant"] for row in rows}):
                bucket = [
                    row
                    for row in rows
                    if predicate(row) and row["prompt_type"] == prompt_type and row["variant"] == variant
                ]
                if not bucket:
                    continue
                grouped[group_name][prompt_type][variant] = {
                    "models": len(bucket),
                    "model_labels": [model_label_for(row["model_key"]) for row in bucket],
                    "final_layer_mean_diff_norm": mean([row["final_layer_mean_diff_norm"] for row in bucket]),
                    "replacement_family_score": mean([row["replacement_family_score"] for row in bucket]),
                    "deprecated_family_score": mean([row["deprecated_family_score"] for row in bucket]),
                    "library_version_family_score": mean([row["library_version_family_score"] for row in bucket]),
                    "replacement_minus_deprecated_score": mean(
                        [row["replacement_minus_deprecated_score"] for row in bucket]
                    ),
                }

    aggregate = {
        "rows_file": str((output_dir / "adl_summary_rows.csv").resolve()),
        "groups": grouped,
    }
    (output_dir / "adl_aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
