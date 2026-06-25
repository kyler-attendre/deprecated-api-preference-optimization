#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MECH_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MECH_ROOT.parent
SRC_ROOT = MECH_ROOT
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.variant_compare import build_variant_rows, model_label_for  # noqa: E402


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def round4(value):
    return round(float(value), 4)


def stable_positive_depth(layerwise_rows, margin_key: str = "sequence_logprob_margin"):
    margins = [float(row[margin_key]) for row in layerwise_rows]
    for idx in range(len(margins)):
        if all(value > 0.0 for value in margins[idx:]):
            return idx
    return None


def compute_depth_stats(sample_file: str) -> dict:
    rows = load_jsonl(Path(sample_file))
    if not rows:
        return {
            "samples": 0,
            "stable_depth_reach_rate": 0.0,
            "mean_stable_depth_when_reached": None,
            "mean_stable_depth_with_fallback": None,
        }
    total_layers = len(rows[0]["layerwise"])
    fallback_depth = total_layers
    depths = [stable_positive_depth(row["layerwise"]) for row in rows]
    reached = [depth for depth in depths if depth is not None]
    filled = [depth if depth is not None else fallback_depth for depth in depths]
    return {
        "samples": len(rows),
        "stable_depth_reach_rate": round4(len(reached) / len(rows)),
        "mean_stable_depth_when_reached": round4(sum(reached) / len(reached)) if reached else None,
        "mean_stable_depth_with_fallback": round4(sum(filled) / len(filled)),
    }


def enrich_run_summary(summary: dict) -> dict:
    enriched = {"variants": {}}
    for variant, lens_payloads in (summary.get("variants") or {}).items():
        enriched["variants"][variant] = {}
        for lens, payload in lens_payloads.items():
            enriched["variants"][variant][lens] = {
                **payload,
                "depth_stats": compute_depth_stats(payload["sample_file"]),
            }
    return enriched


def aggregate_rows(rows, selector):
    chosen = [row for row in rows if selector(row)]
    if not chosen:
        return None
    return {
        "models": len({row["model_key"] for row in chosen}),
        "avg_sequence_margin": round4(sum(row["sequence_margin"] for row in chosen) / len(chosen)),
        "avg_avg_token_margin": round4(sum(row["avg_token_margin"] for row in chosen) / len(chosen)),
        "avg_log10_perplexity_ratio": round4(sum(row["log10_perplexity_ratio"] for row in chosen) / len(chosen)),
        "avg_first_token_margin": round4(sum(row["first_token_margin"] for row in chosen) / len(chosen)),
        "avg_sequence_win_rate": round4(sum(row["sequence_win_rate"] for row in chosen) / len(chosen)),
        "avg_first_token_win_rate": round4(sum(row["first_token_win_rate"] for row in chosen) / len(chosen)),
        "avg_stable_depth_reach_rate": round4(sum(row["stable_depth_reach_rate"] for row in chosen) / len(chosen)),
        "avg_stable_depth_with_fallback": round4(
            sum(row["mean_stable_depth_with_fallback"] for row in chosen) / len(chosen)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize variant-aware mechanism results.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root if args.result_root.is_absolute() else (Path.cwd() / args.result_root)
    result_root = result_root.resolve()
    output_dir = args.output_dir if args.output_dir else (result_root / "aggregate")
    output_dir = output_dir if output_dir.is_absolute() else (Path.cwd() / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_model = {}
    for model_dir in sorted(result_root.iterdir()):
        run_summary_path = model_dir / "compare" / "run_summary.json"
        if not run_summary_path.exists():
            continue
        run_summary = load_json(run_summary_path)
        enriched = enrich_run_summary(run_summary)
        model_key = run_summary["model_key"]
        model_label = model_label_for(model_key)
        model_rows = build_variant_rows(model_key=model_key, model_label=model_label, run_summary=enriched)
        for row in model_rows:
            row["log10_perplexity_ratio"] = round4(row["avg_token_margin"] / math.log(10.0))
        rows.extend(model_rows)
        per_model[model_key] = model_rows

    csv_path = output_dir / "final_layer_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    overall = {}
    variants = sorted({row["variant"] for row in rows})
    lenses = sorted({row["lens"] for row in rows})
    selectors = {
        "all_models": lambda row: True,
        "starcoder_scale": lambda row: row["model_group_starcoder_scale"],
        "cross_family_same_scale": lambda row: row["model_group_cross_family_same_scale"],
    }
    for group_name, group_selector in selectors.items():
        overall[group_name] = {}
        for lens in lenses:
            overall[group_name][lens] = {}
            for variant in variants:
                overall[group_name][lens][variant] = aggregate_rows(
                    rows,
                    selector=lambda row, lens=lens, variant=variant, group_selector=group_selector: (
                        row["lens"] == lens and row["variant"] == variant and group_selector(row)
                    ),
                )

    payload = {
        "result_root": str(result_root),
        "rows": rows,
        "overall": overall,
        "per_model": per_model,
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(output_dir)


if __name__ == "__main__":
    main()
