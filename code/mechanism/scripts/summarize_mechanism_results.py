#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427"
OUTPUT_DIR = RESULT_ROOT / "aggregate"

MODEL_LABELS = {
    "starcoder2_3b": "StarCoder2-3B",
    "starcoder2_7b": "StarCoder2-7B",
    "starcoder2_15b": "StarCoder2-15B",
    "deepseek_coder_6_7b_instruct": "DeepSeek-Coder-6.7B-Instruct",
    "qwen2_5_coder_3b_instruct": "Qwen2.5-Coder-3B-Instruct",
    "qwen2_5_coder_7b_instruct": "Qwen2.5-Coder-7B-Instruct",
    "qwen2_5_coder_14b_instruct": "Qwen2.5-Coder-14B-Instruct",
}


def load_summary(model_key: str):
    path = RESULT_ROOT / model_key / "compare" / "run_summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def round4(x):
    return round(float(x), 4)


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
            "median_stable_depth_when_reached": None,
        }
    total_layers = len(rows[0]["layerwise"])
    fallback_depth = total_layers
    depths = [stable_positive_depth(row["layerwise"]) for row in rows]
    reached = [depth for depth in depths if depth is not None]
    filled = [depth if depth is not None else fallback_depth for depth in depths]
    reached_sorted = sorted(reached)
    median = None
    if reached_sorted:
        mid = len(reached_sorted) // 2
        if len(reached_sorted) % 2 == 1:
            median = reached_sorted[mid]
        else:
            median = 0.5 * (reached_sorted[mid - 1] + reached_sorted[mid])
    return {
        "samples": len(rows),
        "stable_depth_reach_rate": round4(len(reached) / len(rows)),
        "mean_stable_depth_when_reached": round4(sum(reached) / len(reached)) if reached else None,
        "mean_stable_depth_with_fallback": round4(sum(filled) / len(filled)),
        "median_stable_depth_when_reached": round4(median) if median is not None else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_keys = sorted(
        [
            p.name
            for p in RESULT_ROOT.iterdir()
            if p.is_dir() and p.name not in {"logs", "aggregate"} and (p / "compare" / "run_summary.json").exists()
        ]
    )

    rows = []
    overall = {}
    per_model = {}
    for model_key in model_keys:
        summary = load_summary(model_key)
        per_model[model_key] = {}
        for lens_key in ["logit_lens", "tuned_lens"]:
            base_last = summary["official_base"][lens_key]["layer_summary"][-1]
            dpo_last = summary["anchored_dpo"][lens_key]["layer_summary"][-1]
            base_nonfinal_jsd_rows = summary["official_base"][lens_key]["layer_summary"][:-1]
            dpo_nonfinal_jsd_rows = summary["anchored_dpo"][lens_key]["layer_summary"][:-1]
            base_depth_stats = compute_depth_stats(summary["official_base"][lens_key]["sample_file"])
            dpo_depth_stats = compute_depth_stats(summary["anchored_dpo"][lens_key]["sample_file"])
            per_model[model_key][lens_key] = {
                "official_base": base_last,
                "anchored_dpo": dpo_last,
                "official_base_depth": base_depth_stats,
                "anchored_dpo_depth": dpo_depth_stats,
                "official_base_nonfinal_mean_jsd_to_final": round4(
                    sum(row["mean_decision_jsd_to_final"] for row in base_nonfinal_jsd_rows)
                    / max(len(base_nonfinal_jsd_rows), 1)
                ),
                "anchored_dpo_nonfinal_mean_jsd_to_final": round4(
                    sum(row["mean_decision_jsd_to_final"] for row in dpo_nonfinal_jsd_rows)
                    / max(len(dpo_nonfinal_jsd_rows), 1)
                ),
                "sequence_margin_delta": round4(
                    dpo_last["mean_sequence_logprob_margin"] - base_last["mean_sequence_logprob_margin"]
                ),
                "first_margin_delta": round4(
                    dpo_last["mean_first_token_logprob_margin"] - base_last["mean_first_token_logprob_margin"]
                ),
                "avg_token_margin_delta": round4(
                    dpo_last["mean_avg_token_logprob_margin"] - base_last["mean_avg_token_logprob_margin"]
                ),
                "sequence_win_rate_delta": round4(
                    dpo_last["replacement_sequence_win_rate"] - base_last["replacement_sequence_win_rate"]
                ),
                "first_win_rate_delta": round4(
                    dpo_last["replacement_first_token_win_rate"] - base_last["replacement_first_token_win_rate"]
                ),
                "stable_depth_reach_rate_delta": round4(
                    dpo_depth_stats["stable_depth_reach_rate"] - base_depth_stats["stable_depth_reach_rate"]
                ),
                "stable_depth_with_fallback_delta": round4(
                    dpo_depth_stats["mean_stable_depth_with_fallback"]
                    - base_depth_stats["mean_stable_depth_with_fallback"]
                ),
            }
            for variant, last, depth_stats in [
                ("official_base", base_last, base_depth_stats),
                ("anchored_dpo", dpo_last, dpo_depth_stats),
            ]:
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": MODEL_LABELS.get(model_key, model_key),
                        "lens": lens_key,
                        "variant": variant,
                        "final_layer": int(last["layer"]),
                        "samples": int(last["samples"]),
                        "sequence_win_rate": round4(last["replacement_sequence_win_rate"]),
                        "first_token_win_rate": round4(last["replacement_first_token_win_rate"]),
                        "sequence_margin": round4(last["mean_sequence_logprob_margin"]),
                        "avg_token_margin": round4(last["mean_avg_token_logprob_margin"]),
                        "geometric_mean_perplexity_ratio": round4(
                            last["geometric_mean_perplexity_ratio_deprecated_over_replacement"]
                        ),
                        "first_token_margin": round4(last["mean_first_token_logprob_margin"]),
                        "replacement_rank": round4(last["mean_replacement_first_token_rank"]),
                        "deprecated_rank": round4(last["mean_deprecated_first_token_rank"]),
                        "stable_depth_reach_rate": depth_stats["stable_depth_reach_rate"],
                        "mean_stable_depth_when_reached": depth_stats["mean_stable_depth_when_reached"],
                        "mean_stable_depth_with_fallback": depth_stats["mean_stable_depth_with_fallback"],
                    }
                )

    for lens_key in ["logit_lens", "tuned_lens"]:
        base_rows = [r for r in rows if r["lens"] == lens_key and r["variant"] == "official_base"]
        dpo_rows = [r for r in rows if r["lens"] == lens_key and r["variant"] == "anchored_dpo"]
        base_nonfinal_jsd_values = [
            per_model[model_key][lens_key]["official_base_nonfinal_mean_jsd_to_final"] for model_key in model_keys
        ]
        dpo_nonfinal_jsd_values = [
            per_model[model_key][lens_key]["anchored_dpo_nonfinal_mean_jsd_to_final"] for model_key in model_keys
        ]
        base_log_ppl_ratio_values = [r["avg_token_margin"] for r in base_rows]
        dpo_log_ppl_ratio_values = [r["avg_token_margin"] for r in dpo_rows]
        overall[lens_key] = {
            "official_base": {
                "avg_sequence_margin": round4(sum(r["sequence_margin"] for r in base_rows) / len(base_rows)),
                "avg_avg_token_margin": round4(sum(r["avg_token_margin"] for r in base_rows) / len(base_rows)),
                "geometric_mean_perplexity_ratio": round4(
                    math.exp(sum(base_log_ppl_ratio_values) / len(base_log_ppl_ratio_values))
                ),
                "avg_first_token_margin": round4(sum(r["first_token_margin"] for r in base_rows) / len(base_rows)),
                "avg_sequence_win_rate": round4(sum(r["sequence_win_rate"] for r in base_rows) / len(base_rows)),
                "avg_first_token_win_rate": round4(sum(r["first_token_win_rate"] for r in base_rows) / len(base_rows)),
                "avg_nonfinal_jsd_to_final": round4(sum(base_nonfinal_jsd_values) / len(base_nonfinal_jsd_values)),
                "avg_stable_depth_reach_rate": round4(
                    sum(r["stable_depth_reach_rate"] for r in base_rows) / len(base_rows)
                ),
                "avg_stable_depth_with_fallback": round4(
                    sum(r["mean_stable_depth_with_fallback"] for r in base_rows) / len(base_rows)
                ),
            },
            "anchored_dpo": {
                "avg_sequence_margin": round4(sum(r["sequence_margin"] for r in dpo_rows) / len(dpo_rows)),
                "avg_avg_token_margin": round4(sum(r["avg_token_margin"] for r in dpo_rows) / len(dpo_rows)),
                "geometric_mean_perplexity_ratio": round4(
                    math.exp(sum(dpo_log_ppl_ratio_values) / len(dpo_log_ppl_ratio_values))
                ),
                "avg_first_token_margin": round4(sum(r["first_token_margin"] for r in dpo_rows) / len(dpo_rows)),
                "avg_sequence_win_rate": round4(sum(r["sequence_win_rate"] for r in dpo_rows) / len(dpo_rows)),
                "avg_first_token_win_rate": round4(sum(r["first_token_win_rate"] for r in dpo_rows) / len(dpo_rows)),
                "avg_nonfinal_jsd_to_final": round4(sum(dpo_nonfinal_jsd_values) / len(dpo_nonfinal_jsd_values)),
                "avg_stable_depth_reach_rate": round4(
                    sum(r["stable_depth_reach_rate"] for r in dpo_rows) / len(dpo_rows)
                ),
                "avg_stable_depth_with_fallback": round4(
                    sum(r["mean_stable_depth_with_fallback"] for r in dpo_rows) / len(dpo_rows)
                ),
            },
        }

    csv_path = OUTPUT_DIR / "final_layer_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_key",
                "model_label",
                "lens",
                "variant",
                "final_layer",
                "samples",
                "sequence_win_rate",
                "first_token_win_rate",
                "sequence_margin",
                "avg_token_margin",
                "geometric_mean_perplexity_ratio",
                "first_token_margin",
                "replacement_rank",
                "deprecated_rank",
                "stable_depth_reach_rate",
                "mean_stable_depth_when_reached",
                "mean_stable_depth_with_fallback",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "models": model_keys,
        "final_layer_rows": rows,
        "per_model": per_model,
        "overall": overall,
    }
    json_path = OUTPUT_DIR / "aggregate_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
