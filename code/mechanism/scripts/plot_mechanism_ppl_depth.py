#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427" / "aggregate" / "final_layer_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "05_positive_engineering" / "md" / "pic" / "fig6_mechanism_ppl_depth.png"
LENS_ORDER = ["logit_lens", "tuned_lens"]
LENS_LABELS = {"logit_lens": "Logit Lens", "tuned_lens": "Tuned Lens"}
VARIANT_ORDER = ["official_base", "anchored_dpo"]
VARIANT_LABELS = {"official_base": "Base", "anchored_dpo": "Anchored DPO"}
COLORS = {
    "official_base": "#4C78A8",
    "anchored_dpo": "#E45756",
}


def load_rows():
    rows = []
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in [
                "geometric_mean_perplexity_ratio",
                "avg_token_margin",
                "stable_depth_reach_rate",
                "mean_stable_depth_with_fallback",
            ]:
                row[key] = float(row[key])
            row["log10_geometric_mean_perplexity_ratio"] = row["avg_token_margin"] / math.log(10.0)
            rows.append(row)
    return rows


def get_model_order(rows):
    seen = []
    for row in rows:
        label = row["model_label"]
        if label not in seen:
            seen.append(label)
    return seen


def values(rows, *, lens, variant, metric, model_order):
    mapping = {
        row["model_label"]: row[metric]
        for row in rows
        if row["lens"] == lens and row["variant"] == variant
    }
    return [mapping[name] for name in model_order]


def main():
    rows = load_rows()
    model_order = get_model_order(rows)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = list(range(len(model_order)))
    width = 0.18
    n_lenses = len(LENS_ORDER)
    n_variants = len(VARIANT_ORDER)
    total_width = n_lenses * n_variants * width
    group_offset = total_width / 2

    for lens_idx, lens in enumerate(LENS_ORDER):
        for variant_idx, variant in enumerate(VARIANT_ORDER):
            shift = -group_offset + (lens_idx * n_variants + variant_idx) * width + width / 2
            xs = [v + shift for v in x]
            ys = values(rows, lens=lens, variant=variant, metric="log10_geometric_mean_perplexity_ratio", model_order=model_order)
            label = f"{LENS_LABELS[lens]} / {VARIANT_LABELS[variant]}"
            color = COLORS[variant]
            alpha = 0.75 if lens == "logit_lens" else 1.0
            hatch = "" if lens == "logit_lens" else "//"
            ax.bar(xs, ys, width=width, label=label, color=color, alpha=alpha, hatch=hatch, edgecolor="white")

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title("Final log10[ PPL(deprecated) / PPL(replacement) ]\n(positive = deprecated less natural per token)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=20, ha="right")
    ax.tick_params(labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
