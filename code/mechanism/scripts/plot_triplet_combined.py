#!/usr/bin/env python3
"""
Combined 2×2 figure for §5.6 triplet mechanism comparison.

Left column  : Logit Lens avg_token_margin  (base / dpo / anchored_dpo)
Right column : Tuned Lens stable_decision_depth (base / anchored_dpo only)

Row 0 : StarCoder2 scale (3B / 7B / 15B)
Row 1 : Cross-family same scale (StarCoder2-7B / Qwen2.5-Coder-7B / DeepSeek-Coder-6.7B)
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIPLET_CSV = PROJECT_ROOT / "06_mechanism" / "output" / "triplet_mechanism_20260506" / "aggregate" / "final_layer_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "05_positive_engineering" / "md" / "pic" / "fig_triplet_combined_20260507.png"

GROUPS = {
    "starcoder_scale": ["StarCoder2-3B", "StarCoder2-7B", "StarCoder2-15B"],
    "cross_family_same_scale": ["StarCoder2-7B", "Qwen2.5-Coder-7B-Instruct", "DeepSeek-Coder-6.7B-Instruct"],
}
GROUP_TITLES = {
    "starcoder_scale": "Same Family · Varying Scale",
    "cross_family_same_scale": "Same Scale · Cross Family",
}

LOGIT_VARIANTS = ["official_base", "plain_dpo", "anchored_dpo"]
TUNED_VARIANTS = ["official_base", "anchored_dpo"]

VARIANT_LABELS = {
    "official_base": "Base",
    "plain_dpo": "DPO",
    "anchored_dpo": "SFT+DPO",
}
VARIANT_COLORS = {
    "official_base": "#4C78A8",
    "plain_dpo": "#F2A541",
    "anchored_dpo": "#E45756",
}


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["avg_token_margin"] = float(row["avg_token_margin"])
            row["mean_stable_depth_with_fallback"] = float(row["mean_stable_depth_with_fallback"])
            rows.append(row)
    return rows


def lookup(rows, model_label, lens, variant):
    for row in rows:
        if row["model_label"] == model_label and row["lens"] == lens and row["variant"] == variant:
            return row
    return None


def plot_col(ax, rows, group_models, lens, variants, metric, title, ylabel, draw_zero=False):
    x = list(range(len(group_models)))
    width = 0.22
    n = len(variants)
    for vi, variant in enumerate(variants):
        shift = (vi - (n - 1) / 2) * width
        xs, ys = [], []
        for xi, model in enumerate(group_models):
            row = lookup(rows, model, lens, variant)
            if row is None:
                continue
            xs.append(xi + shift)
            ys.append(row[metric])
        if xs:
            ax.bar(xs, ys, width=width,
                   label=VARIANT_LABELS[variant],
                   color=VARIANT_COLORS[variant],
                   edgecolor="white")
    if draw_zero:
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_models, rotation=18, ha="right", fontsize=8)
    ax.tick_params(labelsize=9)


def main():
    rows = load_rows(TRIPLET_CSV)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    col_configs = [
        # (lens, variants, metric, col_title, ylabel, draw_zero)
        ("logit_lens", LOGIT_VARIANTS, "avg_token_margin",
         "Logit Lens · Avg Token Margin\n(replacement − deprecated, higher = more replacement-preferring)",
         "Avg Token Margin", True),
        ("tuned_lens", TUNED_VARIANTS, "mean_stable_depth_with_fallback",
         "Tuned Lens · Stable Decision Depth\n(lower layer = earlier commitment to replacement)",
         "Stable Decision Depth", False),
    ]

    for row_idx, (group_key, group_models) in enumerate(GROUPS.items()):
        row_label = GROUP_TITLES[group_key]
        for col_idx, (lens, variants, metric, col_title, ylabel, draw_zero) in enumerate(col_configs):
            ax = axes[row_idx][col_idx]
            title = f"{col_title}" if row_idx == 0 else ""
            plot_col(ax, rows, group_models, lens, variants, metric, title, ylabel, draw_zero)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}", fontsize=9)

    # Shared legend from the first axis that has all logit variants
    handles, labels = axes[0][0].get_legend_handles_labels()
    tuned_handles, tuned_labels = axes[0][1].get_legend_handles_labels()
    all_handles = handles + [h for h, l in zip(tuned_handles, tuned_labels) if l not in labels]
    all_labels = labels + [l for l in tuned_labels if l not in labels]
    fig.legend(all_handles, all_labels, loc="upper center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
