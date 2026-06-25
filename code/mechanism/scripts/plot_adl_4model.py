#!/usr/bin/env python3
"""
ADL figure for the 4-model scope: StarCoder2 (3B/7B/15B) + DeepSeek-Coder-6.7B.
  Row 0 : StarCoder2 scale  (3-model average)
  Row 1 : DeepSeek-Coder-6.7B-Instruct  (single model)
  Col 0 : Final-layer activation-diff L2 norm
  Col 1 : Replacement-minus-deprecated logit readout score

Base is always 0 by construction (diff with itself), so it is omitted from bars.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_JSON = (
    PROJECT_ROOT
    / "06_mechanism" / "output" / "adl_neutral_20260506"
    / "aggregate" / "adl_aggregate_4model.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "05_positive_engineering" / "md" / "pic"
    / "fig_adl_neutral_4model.png"
)

# Base omitted: always 0 by construction
VARIANT_ORDER  = ["dpo", "anchored_dpo"]
VARIANT_LABELS = {"dpo": "DPO", "anchored_dpo": "Anchored DPO"}
COLORS = {
    "dpo":          "#4c78a8",
    "anchored_dpo": "#e15759",
}

PROMPT_ORDER  = ["random_neutral_text", "library_neutral_code"]
PROMPT_LABELS = {
    "random_neutral_text":  "Random /\nNeutral Text",
    "library_neutral_code": "Library-\nNeutral Code",
}

GROUPS = [
    ("starcoder_scale", "StarCoder2 Scale\n(3B / 7B / 15B avg.)"),
    ("deepseek_6_7b",   "DeepSeek-Coder-\n6.7B-Instruct"),
]

COL_TITLES = [
    "Final-layer ADL Norm (L2)",
    "Replacement - Deprecated\nLogit Readout Score",
]
COL_METRICS = [
    "final_layer_mean_diff_norm",
    "replacement_minus_deprecated_score",
]


def draw_panel(ax, group_data, metric):
    x = np.arange(len(PROMPT_ORDER))
    width = 0.30
    offsets = np.array([-0.5, 0.5]) * width

    for vi, variant in enumerate(VARIANT_ORDER):
        ys = []
        for pt in PROMPT_ORDER:
            val = group_data.get(pt, {}).get(variant, {}).get(metric)
            ys.append(float(val) if val is not None else float("nan"))
        bars = ax.bar(
            x + offsets[vi], ys, width=width,
            color=COLORS[variant],
            label=VARIANT_LABELS[variant],
            edgecolor="white",
            linewidth=0.8,
            alpha=1.0,
            zorder=3,
        )
        # value label on top of each bar
        for bar, y in zip(bars, ys):
            if not np.isnan(y) and y > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{y:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PROMPT_LABELS[p] for p in PROMPT_ORDER],
        fontsize=9, ha="center",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    payload = json.loads(AGGREGATE_JSON.read_text(encoding="utf-8"))

    plt.style.use("default")
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    for ri, (group_key, group_label) in enumerate(GROUPS):
        group_data = payload["groups"].get(group_key, {})
        for ci, (metric_key, col_title) in enumerate(zip(COL_METRICS, COL_TITLES)):
            ax = axes[ri][ci]
            draw_panel(ax, group_data, metric_key)

            # column title only on top row
            if ri == 0:
                ax.set_title(col_title, fontsize=11, pad=10, fontweight="medium")

            # row label only on left column
            if ci == 0:
                ax.set_ylabel(group_label, fontsize=9, labelpad=6)

    # shared legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2,
        frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
