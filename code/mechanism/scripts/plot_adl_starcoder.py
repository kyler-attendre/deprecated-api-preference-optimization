#!/usr/bin/env python3
"""
ADL figure: StarCoder2 (3B / 7B / 15B) only, 1-row × 2-col.
  Col 0 : Final-layer activation-diff L2 norm
  Col 1 : Replacement-minus-deprecated logit readout score
Base omitted (always 0 by construction).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_JSON = (
    PROJECT_ROOT
    / "06_mechanism" / "output" / "adl_neutral_20260506"
    / "aggregate" / "adl_aggregate_starcoder.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "05_positive_engineering" / "md" / "pic"
    / "fig_adl_neutral_starcoder.png"
)

VARIANT_ORDER  = ["dpo", "anchored_dpo"]
VARIANT_LABELS = {"dpo": "DPO", "anchored_dpo": "Anchored DPO"}
COLORS = {"dpo": "#4c78a8", "anchored_dpo": "#e15759"}

PROMPT_ORDER  = ["random_neutral_text", "library_neutral_code"]
PROMPT_LABELS = {
    "random_neutral_text":  "Random /\nNeutral Text",
    "library_neutral_code": "Library-\nNeutral Code",
}

COL_TITLES  = ["Final-layer ADL Norm (L2)", "Replacement - Deprecated\nLogit Readout Score"]
COL_METRICS = ["final_layer_mean_diff_norm", "replacement_minus_deprecated_score"]


def draw_panel(ax, group_data, metric, col_title):
    x = np.arange(len(PROMPT_ORDER))
    width = 0.30
    offsets = np.array([-0.5, 0.5]) * width

    for vi, variant in enumerate(VARIANT_ORDER):
        ys = [float(group_data.get(pt, {}).get(variant, {}).get(metric) or 0)
              for pt in PROMPT_ORDER]
        bars = ax.bar(x + offsets[vi], ys, width=width,
                      color=COLORS[variant], label=VARIANT_LABELS[variant],
                      edgecolor="white", linewidth=0.8, alpha=1.0, zorder=3)
        for bar, y in zip(bars, ys):
            if y > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{y:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#333333")

    ax.set_title(col_title, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=9, ha="center")
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    payload = json.loads(AGGREGATE_JSON.read_text(encoding="utf-8"))
    group_data = payload["groups"]["starcoder_scale"]

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"wspace": 0.35})

    for ci, (metric, col_title) in enumerate(zip(COL_METRICS, COL_TITLES)):
        draw_panel(axes[ci], group_data, metric, col_title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("StarCoder2 Scale (3B / 7B / 15B avg.)", fontsize=11, y=1.10)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
