#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = ["base", "dpo", "anchored_dpo"]
VARIANT_LABELS = {
    "base": "Base",
    "dpo": "DPO",
    "anchored_dpo": "Anchored DPO",
}
COLORS = {
    "base": "#b8b8b8",
    "dpo": "#4c78a8",
    "anchored_dpo": "#e15759",
}
PROMPT_ORDER = ["random_neutral_text", "library_neutral_code"]
PROMPT_LABELS = {
    "random_neutral_text": "Random/Neutral Text",
    "library_neutral_code": "Library-Neutral Code",
}


def value(payload, group_name, prompt_type, variant, metric):
    return (
        payload.get("groups", {})
        .get(group_name, {})
        .get(prompt_type, {})
        .get(variant, {})
        .get(metric)
    )


def draw_panel(ax, payload, *, group_name: str, metric: str, title: str):
    x = np.arange(len(PROMPT_ORDER))
    width = 0.22
    offsets = [-width, 0.0, width]
    for idx, variant in enumerate(VARIANT_ORDER):
        ys = []
        for prompt_type in PROMPT_ORDER:
            metric_value = value(payload, group_name, prompt_type, variant, metric)
            ys.append(np.nan if metric_value is None else metric_value)
        ax.bar(x + offsets[idx], ys, width=width, color=COLORS[variant], label=VARIANT_LABELS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[name] for name in PROMPT_ORDER], rotation=12, ha="right")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def main():
    parser = argparse.ArgumentParser(description="Plot ADL aggregate summaries.")
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.aggregate_json.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    draw_panel(
        axes[0, 0],
        payload,
        group_name="starcoder_scale",
        metric="final_layer_mean_diff_norm",
        title="StarCoder2 Scale: Final-Layer ADL Norm",
    )
    draw_panel(
        axes[0, 1],
        payload,
        group_name="starcoder_scale",
        metric="replacement_minus_deprecated_score",
        title="StarCoder2 Scale: Replacement-Deprecated Readout Gap",
    )
    draw_panel(
        axes[1, 0],
        payload,
        group_name="cross_family_same_scale",
        metric="final_layer_mean_diff_norm",
        title="Cross-Family Same Scale: Final-Layer ADL Norm",
    )
    draw_panel(
        axes[1, 1],
        payload,
        group_name="cross_family_same_scale",
        metric="replacement_minus_deprecated_score",
        title="Cross-Family Same Scale: Replacement-Deprecated Readout Gap",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=220, bbox_inches="tight")
    print(str(args.output_file.resolve()))


if __name__ == "__main__":
    main()
