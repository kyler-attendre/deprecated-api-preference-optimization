#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427" / "aggregate" / "final_layer_summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "05_positive_engineering" / "md" / "pic" / "fig4_mechanism_overview.png"
LENS_ORDER = ["logit_lens", "tuned_lens"]
VARIANT_ORDER = ["official_base", "anchored_dpo"]
COLORS = {
    "official_base": "#4C78A8",
    "anchored_dpo": "#E45756",
}


def load_rows():
    rows = []
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sequence_margin"] = float(row["sequence_margin"])
            row["first_token_margin"] = float(row["first_token_margin"])
            row["sequence_win_rate"] = float(row["sequence_win_rate"])
            row["first_token_win_rate"] = float(row["first_token_win_rate"])
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)

    metrics = [
        ("sequence_margin", "Final Sequence Margin", axes[0][0]),
        ("first_token_margin", "Final First-Token Margin", axes[0][1]),
        ("sequence_win_rate", "Final Sequence Win Rate", axes[1][0]),
        ("first_token_win_rate", "Final First-Token Win Rate", axes[1][1]),
    ]

    x = list(range(len(model_order)))
    width = 0.18

    for metric, title, ax in metrics:
        for lens_idx, lens in enumerate(LENS_ORDER):
            offset_base = (-0.5 + lens_idx) * 2 * width
            for variant_idx, variant in enumerate(VARIANT_ORDER):
                shift = offset_base + (variant_idx * width)
                xs = [v + shift for v in x]
                ys = values(rows, lens=lens, variant=variant, metric=metric, model_order=model_order)
                label = f"{'Logit' if lens == 'logit_lens' else 'Tuned'} / {'Base' if variant == 'official_base' else 'DPO'}"
                color = COLORS[variant]
                alpha = 0.75 if lens == "logit_lens" else 1.0
                hatch = "" if lens == "logit_lens" else "//"
                ax.bar(xs, ys, width=width, label=label, color=color, alpha=alpha, hatch=hatch, edgecolor="white")
        if "Win Rate" in title:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=9)

    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20, ha="right")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
