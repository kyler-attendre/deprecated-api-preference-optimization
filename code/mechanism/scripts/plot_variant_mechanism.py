#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


VARIANT_ORDER = ["official_base", "plain_dpo", "anchored_dpo"]
VARIANT_LABELS = {
    "official_base": "Base",
    "plain_dpo": "DPO",
    "anchored_dpo": "Anchored DPO",
}
VARIANT_COLORS = {
    "official_base": "#4C78A8",
    "plain_dpo": "#F2A541",
    "anchored_dpo": "#E45756",
}
GROUP_TO_MODELS = {
    "starcoder_scale": ["StarCoder2-3B", "StarCoder2-7B", "StarCoder2-15B"],
    "cross_family_same_scale": ["StarCoder2-7B", "Qwen2.5-Coder-7B-Instruct", "DeepSeek-Coder-6.7B-Instruct"],
}
METRICS = [
    ("avg_token_margin", "Final Avg Token Margin"),
    ("log10_perplexity_ratio", "Final log10 PPL Ratio"),
    ("mean_stable_depth_with_fallback", "Stable Decision Depth"),
    ("sequence_win_rate", "Sequence Win Rate"),
]


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["avg_token_margin"] = float(row["avg_token_margin"])
            row["log10_perplexity_ratio"] = float(row["log10_perplexity_ratio"])
            row["mean_stable_depth_with_fallback"] = float(row["mean_stable_depth_with_fallback"])
            row["sequence_win_rate"] = float(row["sequence_win_rate"])
            rows.append(row)
    return rows


def row_lookup(rows, model_label, lens, variant):
    for row in rows:
        if row["model_label"] == model_label and row["lens"] == lens and row["variant"] == variant:
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot grouped variant-aware mechanism summaries.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--lens", choices=["logit_lens", "tuned_lens"], default="logit_lens")
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.summary_csv.resolve())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    width = 0.22

    for row_idx, (group_name, model_order) in enumerate(GROUP_TO_MODELS.items()):
        x = list(range(len(model_order)))
        for col_idx, (metric, title) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for variant_idx, variant in enumerate(VARIANT_ORDER):
                shift = (variant_idx - 1) * width
                xs = [pos + shift for pos in x]
                ys = []
                valid_xs = []
                for pos, model_label in zip(xs, model_order):
                    row = row_lookup(rows, model_label=model_label, lens=args.lens, variant=variant)
                    if row is None:
                        continue
                    valid_xs.append(pos)
                    ys.append(row[metric])
                if valid_xs:
                    ax.bar(
                        valid_xs,
                        ys,
                        width=width,
                        label=VARIANT_LABELS[variant] if row_idx == 0 and col_idx == 0 else None,
                        color=VARIANT_COLORS[variant],
                        edgecolor="white",
                    )
            if metric == "sequence_win_rate":
                ax.set_ylim(0.0, 1.05)
            else:
                ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
            ax.set_title(f"{title}\n{group_name.replace('_', ' ')}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=20, ha="right")
            ax.tick_params(labelsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    print(args.output_path.resolve())


if __name__ == "__main__":
    main()
