#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "06_mechanism" / "output" / "full_mechanism_20260427"
OUTPUT_PATH = PROJECT_ROOT / "05_positive_engineering" / "md" / "pic" / "fig3_mechanism_trajectory.png"

MODELS = [
    ("starcoder2_3b",                "StarCoder2-3B"),
    ("starcoder2_7b",                "StarCoder2-7B"),
    ("starcoder2_15b",               "StarCoder2-15B"),
    ("qwen2_5_coder_7b_instruct",    "Qwen2.5-Coder-7B"),
    ("deepseek_coder_6_7b_instruct", "DeepSeek-Coder-6.7B"),
]
LENSES = [
    ("logit_lens", "Logit Lens"),
    ("tuned_lens", "Tuned Lens"),
]
COLORS = {
    "official_base": "#4C78A8",
    "anchored_dpo": "#E45756",
}


def load_summary(model_key: str):
    path = RESULT_ROOT / model_key / "compare" / "run_summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharex=False, sharey=False)

    for col, (model_key, model_label) in enumerate(MODELS):
        summary = load_summary(model_key)
        for row, (lens_key, lens_label) in enumerate(LENSES):
            ax = axes[row][col]
            for variant, label in [("official_base", "Base Model"), ("anchored_dpo", "Anchored DPO")]:
                rows = summary[variant][lens_key]["layer_summary"]
                xs = [item["layer"] for item in rows]
                ys = [item["mean_sequence_logprob_margin"] for item in rows]
                ax.plot(xs, ys, label=label, color=COLORS[variant], linewidth=2.2)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
            ax.set_title(f"{model_label}\n{lens_label}", fontsize=11)
            ax.set_xlabel("Layer", fontsize=10)
            if col == 0:
                ax.set_ylabel("Replacement - Deprecated\nSequence Logprob Margin", fontsize=10)
            ax.tick_params(labelsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
