#!/usr/bin/env python3
"""
Plot side-by-side Frobenius norm heatmaps for DPO and SFT+DPO delta weights.
Used in §5.9.1.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "output" / "lora_delta_20260509" / "starcoder2_7b"
OUTPUT_PATH = PROJECT_ROOT / "md" / "pic" / "fig_delta_fro_heatmap.png"

MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj"]
MODULE_LABELS = ["q", "k", "v", "o"]
NUM_LAYERS = 32

VARIANT_FILES = {
    "DPO": DATA_DIR / "dpo_module_stats.csv",
    "SFT+DPO": DATA_DIR / "anchored_dpo_module_stats.csv",
}


def load_matrix(csv_path: Path) -> np.ndarray:
    """Returns shape (NUM_LAYERS, 4) matrix of fro_norm values."""
    data = {m: {} for m in MODULE_ORDER}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            module = row["module"]
            if module in data:
                data[module][layer] = float(row["fro_norm"])
    mat = np.zeros((NUM_LAYERS, len(MODULE_ORDER)))
    for ci, mod in enumerate(MODULE_ORDER):
        for layer in range(NUM_LAYERS):
            mat[layer, ci] = data[mod].get(layer, 0.0)
    return mat


def main():
    matrices = {label: load_matrix(path) for label, path in VARIANT_FILES.items()}

    # shared color scale
    vmin = min(m.min() for m in matrices.values())
    vmax = max(m.max() for m in matrices.values())

    fig, axes = plt.subplots(1, 2, figsize=(9, 9), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, (label, mat) in zip(axes, matrices.items()):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xticks(range(len(MODULE_ORDER)))
        ax.set_xticklabels(MODULE_LABELS, fontsize=11)
        ax.set_xlabel("Projection", fontsize=10)

        # mark the 17-31 boundary
        ax.axhline(16.5, color="steelblue", linewidth=1.2, linestyle="--", alpha=0.7)

    # y-axis on the left plot only
    axes[0].set_yticks(range(0, NUM_LAYERS, 4))
    axes[0].set_yticklabels(range(0, NUM_LAYERS, 4), fontsize=9)
    axes[0].set_ylabel("Layer", fontsize=10)

    # annotation for the boundary
    for ax in axes:
        ax.text(
            3.55, 16.5, "layer 17",
            va="center", ha="right", fontsize=7.5, color="steelblue",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1),
        )

    # colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.04, shrink=0.85)
    cbar.set_label("Frobenius Norm of ΔW", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
