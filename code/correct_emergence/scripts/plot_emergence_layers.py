#!/usr/bin/env python3
"""Step 2 (part 2): visualizations for the layer-wise emergence analysis.

Reads `topk_trajectory.jsonl` + `cross_sample_analysis.json` +
`emergence_summary.json` (produced by `run_topk_trajectory.py` /
`summarize_emergence.py`) and renders four figures into
`output/{run}/figures/`:

  fig1_emergence_layer_histogram.png   -- where in the network correct APIs emerge
                                          (logit lens vs tuned lens, top-5/top-10)
  fig2_emergence_by_library.png        -- emergence_layer / saturation_layer by library
  fig3_competitor_profile.png          -- what kind of tokens compete pre-emergence
  fig4_api_in_context.png              -- "copy"-type vs "recall"-type emergence

No GPU needed -- pure post-hoc plotting over the persisted aggregates.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_07 = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_ROOT_07.parent
MECH_SRC_DIR = PROJECT_ROOT / "06_mechanism" / "src"

for path in (MECH_SRC_DIR, PROJECT_ROOT_07):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.correct_selection import load_jsonl_rows  # noqa: E402

LENS_COLORS = {"logit_lens": "#4C78A8", "tuned_lens": "#E45756"}
LENS_LABELS = {"logit_lens": "Logit Lens", "tuned_lens": "Tuned Lens"}
LENSES = ["logit_lens", "tuned_lens"]
NUM_LAYERS = 32


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 1: emergence layer histogram
# ---------------------------------------------------------------------------


def plot_emergence_histogram(trajectory: List[Dict], cross_sample: Dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    bins = np.arange(-0.5, NUM_LAYERS + 1.5, 1)
    thresholds = cross_sample["bucket_thresholds_emergence_layer_top10"]

    for ax, k in zip(axes, (5, 10)):
        for lens in LENSES:
            values = [r[f"emergence_layer_top{k}"] for r in trajectory if r["lens"] == lens and r[f"emergence_layer_top{k}"] is not None]
            ax.hist(values, bins=bins, alpha=0.55, label=LENS_LABELS[lens], color=LENS_COLORS[lens])
        ax.axvline(thresholds["low"], color="#666666", linestyle="--", linewidth=1, label="bucket boundary (top-10)" if k == 5 else None)
        ax.axvline(thresholds["high"], color="#666666", linestyle="--", linewidth=1)
        ax.set_title(f"Top-{k} entry layer", fontsize=11)
        ax.set_xlabel("Layer (0 = first decoder layer, 31 = last)", fontsize=9)
        ax.set_xlim(-1, NUM_LAYERS)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("# correct examples", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Layer distribution at which the correct token first enters the top-k", fontsize=12, y=1.13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


# ---------------------------------------------------------------------------
# Fig 2: emergence / saturation by library
# ---------------------------------------------------------------------------


def plot_by_library(cross_sample: Dict, output_path: Path):
    libraries = list(cross_sample["by_lens"]["logit_lens"]["by_library"].keys())
    metrics = [("emergence_layer_top10", "Top-10 entry layer"), ("saturation_layer", "Stable decision depth")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.35
    x = np.arange(len(libraries))
    for ax, (metric, label) in zip(axes, metrics):
        for offset, lens in zip((-width / 2, width / 2), LENSES):
            stats_by_lib = cross_sample["by_lens"][lens]["by_library"]
            means = [stats_by_lib[lib][metric].get("mean", float("nan")) for lib in libraries]
            counts = [stats_by_lib[lib]["count"] for lib in libraries]
            bars = ax.bar(x + offset, means, width=width, color=LENS_COLORS[lens], alpha=0.8, label=LENS_LABELS[lens])
            for bar, n in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"n={n}", ha="center", fontsize=7, color="#444444")
        ax.set_xticks(x)
        ax.set_xticklabels(libraries, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Layer", fontsize=10)
        ax.tick_params(labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Mean top-10 entry layer and stable decision depth by library (sample counts annotated)", fontsize=12, y=1.16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


# ---------------------------------------------------------------------------
# Fig 3: competitor token-class profile
# ---------------------------------------------------------------------------


def plot_competitor_profile(emergence_summary: Dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    classes = sorted({
        cls
        for lens in LENSES
        for cls in emergence_summary["by_lens"][lens]["competitor_token_class_distribution"]
    })
    x = np.arange(len(classes))
    width = 0.35
    for ax_idx, lens in enumerate(LENSES):
        ax = axes[ax_idx]
        dist = emergence_summary["by_lens"][lens]["competitor_token_class_distribution"]
        counts = [dist.get(cls, 0) for cls in classes]
        ax.bar(x, counts, width=width * 1.6, color=LENS_COLORS[lens], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=9, rotation=20, ha="right")
        ax.set_title(LENS_LABELS[lens], fontsize=11)
        ax.set_ylabel("# (token, example) appearances\nbefore the top-10 entry layer", fontsize=8.5)
        ax.tick_params(labelsize=9)
    fig.suptitle("Competitor token-type distribution before the top-10 entry layer", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


# ---------------------------------------------------------------------------
# Fig 4: "copy"-type vs "recall"-type emergence
# ---------------------------------------------------------------------------


def plot_api_in_context(cross_sample: Dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = ["in_context", "not_in_context"]
    group_labels = ["API form\nin prompt", "API form\nnot in prompt"]
    metrics = [("emergence_layer_top10", "Top-10 entry layer"), ("saturation_layer", "Stable decision depth")]

    x = np.arange(len(groups))
    width = 0.35
    for ax, (metric, label) in zip(axes, metrics):
        for offset, lens in zip((-width / 2, width / 2), LENSES):
            stats = cross_sample["by_lens"][lens]["by_api_in_context"]
            means = [stats[g][metric].get("mean", float("nan")) for g in groups]
            counts = [stats[g]["count"] for g in groups]
            bars = ax.bar(x + offset, means, width=width, color=LENS_COLORS[lens], alpha=0.8, label=LENS_LABELS[lens])
            for bar, n in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"n={n}", ha="center", fontsize=7, color="#444444")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8.5)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Layer", fontsize=10)
        ax.tick_params(labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle(
        f"Top-10 entry layer and stable decision depth by API-in-context status "
        f"(API form already present in prompt: {cross_sample['api_in_context']['hit_count']}/{cross_sample['n_examples']} examples)",
        fontsize=11.5, y=1.2,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Step 2: render emergence-analysis figures.")
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT_07 / "output" / "starcoder2_7b")
    parser.add_argument("--figures-dir", type=Path, default=None, help="defaults to <analysis-dir>/figures")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir if args.analysis_dir.is_absolute() else (Path.cwd() / args.analysis_dir)
    analysis_dir = analysis_dir.resolve()
    figures_dir = args.figures_dir or (analysis_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_jsonl_rows(analysis_dir / "topk_trajectory.jsonl")
    cross_sample = load_json(analysis_dir / "cross_sample_analysis.json")
    # `emergence_summary.json` is `run_topk_trajectory.py`'s full run summary;
    # the per-lens aggregates (incl. competitor profiles) live under its
    # `emergence_summary` key.
    emergence_summary = load_json(analysis_dir / "emergence_summary.json")["emergence_summary"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_emergence_histogram(trajectory, cross_sample, figures_dir / "fig1_emergence_layer_histogram.png")
    plot_by_library(cross_sample, figures_dir / "fig2_emergence_by_library.png")
    plot_competitor_profile(emergence_summary, figures_dir / "fig3_competitor_profile.png")
    plot_api_in_context(cross_sample, figures_dir / "fig4_api_in_context.png")


if __name__ == "__main__":
    main()
