#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


LIBRARY_ORDER = [
    "numpy",
    "pandas",
    "pytorch",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorflow",
]

LABEL_ORDER = ["base", "dpo", "anchored_dpo"]
COLORS = {
    "base": "#B0B7C3",
    "dpo": "#2B6CB0",
    "anchored_dpo": "#C53030",
}
LINESTYLES = {
    "base": "-",
    "dpo": "--",
    "anchored_dpo": "-",
}
DISPLAY_LABELS = {
    "base": "Base",
    "dpo": "DPO",
    "anchored_dpo": "SFT+DPO",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_angles(num_axes: int) -> List[float]:
    angles = [2.0 * math.pi * idx / num_axes for idx in range(num_axes)]
    angles.append(angles[0])
    return angles


def collect_panel_series(payloads: Dict[str, Dict], *, metric_name: str, libraries: List[str]) -> Dict[str, Dict[str, List[float]]]:
    panel = {}
    for model_key, payload in payloads.items():
        model_series = {}
        for label in LABEL_ORDER:
            model_series[label] = [payload["metrics"][label][library][metric_name] for library in libraries]
        panel[model_key] = model_series
    return panel


def plot_model_panel(ax, *, model_title: str, payload: Dict, metric_name: str, libraries: List[str]) -> None:
    angles = build_angles(len(libraries))
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(libraries, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(model_title, pad=16)
    for label in LABEL_ORDER:
        series = [payload["metrics"][label][library][metric_name] for library in libraries]
        series.append(series[0])
        ax.plot(
            angles,
            series,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=2.2,
            label=DISPLAY_LABELS[label],
        )
        ax.fill(angles, series, color=COLORS[label], alpha=0.08)


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Plot a 1x3 panel of library radar charts for StarCoder2 scales.")
    parser.add_argument("--summary-3b", type=Path, required=True)
    parser.add_argument("--summary-7b", type=Path, required=True)
    parser.add_argument("--summary-15b", type=Path, required=True)
    parser.add_argument("--metric-name", choices=["deprecated_usage_rate", "replacement_hit_rate"], required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    payloads = {
        "StarCoder2-3B": load_json(args.summary_3b.resolve()),
        "StarCoder2-7B": load_json(args.summary_7b.resolve()),
        "StarCoder2-15B": load_json(args.summary_15b.resolve()),
    }
    libraries = list(LIBRARY_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})
    metric_title = "Deprecated Usage Rate" if args.metric_name == "deprecated_usage_rate" else "Replacement Hit Rate"
    fig.suptitle(f"StarCoder2 scale comparison: {metric_title}", y=1.04)
    for ax, (title, payload) in zip(axes, payloads.items()):
        plot_model_panel(ax, model_title=title, payload=payload, metric_name=args.metric_name, libraries=libraries)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    output_figure = args.output_figure.resolve()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
