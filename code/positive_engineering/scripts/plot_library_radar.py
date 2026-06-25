#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


LIBRARY_ORDER = [
    "numpy",
    "pandas",
    "pytorch",
    "scipy",
    "seaborn",
    "sklearn",
    "tensorflow",
]

DISPLAY_NAMES = {
    "base": "base",
    "dpo": "dpo",
    "anchored_dpo": "anchored_dpo",
}

COLORS = {
    "base": "#4C566A",
    "dpo": "#D08770",
    "anchored_dpo": "#5E81AC",
}


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_library_metrics(rows: Iterable[Dict], libraries: Iterable[str] = LIBRARY_ORDER) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, float]] = {
        library: {"samples": 0, "deprecated_hits": 0, "replacement_hits": 0}
        for library in libraries
    }

    for row in rows:
        library = row.get("library")
        if library not in buckets:
            continue
        buckets[library]["samples"] += 1
        buckets[library]["deprecated_hits"] += int(bool(row.get("has_deprecated")))
        buckets[library]["replacement_hits"] += int(bool(row.get("has_replacement")))

    summary: Dict[str, Dict[str, float]] = {}
    for library in libraries:
        samples = buckets[library]["samples"]
        deprecated_hits = buckets[library]["deprecated_hits"]
        replacement_hits = buckets[library]["replacement_hits"]
        summary[library] = {
            "samples": samples,
            "deprecated_usage_rate": deprecated_hits / samples if samples else 0.0,
            "replacement_hit_rate": replacement_hits / samples if samples else 0.0,
        }
    return summary


def build_radar_series(metrics: Dict[str, Dict[str, float]], metric_name: str, libraries: Iterable[str] = LIBRARY_ORDER) -> List[float]:
    ordered = [metrics[library][metric_name] for library in libraries]
    if ordered:
        ordered.append(ordered[0])
    return ordered


def build_angles(num_axes: int) -> List[float]:
    angles = [2.0 * math.pi * idx / num_axes for idx in range(num_axes)]
    if angles:
        angles.append(angles[0])
    return angles


def parse_prediction_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if label not in DISPLAY_NAMES:
        raise argparse.ArgumentTypeError(f"Unsupported label: {label}")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Prediction file not found: {path}")
    return label, path


def plot_metric_radar(ax, *, title: str, metric_name: str, label_metrics: Dict[str, Dict[str, Dict[str, float]]], libraries: List[str]) -> None:
    angles = build_angles(len(libraries))
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(libraries)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, pad=18)

    for label in ["base", "dpo", "anchored_dpo"]:
        if label not in label_metrics:
            continue
        series = build_radar_series(label_metrics[label], metric_name, libraries=libraries)
        ax.plot(angles, series, color=COLORS[label], linewidth=2, label=DISPLAY_NAMES[label])
        ax.fill(angles, series, color=COLORS[label], alpha=0.08)


def write_summary(
    *,
    output_path: Path,
    model_label: str,
    label_metrics: Dict[str, Dict[str, Dict[str, float]]],
    libraries: List[str],
    prediction_files: Dict[str, str],
) -> None:
    payload = {
        "model_label": model_label,
        "libraries": libraries,
        "prediction_files": prediction_files,
        "metrics": label_metrics,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-library radar charts from prediction jsonl files.")
    parser.add_argument("--prediction", action="append", required=True, help="Prediction file in LABEL=PATH format.")
    parser.add_argument("--model-label", required=True, help="Display label for the model shown in the figure title.")
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    parsed_predictions = dict(parse_prediction_arg(value) for value in args.prediction)
    label_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    prediction_files = {}
    for label, path in parsed_predictions.items():
        label_metrics[label] = summarize_library_metrics(load_jsonl(path))
        prediction_files[label] = str(path)

    libraries = list(LIBRARY_ORDER)
    output_figure = args.output_figure.resolve()
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_summary = args.output_summary.resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
    fig.suptitle(f"{args.model_label}: library-level API-version metrics", y=1.02)
    plot_metric_radar(
        axes[0],
        title="Deprecated Usage Rate",
        metric_name="deprecated_usage_rate",
        label_metrics=label_metrics,
        libraries=libraries,
    )
    plot_metric_radar(
        axes[1],
        title="Replacement Hit Rate",
        metric_name="replacement_hit_rate",
        label_metrics=label_metrics,
        libraries=libraries,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(output_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_summary(
        output_path=output_summary,
        model_label=args.model_label,
        label_metrics=label_metrics,
        libraries=libraries,
        prediction_files=prediction_files,
    )


if __name__ == "__main__":
    main()
