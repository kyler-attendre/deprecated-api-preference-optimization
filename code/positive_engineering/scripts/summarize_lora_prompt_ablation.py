#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_MODELS = [
    "starcoder2_3b",
    "starcoder2_7b",
    "starcoder2_15b",
    "deepseek_coder_6_7b_instruct",
    "qwen2_5_coder_3b_instruct",
    "qwen2_5_coder_7b_instruct",
    "qwen2_5_coder_14b_instruct",
]

PROMPT_DIRS = ["original_prompt", "version_prompt"]


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: Optional[Dict], branch: str, name: str) -> Optional[float]:
    if summary is None:
        return None
    value = summary.get(branch, {}).get(name)
    return float(value) if value is not None else None


def build_rows(run_root: Path, models: Iterable[str]) -> List[Dict]:
    rows: List[Dict] = []
    for model in models:
        for prompt_dir in PROMPT_DIRS:
            base_dir = run_root / model / prompt_dir
            with_version = read_json(base_dir / "lora_version_compare" / "comparison_summary.json")
            without_version = read_json(base_dir / "lora_no_version_compare" / "comparison_summary.json")

            if with_version is None and without_version is None:
                continue

            baseline_summary = with_version or without_version
            rows.append(
                {
                    "model": model,
                    "test_prompt": prompt_dir,
                    "baseline_deprecated": metric(baseline_summary, "base", "deprecated_usage_rate"),
                    "baseline_replacement": metric(baseline_summary, "base", "replacement_hit_rate"),
                    "lora_with_version_deprecated": metric(with_version, "lora", "deprecated_usage_rate"),
                    "lora_with_version_replacement": metric(with_version, "lora", "replacement_hit_rate"),
                    "lora_with_version_exact": metric(with_version, "lora", "exact_match_target_rate"),
                    "lora_without_version_deprecated": metric(without_version, "lora", "deprecated_usage_rate"),
                    "lora_without_version_replacement": metric(without_version, "lora", "replacement_hit_rate"),
                    "lora_without_version_exact": metric(without_version, "lora", "exact_match_target_rate"),
                }
            )
    return rows


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def render_markdown(rows: List[Dict]) -> str:
    header = [
        "model",
        "test prompt",
        "baseline dep.",
        "baseline repl.",
        "LoRA w/ version dep.",
        "LoRA w/ version repl.",
        "LoRA w/o version dep.",
        "LoRA w/o version repl.",
        "LoRA w/ version exact",
        "LoRA w/o version exact",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["test_prompt"],
                    fmt(row["baseline_deprecated"]),
                    fmt(row["baseline_replacement"]),
                    fmt(row["lora_with_version_deprecated"]),
                    fmt(row["lora_with_version_replacement"]),
                    fmt(row["lora_without_version_deprecated"]),
                    fmt(row["lora_without_version_replacement"]),
                    fmt(row["lora_with_version_exact"]),
                    fmt(row["lora_without_version_exact"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize the LoRA prompt-field ablation grid.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = build_rows(args.run_root, models)
    markdown = render_markdown(rows)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")

    print(markdown, end="")


if __name__ == "__main__":
    main()
