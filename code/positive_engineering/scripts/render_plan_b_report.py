#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def model_label_from_path(path: Path) -> str:
    name = path.parent.name
    if "3b" in name.lower():
        return "StarCoder2-3B"
    if "7b" in name.lower():
        return "StarCoder2-7B"
    if "15b" in name.lower():
        return "StarCoder2-15B"
    return name


def extract_row(model_label: str, summary: Dict) -> Dict:
    prompt_only = summary["prompt_only"]
    rerank = summary["rerank"]
    delta = summary["delta"]
    return {
        "model": model_label,
        "samples": rerank["samples"],
        "prompt_only_deprecated": prompt_only["deprecated_usage_rate"],
        "rerank_deprecated": rerank["deprecated_usage_rate"],
        "delta_deprecated": delta["deprecated_usage_rate"],
        "prompt_only_replacement": prompt_only["replacement_hit_rate"],
        "rerank_replacement": rerank["replacement_hit_rate"],
        "delta_replacement": delta["replacement_hit_rate"],
        "rerank_exact": rerank["exact_match_target_rate"],
        "rerank_nonempty": rerank["nonempty_prediction_rate"],
        "rerank_applied_steps": rerank.get("avg_rerank_applied_steps"),
        "rerank_changed_steps": rerank.get("avg_rerank_changed_steps"),
    }


def build_results_table(rows: List[Dict]) -> str:
    header = (
        "| Model | Samples | Prompt-Only Deprecated | Rerank Deprecated | Delta Deprecated | "
        "Prompt-Only Replacement | Rerank Replacement | Delta Replacement | "
        "Rerank Exact Match | Rerank Nonempty | Avg Applied Steps | Avg Changed Steps |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body = []
    for row in rows:
        body.append(
            "| {model} | {samples} | {prompt_only_deprecated} | {rerank_deprecated} | {delta_deprecated} | "
            "{prompt_only_replacement} | {rerank_replacement} | {delta_replacement} | "
            "{rerank_exact} | {rerank_nonempty} | {rerank_applied_steps} | {rerank_changed_steps} |".format(
                model=row["model"],
                samples=row["samples"],
                prompt_only_deprecated=format_metric(row["prompt_only_deprecated"]),
                rerank_deprecated=format_metric(row["rerank_deprecated"]),
                delta_deprecated=format_metric(row["delta_deprecated"]),
                prompt_only_replacement=format_metric(row["prompt_only_replacement"]),
                rerank_replacement=format_metric(row["rerank_replacement"]),
                delta_replacement=format_metric(row["delta_replacement"]),
                rerank_exact=format_metric(row["rerank_exact"]),
                rerank_nonempty=format_metric(row["rerank_nonempty"]),
                rerank_applied_steps=format_metric(row["rerank_applied_steps"]),
                rerank_changed_steps=format_metric(row["rerank_changed_steps"]),
            )
        )
    return header + "\n" + "\n".join(body)


def build_conclusion(rows: List[Dict]) -> str:
    if not rows:
        return "结果尚未提供，因此结论部分暂不生成。"

    dep_improved = all(row["delta_deprecated"] < 0 for row in rows)
    rep_improved = all(row["delta_replacement"] > 0 for row in rows)

    parts = []
    if dep_improved:
        parts.append("在所有已汇总模型规模上，Version-Aware Reranking 均降低了 deprecated API 的使用率。")
    else:
        parts.append("Version-Aware Reranking 在不同模型规模上的 deprecated API 抑制效果并不完全一致。")

    if rep_improved:
        parts.append("与此同时，replacement API 命中率在所有已汇总模型规模上均得到提升。")
    else:
        parts.append("replacement API 命中率的改善趋势在不同模型规模上并不完全一致。")

    parts.append(
        "因此，若最终实验趋势稳定，Plan B 可以被表述为一种无需参数训练、直接在解码阶段利用版本信息进行候选校正的轻量工程方案。"
    )
    return " ".join(parts)


def render_report(rows: List[Dict]) -> str:
    return f"""# Plan B: Version-Aware Reranking 技术报告

## 1. 方法定位

本报告汇总 `Plan B: Version-Aware Reranking` 的实验结果。该方案不改变模型参数，而是在推理阶段利用库版本信息对候选 token 或候选 API 进行重排序，从而压低 deprecated API 的选择概率，并提升 replacement API 的生成概率。与 `Plan A` 相比，`Plan B` 的核心价值在于其轻量性和部署友好性：模型无需重新训练，只需在解码过程中加入版本感知校正。

## 2. 数据集

`Plan B` 使用的是 `rerank_eval` 数据。该数据由三类语义角色不同的样本组成：

- `repair`: 直接对应“无版本控制时易生成 deprecated API，而在版本控制下会被纠正”的修复型样本
- `consistency`: 对应“本来就应当使用 updated API”的一致性样本
- `reference`: 对应稳定的版本一致性参考样本

测试集规模为 `7273`，用于统一比较 `prompt_only` 与 `prompt + reranking` 两种推理方式。

## 3. 推理流程

`Plan B` 的推理流程为：

1. 输入带显式版本前缀的 prompt
2. 模型在当前步产生 top-k 候选
3. 校正器判断每个候选是否更可能把当前补全继续推向 deprecated API 或 replacement API
4. 对 replacement 相关候选加分，对 deprecated 相关候选减分
5. 依据调整后的分数重新选择下一步输出 token

因此，`Plan B` 并不是生成完结果后再做字符串替换，而是在最终 token 被选出之前，对候选排序本身进行干预。

## 4. 评价指标

本实验统一报告以下指标：

- `Deprecated Usage Rate`
- `Replacement Hit Rate`
- `Exact Match Target Rate`
- `Nonempty Prediction Rate`
- `Avg Rerank Applied Steps`
- `Avg Rerank Changed Steps`

其中，前四项衡量最终输出质量与 API 行为，后两项衡量 reranking 在解码过程中实际介入的强度。

## 5. 结果

{build_results_table(rows)}

## 6. 结论

{build_conclusion(rows)}
"""


def main():
    parser = argparse.ArgumentParser(description="Render a Plan B reranking markdown report from comparison summaries")
    parser.add_argument("--summary-3b", type=Path, default=None)
    parser.add_argument("--summary-7b", type=Path, default=None)
    parser.add_argument("--summary-15b", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    specs: List[Tuple[str, Optional[Path]]] = [
        ("StarCoder2-3B", args.summary_3b),
        ("StarCoder2-7B", args.summary_7b),
        ("StarCoder2-15B", args.summary_15b),
    ]

    rows: List[Dict] = []
    for default_label, path in specs:
        if path is None:
            continue
        resolved = resolve_path(path)
        summary = load_json(resolved)
        label = model_label_from_path(resolved) or default_label
        rows.append(extract_row(label, summary))

    if not rows:
        raise SystemExit("At least one summary file must be provided.")

    output_file = resolve_path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(rows), encoding="utf-8")
    print(str(output_file))


if __name__ == "__main__":
    main()
