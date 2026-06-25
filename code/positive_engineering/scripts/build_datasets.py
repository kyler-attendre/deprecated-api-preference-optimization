#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_utils import (
    baseline_bad_output,
    build_record_id,
    is_up_to_dated,
    iter_result_files,
    make_preference_record,
    make_sft_record,
    normalize_whitespace,
    read_json,
    replacement_target_from_bad2good,
    safe_model_name,
    split_name,
    write_jsonl,
)


def sample_key(sample: Dict) -> str:
    function = normalize_whitespace(sample.get("function", ""))
    if function:
        return function
    probing_input = normalize_whitespace(sample.get("probing input", ""))
    reference = normalize_whitespace(sample.get("reference", ""))
    return f"{probing_input}|||{reference}"


def split_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    buckets = defaultdict(list)
    for record in records:
        split = split_name(record["id"])
        buckets[split].append(record)
    return buckets


def write_split_group(output_dir: Path, base_name: str, records: List[Dict]) -> None:
    buckets = split_records(records)
    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"{base_name}_{split}.jsonl", buckets.get(split, []))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collect_bad2good_samples(bad2good_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for model_dir in sorted(bad2good_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for json_file in sorted(model_dir.glob("*_bad2good.json")):
            library = json_file.stem.replace("_bad2good", "")
            samples = read_json(json_file)
            for sample in samples:
                sample = dict(sample)
                sample["library"] = sample.get("library") or library
                sample["_source_file"] = str(json_file)
                sample["_source_group"] = "bad2good"
                sample["_source_model"] = model_name
                sample["_sample_key"] = sample_key(sample)
                rows.append(sample)
    return rows


def collect_good2bad_samples(good2bad_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not good2bad_root.exists():
        return rows
    for model_dir in sorted(good2bad_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for json_file in sorted(model_dir.glob("*_good2bad.json")):
            library = json_file.stem.replace("_good2bad", "")
            samples = read_json(json_file)
            for sample in samples:
                sample = dict(sample)
                sample["library"] = sample.get("library") or library
                sample["_source_file"] = str(json_file)
                sample["_source_group"] = "good2bad"
                sample["_source_model"] = model_name
                sample["_sample_key"] = sample_key(sample)
                rows.append(sample)
    return rows


def collect_version_samples(version_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for library, model_name, result_file in iter_result_files(version_root):
        samples = read_json(result_file)
        for sample in samples:
            sample = dict(sample)
            sample["library"] = sample.get("library") or library
            sample["_source_file"] = str(result_file)
            sample["_source_group"] = "version_results"
            sample["_source_model"] = model_name
            sample["_sample_key"] = sample_key(sample)
            rows.append(sample)
    return rows


def make_repair_sft_records(samples: Iterable[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for sample in samples:
        target = replacement_target_from_bad2good(sample)
        record = make_sft_record(
            sample=sample,
            model=sample["_source_model"],
            sample_type="repair_outdated_bad2good_distilled",
            target=target,
            source_file=sample["_source_file"],
        )
        if record:
            record["task_family"] = "repair"
            record["label_axis"] = "behavior_shift"
            records.append(record)
    return records


def make_consistency_sft_records(samples: Iterable[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for sample in samples:
        target = replacement_target_from_bad2good(sample)
        record = make_sft_record(
            sample=sample,
            model=sample["_source_model"],
            sample_type="consistency_uptodated_bad2good_distilled",
            target=target,
            source_file=sample["_source_file"],
        )
        if record:
            record["task_family"] = "consistency"
            record["label_axis"] = "behavior_shift"
            records.append(record)
    return records


def make_reference_sft_records(samples: Iterable[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for sample in samples:
        record = make_sft_record(
            sample=sample,
            model=sample["_source_model"],
            sample_type="reference_uptodated_exclusive",
            target=sample.get("reference", ""),
            source_file=sample["_source_file"],
        )
        if record:
            record["task_family"] = "reference"
            record["label_axis"] = "context_label"
            records.append(record)
    return records


def make_preference_records(samples: Iterable[Dict], sample_type: str, task_family: str) -> List[Dict]:
    records: List[Dict] = []
    for sample in samples:
        record = make_preference_record(
            sample=sample,
            model=sample["_source_model"],
            source_file=sample["_source_file"],
        )
        if record:
            record["sample_type"] = sample_type
            record["task_family"] = task_family
            record["label_axis"] = "behavior_shift"
            records.append(record)
    return records


def make_rerank_records(records: Iterable[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for record in records:
        rows.append(
            {
                "id": record["id"],
                "model": record["model"],
                "library": record["library"],
                "version_prompt": record["version_prompt"],
                "probing_input": record["probing_input"],
                "deprecated_api": record["deprecated_api"],
                "replacement_api": record["replacement_api"],
                "category": record["category"],
                "sample_type": record["sample_type"],
                "task_family": record["task_family"],
                "reference": record["reference"],
            }
        )
    return rows


def build_inventory_summary(
    bad2good_rows: List[Dict],
    good2bad_rows: List[Dict],
    version_rows: List[Dict],
    overlap_keys: Set[str],
) -> Dict:
    def category_counter(rows: Iterable[Dict]) -> Dict[str, int]:
        return dict(sorted(Counter(row.get("category", "unknown") for row in rows).items()))

    def by_model_and_category(rows: Iterable[Dict]) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            stats[safe_model_name(row["_source_model"])][row.get("category", "unknown")] += 1
        return {model: dict(sorted(counter.items())) for model, counter in sorted(stats.items())}

    return {
        "semantic_axes": {
            "context_label": "category 字段描述样本语境，主要是 outdated / up-to-dated",
            "behavior_shift": "bad2good / good2bad 描述模型在不同提示或设置下的行为变化",
        },
        "raw_sources": {
            "bad2good_total": len(bad2good_rows),
            "bad2good_by_category": category_counter(bad2good_rows),
            "bad2good_by_model": by_model_and_category(bad2good_rows),
            "good2bad_total": len(good2bad_rows),
            "good2bad_by_category": category_counter(good2bad_rows),
            "good2bad_by_model": by_model_and_category(good2bad_rows) if good2bad_rows else {},
            "version_results_total": len(version_rows),
            "version_results_by_category": category_counter(version_rows),
            "bad2good_unique_keys": len({row["_sample_key"] for row in bad2good_rows}),
            "version_uptodated_unique_keys": len({row["_sample_key"] for row in version_rows if is_up_to_dated(row)}),
            "bad2good_vs_uptodated_overlap_unique_keys": len(overlap_keys),
        },
    }


def build_dataset_summary(groups: Dict[str, List[Dict]]) -> Dict:
    summary = {}
    for name, records in groups.items():
        split_counts = {split: len(rows) for split, rows in split_records(records).items()}
        summary[name] = {
            "total": len(records),
            "split_counts": {
                "train": split_counts.get("train", 0),
                "val": split_counts.get("val", 0),
                "test": split_counts.get("test", 0),
            },
            "models": sorted({record["model"] for record in records}) if records else [],
            "libraries": sorted({record["library"] for record in records}) if records else [],
        }
    return summary


def build_readme_text(
    inventory: Dict,
    dataset_summary: Dict,
) -> str:
    lines = [
        "# Dataset Classification",
        "",
        "这份说明把正向工程用到的数据重新按语义分类，避免把 `category` 和 `bad2good` 这两个不同维度混在一起。",
        "",
        "## 两条正交维度",
        "",
        "- `category` 是语境标签，表示原始样本属于 `outdated` 还是 `up-to-dated`。",
        "- `bad2good` / `good2bad` 是行为转移标签，表示同一样本在不同设置下，模型输出是从坏变好还是从好变坏。",
        "- 所以一个样本完全可能同时是 `up-to-dated`，又属于 `bad2good`。这不是脏数据，而是两个标签轴描述的不是同一件事。",
        "",
        "## 原始数据源",
        "",
        f"- `bad2good` 总数: {inventory['raw_sources']['bad2good_total']}",
        f"- `bad2good` 按 category: {inventory['raw_sources']['bad2good_by_category']}",
        f"- `good2bad` 总数: {inventory['raw_sources']['good2bad_total']}",
        f"- `good2bad` 按 category: {inventory['raw_sources']['good2bad_by_category']}",
        f"- `version-control-results-version` 总数: {inventory['raw_sources']['version_results_total']}",
        f"- `version-control-results-version` 按 category: {inventory['raw_sources']['version_results_by_category']}",
        f"- `bad2good` 与 `version up-to-dated` 的唯一键交集: {inventory['raw_sources']['bad2good_vs_uptodated_overlap_unique_keys']}",
        "",
        "## 重构后的训练桶",
        "",
        "- `repair_sft_*`: 只保留 `bad2good` 且 `category == outdated`。这是真正直接对应“降低弃用率”的修复型训练集。",
        "- `repair_preference_*`: 同一批 `outdated bad2good` 样本，保留 chosen/rejected 对，适合后续 DPO 或 ranking。",
        "- `consistency_sft_*`: 只保留 `bad2good` 且 `category == up-to-dated`。这表示版本信息帮助模型在本来就该用新 API 的语境里从错改对。",
        "- `consistency_preference_*`: 同一批 `up-to-dated bad2good` 样本的偏好版。",
        "- `reference_sft_*`: 来自 `version-control-results-version` 且 `category == up-to-dated`，同时排除了所有与 `bad2good` 重叠的样本键。它更像干净的参考监督集，而不是行为转移集。",
        "- `rerank_eval_*`: 合并上面三类 SFT 记录，只用于 reranking 评估，不建议直接当训练集。",
        "",
        "## 当前规模",
        "",
    ]
    for name, item in dataset_summary.items():
        lines.append(
            f"- `{name}`: total={item['total']}, train={item['split_counts']['train']}, val={item['split_counts']['val']}, test={item['split_counts']['test']}"
        )
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 如果目标是做“显式版本约束的参数化蒸馏”，优先用 `repair_sft_*` 作为主训练集。",
            "- 如果想让模型保留“本来就该用新 API”的稳定性，可以把 `consistency_sft_*` 作为辅训练集低比例混入。",
            "- 如果要补充通用新 API 分布，可以再混入 `reference_sft_*`，但建议控制比例，避免稀释修复信号。",
            "- `good2bad` 更适合拿来分析失败模式，不建议直接纳入正向训练。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    project_root = args.project_root
    pe_root = project_root / "05_positive_engineering"
    rq1_root = project_root / "rq1_effectiveness" / "src"

    bad2good_root = rq1_root / "bad2goood"
    good2bad_root = rq1_root / "good2bad"
    version_root = rq1_root / "version-control-results-version"
    output_root = pe_root / "data" / "processed_clean"

    bad2good_rows = collect_bad2good_samples(bad2good_root)
    good2bad_rows = collect_good2bad_samples(good2bad_root)
    version_rows = collect_version_samples(version_root)

    bad2good_keys = {row["_sample_key"] for row in bad2good_rows}
    exclusive_reference_rows = [
        row for row in version_rows
        if is_up_to_dated(row) and row["_sample_key"] not in bad2good_keys
    ]
    overlap_keys = {
        row["_sample_key"] for row in version_rows
        if is_up_to_dated(row) and row["_sample_key"] in bad2good_keys
    }

    repair_rows = [row for row in bad2good_rows if row.get("category") == "outdated"]
    consistency_rows = [row for row in bad2good_rows if row.get("category") == "up-to-dated"]

    repair_sft = make_repair_sft_records(repair_rows)
    repair_preference = make_preference_records(
        repair_rows,
        sample_type="repair_outdated_bad2good_preference",
        task_family="repair",
    )
    consistency_sft = make_consistency_sft_records(consistency_rows)
    consistency_preference = make_preference_records(
        consistency_rows,
        sample_type="consistency_uptodated_bad2good_preference",
        task_family="consistency",
    )
    reference_sft = make_reference_sft_records(exclusive_reference_rows)
    rerank_eval = make_rerank_records(repair_sft + consistency_sft + reference_sft)

    groups = {
        "repair_sft": repair_sft,
        "repair_preference": repair_preference,
        "consistency_sft": consistency_sft,
        "consistency_preference": consistency_preference,
        "reference_sft": reference_sft,
        "rerank_eval": rerank_eval,
    }

    for name, records in groups.items():
        write_split_group(output_root, name, records)

    inventory = build_inventory_summary(
        bad2good_rows=bad2good_rows,
        good2bad_rows=good2bad_rows,
        version_rows=version_rows,
        overlap_keys=overlap_keys,
    )
    dataset_summary = build_dataset_summary(groups)

    write_json(output_root / "inventory_summary.json", inventory)
    write_json(output_root / "dataset_summary.json", dataset_summary)
    (output_root / "README.md").write_text(
        build_readme_text(inventory=inventory, dataset_summary=dataset_summary),
        encoding="utf-8",
    )

    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
