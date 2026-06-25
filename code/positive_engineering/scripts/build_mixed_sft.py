#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]
    if isinstance(value, str):
        value = normalize_text(value)
        return [value] if value else []
    value = normalize_text(value)
    return [value] if value else []


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def semantic_key(row: Dict) -> str:
    payload = {
        "library": row.get("library", ""),
        "category": row.get("category", ""),
        "version_prompt": normalize_text(row.get("version_prompt", "")),
        "probing_input": normalize_text(row.get("probing_input", "")),
        "deprecated_api": sorted(ensure_list(row.get("deprecated_api"))),
        "replacement_api": sorted(ensure_list(row.get("replacement_api"))),
    }
    return stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def example_key(row: Dict, sem_key: str) -> str:
    payload = {
        "semantic_key": sem_key,
        "target": normalize_text(row.get("target", "")),
    }
    return stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def split_from_semantic_key(sem_key: str) -> str:
    bucket = int(stable_hash("split::" + sem_key)[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_input_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, PROJECT_ROOT / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Input dir not found: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe mixed SFT dataset from repair/consistency/reference buckets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed_clean"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mixed_sft_v1"),
    )
    parser.add_argument("--reference-train-cap", type=int, default=2000)
    parser.add_argument("--reference-val-cap", type=int, default=250)
    parser.add_argument("--reference-test-cap", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_buckets = {
        "repair_sft": {"keep_all": True},
        "consistency_sft": {"keep_all": True},
        "reference_sft": {
            "keep_all": False,
            "caps": {
                "train": args.reference_train_cap,
                "val": args.reference_val_cap,
                "test": args.reference_test_cap,
            },
        },
    }
    splits = ["train", "val", "test"]

    all_rows: List[Dict] = []
    source_file_counts: Dict[str, int] = {}

    for bucket_name in source_buckets:
        for split in splits:
            path = input_dir / f"{bucket_name}_{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")
            rows = load_jsonl(path)
            source_file_counts[f"{bucket_name}_{split}"] = len(rows)
            for row in rows:
                row = dict(row)
                row["_source_bucket"] = bucket_name
                row["_original_split"] = split
                row["_semantic_key"] = semantic_key(row)
                row["_example_key"] = example_key(row, row["_semantic_key"])
                all_rows.append(row)

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in tqdm(all_rows, desc="Grouping semantic keys", total=len(all_rows)):
        groups[row["_semantic_key"]].append(row)

    split_collisions = 0
    assigned_split_by_semantic: Dict[str, str] = {}
    for sem_key, rows in groups.items():
        seen_splits = sorted({row["_original_split"] for row in rows})
        if len(seen_splits) == 1:
            assigned_split = seen_splits[0]
        else:
            split_collisions += 1
            assigned_split = split_from_semantic_key(sem_key)
        assigned_split_by_semantic[sem_key] = assigned_split

    deduped_by_split_bucket: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in tqdm(all_rows, desc="Assigning split-safe rows", total=len(all_rows)):
        assigned_split = assigned_split_by_semantic[row["_semantic_key"]]
        row["_assigned_split"] = assigned_split
        key = (assigned_split, row["_source_bucket"])
        if row["_example_key"] not in deduped_by_split_bucket[key]:
            clean_row = dict(row)
            clean_row["mixed_source_bucket"] = row["_source_bucket"]
            clean_row["mixed_original_split"] = row["_original_split"]
            clean_row["mixed_assigned_split"] = assigned_split
            clean_row["semantic_group_id"] = row["_semantic_key"]
            for temp_key in ["_source_bucket", "_original_split", "_semantic_key", "_example_key", "_assigned_split"]:
                clean_row.pop(temp_key, None)
            deduped_by_split_bucket[key][row["_example_key"]] = clean_row

    rng = random.Random(args.seed)
    mixed_rows_by_split: Dict[str, List[Dict]] = {split: [] for split in splits}
    selection_summary: Dict[str, Dict[str, int]] = {split: {} for split in splits}

    for split in splits:
        for bucket_name, cfg in source_buckets.items():
            rows = list(deduped_by_split_bucket[(split, bucket_name)].values())
            rows.sort(key=lambda row: (row.get("semantic_group_id", ""), row.get("id", "")))

            if cfg["keep_all"]:
                selected = rows
            else:
                cap = cfg["caps"][split]
                if cap <= 0 or len(rows) <= cap:
                    selected = rows
                else:
                    selected = rng.sample(rows, cap)
                    selected.sort(key=lambda row: (row.get("semantic_group_id", ""), row.get("id", "")))

            mixed_rows_by_split[split].extend(selected)
            selection_summary[split][bucket_name] = len(selected)

    for split in splits:
        mixed_rows_by_split[split].sort(
            key=lambda row: (
                row.get("mixed_source_bucket", ""),
                row.get("library", ""),
                row.get("id", ""),
            )
        )

    train_sem = {row["semantic_group_id"] for row in mixed_rows_by_split["train"]}
    val_sem = {row["semantic_group_id"] for row in mixed_rows_by_split["val"]}
    test_sem = {row["semantic_group_id"] for row in mixed_rows_by_split["test"]}
    leakage_report = {
        "train_val_overlap": len(train_sem & val_sem),
        "train_test_overlap": len(train_sem & test_sem),
        "val_test_overlap": len(val_sem & test_sem),
    }

    if any(leakage_report.values()):
        raise RuntimeError(f"Leakage check failed: {leakage_report}")

    written_counts = {}
    for split in splits:
        written_counts[split] = write_jsonl(output_dir / f"mixed_sft_{split}.jsonl", mixed_rows_by_split[split])

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "reference_caps": {
            "train": args.reference_train_cap,
            "val": args.reference_val_cap,
            "test": args.reference_test_cap,
        },
        "source_file_counts": source_file_counts,
        "semantic_group_count": len(groups),
        "split_collisions_resolved": split_collisions,
        "selection_summary": selection_summary,
        "written_counts": written_counts,
        "leakage_report": leakage_report,
        "label_distribution": {
            split: dict(Counter(row.get("mixed_source_bucket", "") for row in rows))
            for split, rows in mixed_rows_by_split.items()
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
