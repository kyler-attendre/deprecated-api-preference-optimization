#!/usr/bin/env python3
"""
Split mixed_sft_{train,val,test}.jsonl into per-library subsets.
Output: data/mixed_sft_v1/by_library/{library}/mixed_sft_{split}.jsonl
"""
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "mixed_sft_v1"
OUT_ROOT     = DATA_DIR / "by_library"

SPLITS = ["train", "val", "test"]

def main():
    stats = {}
    for split in SPLITS:
        src = DATA_DIR / f"mixed_sft_{split}.jsonl"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue

        buckets = defaultdict(list)
        with src.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                lib = row.get("library", "unknown")
                buckets[lib].append(row)

        for lib, rows in buckets.items():
            out_dir = OUT_ROOT / lib
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"mixed_sft_{split}.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.setdefault(lib, {})[split] = len(rows)
            print(f"  {lib:12s} {split:5s}: {len(rows):5d} → {out_file}")

    print("\n── Per-library sample counts ──────────────────────")
    print(f"{'library':12s}  {'train':>6}  {'val':>5}  {'test':>5}")
    for lib in sorted(stats):
        tr = stats[lib].get("train", 0)
        va = stats[lib].get("val",   0)
        te = stats[lib].get("test",  0)
        print(f"  {lib:12s}  {tr:6d}  {va:5d}  {te:5d}")

if __name__ == "__main__":
    main()
