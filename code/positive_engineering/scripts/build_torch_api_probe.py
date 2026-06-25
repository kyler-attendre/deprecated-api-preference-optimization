#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set


MATCH_RE = re.compile(r"torch\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_trained_torch_prefixes(dataset_paths: Iterable[Path]) -> Set[str]:
    prefixes: Set[str] = set()
    for path in dataset_paths:
        for row in load_jsonl(path):
            if row.get("library") != "pytorch":
                continue
            for key in ("replacement_api", "deprecated_api"):
                value = row.get(key)
                values = value if isinstance(value, list) else [value]
                for api_name in values:
                    if not api_name or not isinstance(api_name, str):
                        continue
                    if not api_name.startswith("torch."):
                        continue
                    remainder = api_name[len("torch.") :]
                    prefix = remainder.split(".", 1)[0].strip()
                    if prefix:
                        prefixes.add(prefix)
    return prefixes


def normalize_prompt_prefix(text: str, *, start: int, prompt_chars: int) -> str:
    left = max(0, start - prompt_chars)
    return text[left : start + len("torch.")]


def extract_samples(
    *,
    source_root: Path,
    trained_prefixes: Set[str],
    max_samples: int,
    prompt_chars: int,
    min_target_freq: int,
) -> List[Dict]:
    candidates: List[Dict] = []
    freq: Dict[str, int] = {}

    for path in sorted(source_root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in MATCH_RE.finditer(text):
            target_api = match.group(1)
            if not target_api:
                continue
            if not target_api[0].islower():
                continue
            if target_api in trained_prefixes:
                continue
            prompt = normalize_prompt_prefix(text, start=match.start(), prompt_chars=prompt_chars)
            row = {
                "source_repo": "pytorch/examples",
                "source_file": str(path),
                "target_api": target_api,
                "prompt": prompt,
                "trained_prefixes": sorted(trained_prefixes),
            }
            candidates.append(row)
            freq[target_api] = freq.get(target_api, 0) + 1

    filtered = [row for row in candidates if freq[row["target_api"]] >= min_target_freq]
    filtered.sort(key=lambda row: (row["target_api"], row["source_file"], len(row["prompt"])))

    deduped: List[Dict] = []
    seen = set()
    for row in filtered:
        key = (row["target_api"], row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        row["id"] = f"torch_api_probe_{len(deduped):04d}"
        deduped.append(row)
        if max_samples > 0 and len(deduped) >= max_samples:
            break
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small normal-API completion probe from official PyTorch examples."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--training-jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="Training/eval jsonl files used to derive trained PyTorch API prefixes.",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--prompt-chars", type=int, default=1600)
    parser.add_argument("--min-target-freq", type=int, default=1)
    args = parser.parse_args()

    trained_prefixes = collect_trained_torch_prefixes(args.training_jsonl)
    rows = extract_samples(
        source_root=args.source_root,
        trained_prefixes=trained_prefixes,
        max_samples=args.max_samples,
        prompt_chars=args.prompt_chars,
        min_target_freq=args.min_target_freq,
    )

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_root": str(args.source_root.resolve()),
        "output_file": str(output_file),
        "samples": len(rows),
        "trained_prefixes": sorted(trained_prefixes),
        "unique_targets": sorted({row["target_api"] for row in rows}),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
