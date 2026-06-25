#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a model-specific retention set from base-correct function-level API completions."
    )
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--apis", type=int, default=40)
    parser.add_argument("--samples-per-api", type=int, default=20)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    args = parser.parse_args()

    candidates = {row["id"]: row for row in load_jsonl(args.candidate_file)}
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in load_jsonl(args.base_predictions):
        if not row.get("exact_api_match"):
            continue
        candidate = candidates.get(row["id"])
        if candidate is None:
            continue
        if len(candidate["prompt"]) < args.min_prompt_chars:
            continue
        grouped[candidate.get("full_api") or candidate["target_api"]].append(candidate)

    eligible = [(api, rows) for api, rows in grouped.items() if len(rows) >= args.samples_per_api]
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    chosen = eligible[: args.apis]

    selected_rows: List[Dict] = []
    for api, rows in chosen:
        for row in rows[: args.samples_per_api]:
            selected_rows.append(row)

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_file": str(output_file),
        "selected_apis": len(chosen),
        "samples": len(selected_rows),
        "apis": [{"api": api, "available_base_correct": len(rows)} for api, rows in chosen],
        "requested_apis": args.apis,
        "requested_samples_per_api": args.samples_per_api,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
