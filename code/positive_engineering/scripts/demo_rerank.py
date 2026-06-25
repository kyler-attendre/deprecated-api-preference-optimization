#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reranker import Candidate, VersionAwareReranker


def main():
    parser = argparse.ArgumentParser(description="Small demo for version-aware reranking")
    parser.add_argument("--input-json", type=Path, required=True, help="JSON file containing candidate list")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates = [Candidate(**item) for item in payload["candidates"]]
    reranker = VersionAwareReranker(alpha=args.alpha, beta=args.beta)
    reranked = reranker.rerank(
        candidates,
        deprecated_apis=payload.get("deprecated_api", []),
        replacement_api=payload.get("replacement_api", ""),
    )

    print(json.dumps([c.__dict__ for c in reranked], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
