#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_rows_by_id(rows: Iterable[Dict]) -> Dict[str, Dict]:
    return {str(row["id"]): row for row in rows}


def _empty_stats() -> Dict[str, float]:
    return {
        "samples": 0,
        "base_exact": 0,
        "variant_exact": 0,
        "changed": 0,
        "recovered": 0,
        "degraded": 0,
    }


def _finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    samples = stats["samples"]
    return {
        "samples": samples,
        "base_exact_match_rate": stats["base_exact"] / samples if samples else 0.0,
        "variant_exact_match_rate": stats["variant_exact"] / samples if samples else 0.0,
        "prediction_changed_rate": stats["changed"] / samples if samples else 0.0,
        "recovered_from_base_error_rate": stats["recovered"] / samples if samples else 0.0,
        "degraded_from_base_correct_rate": stats["degraded"] / samples if samples else 0.0,
    }


def summarize_shift_against_base(
    *,
    candidate_rows: Iterable[Dict],
    base_rows: Iterable[Dict],
    variant_rows: Iterable[Dict],
    libraries: Iterable[str],
) -> Dict:
    base_by_id = index_rows_by_id(base_rows)
    variant_by_id = index_rows_by_id(variant_rows)

    overall = _empty_stats()
    by_library_raw = {library: _empty_stats() for library in libraries}
    changed_pairs = Counter()

    for candidate in candidate_rows:
        sample_id = str(candidate["id"])
        library = candidate["library"]
        if sample_id not in base_by_id or sample_id not in variant_by_id:
            continue

        base_row = base_by_id[sample_id]
        variant_row = variant_by_id[sample_id]
        base_prediction = base_row.get("predicted_api", "")
        variant_prediction = variant_row.get("predicted_api", "")
        base_exact = bool(base_row.get("exact_api_match"))
        variant_exact = bool(variant_row.get("exact_api_match"))
        changed = base_prediction != variant_prediction
        recovered = (not base_exact) and variant_exact
        degraded = base_exact and (not variant_exact)

        for bucket in [overall, by_library_raw.setdefault(library, _empty_stats())]:
            bucket["samples"] += 1
            bucket["base_exact"] += int(base_exact)
            bucket["variant_exact"] += int(variant_exact)
            bucket["changed"] += int(changed)
            bucket["recovered"] += int(recovered)
            bucket["degraded"] += int(degraded)

        if changed:
            changed_pairs[(base_prediction, variant_prediction)] += 1

    by_library = {
        library: _finalize_stats(by_library_raw[library])
        for library in libraries
    }
    return {
        "overall": _finalize_stats(overall),
        "by_library": by_library,
        "top_changed_predictions": [
            {"base_prediction": src, "variant_prediction": dst, "count": count}
            for (src, dst), count in changed_pairs.most_common(20)
        ],
    }


def parse_variant_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Variant file not found: {path}")
    return label, path


def build_triplet_library_table(payload: Dict) -> List[Dict]:
    rows: List[Dict] = []
    dpo_by_library = payload.get("variants", {}).get("dpo", {}).get("summary", {}).get("by_library", {})
    anchor_by_library = payload.get("variants", {}).get("anchored_dpo", {}).get("summary", {}).get("by_library", {})

    for library in payload.get("libraries", []):
        dpo_stats = dpo_by_library.get(library, {})
        anchor_stats = anchor_by_library.get(library, {})
        samples = dpo_stats.get("samples", anchor_stats.get("samples", 0))
        base_exact_match_rate = dpo_stats.get(
            "base_exact_match_rate",
            anchor_stats.get("base_exact_match_rate", 0.0),
        )
        rows.append(
            {
                "library": library,
                "samples": samples,
                "base_exact_match_rate": base_exact_match_rate,
                "dpo_exact_match_rate": dpo_stats.get("variant_exact_match_rate", 0.0),
                "dpo_prediction_changed_rate": dpo_stats.get("prediction_changed_rate", 0.0),
                "anchored_dpo_exact_match_rate": anchor_stats.get("variant_exact_match_rate", 0.0),
                "anchored_dpo_prediction_changed_rate": anchor_stats.get("prediction_changed_rate", 0.0),
            }
        )
    return rows


def write_triplet_table_csv(rows: Iterable[Dict], output_csv: Path) -> None:
    fieldnames = [
        "library",
        "samples",
        "base_exact_match_rate",
        "dpo_exact_match_rate",
        "dpo_prediction_changed_rate",
        "anchored_dpo_exact_match_rate",
        "anchored_dpo_prediction_changed_rate",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_triplet_table_markdown(rows: Iterable[Dict], output_md: Path) -> None:
    rows = list(rows)
    headers = [
        "library",
        "samples",
        "base_exact_match_rate",
        "dpo_exact_match_rate",
        "dpo_prediction_changed_rate",
        "anchored_dpo_exact_match_rate",
        "anchored_dpo_prediction_changed_rate",
    ]
    with output_md.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            cells = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    cells.append(f"{value:.6f}")
                else:
                    cells.append(str(value))
            handle.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze non-target API retention shifts against the base model.")
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="Variant prediction file in LABEL=PATH format.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()

    candidate_rows = load_jsonl(args.candidate_file.resolve())
    base_rows = load_jsonl(args.base_file.resolve())
    libraries = sorted({row["library"] for row in candidate_rows})

    payload = {
        "candidate_file": str(args.candidate_file.resolve()),
        "base_file": str(args.base_file.resolve()),
        "libraries": libraries,
        "variants": {},
    }
    for label, path in dict(parse_variant_arg(item) for item in args.variant).items():
        payload["variants"][label] = {
            "prediction_file": str(path),
            "summary": summarize_shift_against_base(
                candidate_rows=candidate_rows,
                base_rows=base_rows,
                variant_rows=load_jsonl(path),
                libraries=libraries,
            ),
        }

    payload["triplet_library_table"] = build_triplet_library_table(payload)

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if args.output_csv is not None:
        output_csv = args.output_csv.resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_triplet_table_csv(payload["triplet_library_table"], output_csv)

    if args.output_markdown is not None:
        output_md = args.output_markdown.resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_triplet_table_markdown(payload["triplet_library_table"], output_md)


if __name__ == "__main__":
    main()
