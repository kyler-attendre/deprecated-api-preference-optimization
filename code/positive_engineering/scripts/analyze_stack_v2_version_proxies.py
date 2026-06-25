#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set


VERSION_FILENAMES = {"version.py", "_version.py", "versions.py"}
CONFIG_KEYWORDS = ("config", "settings")
EXAMPLE_TEST_SEGMENTS = ("/test/", "/tests/", "/example/", "/examples/")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_library_usage_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_version_proxy_path(path: str) -> Set[str]:
    lowered = path.lower()
    basename = lowered.rsplit("/", 1)[-1]
    tags: Set[str] = set()

    if basename == "setup.py":
        tags.add("setup_py")
    if basename in VERSION_FILENAMES or "version" in basename:
        tags.add("version_module_py")
    if lowered.endswith("/docs/conf.py") or basename == "conf.py" and "/docs/" in lowered:
        tags.add("docs_conf_py")
    if basename == "__init__.py":
        tags.add("init_py")
    if any(keyword in basename for keyword in CONFIG_KEYWORDS):
        tags.add("config_or_settings_py")
    if any(segment in lowered for segment in EXAMPLE_TEST_SEGMENTS):
        tags.add("examples_or_tests_path")

    return tags


def summarize_version_proxy_rows(rows: Iterable[Dict], max_examples: int = 3) -> Dict:
    rows = list(rows)
    category_counts = Counter()
    example_paths: Dict[str, List[str]] = {}

    for row in rows:
        path = row.get("path", "")
        for tag in sorted(classify_version_proxy_path(path)):
            category_counts[tag] += 1
            example_paths.setdefault(tag, [])
            if len(example_paths[tag]) < max_examples:
                example_paths[tag].append(path)

    total = len(rows)
    category_rates = {
        key: (value / total if total else 0.0)
        for key, value in sorted(category_counts.items())
    }
    return {
        "rows_scanned": total,
        "category_counts": dict(sorted(category_counts.items())),
        "category_rates": category_rates,
        "example_paths": example_paths,
    }


def build_markdown_report(*, metadata_path: Path, metadata_summary: Dict, library_rows: List[Dict], backup_report_excerpt: str) -> str:
    lines = []
    lines.append("# Stack v2 Version-Signal Proxy Analysis")
    lines.append("")
    lines.append("## Data Availability")
    lines.append("")
    lines.append(f"- Metadata file: `{metadata_path}`")
    lines.append(f"- Rows scanned from local metadata sample: `{metadata_summary['rows_scanned']}`")
    lines.append("- Available fields are metadata only (`path`, `repo_name`, dates, ids, sizes). No source-code `content` field is present in the local file.")
    lines.append("- Therefore the original Task 5 target, identifying exact version-declaration text forms such as inline comments, `requirements` specifiers, or `__version__` assignments, remains infeasible from the current local dataset.")
    lines.append("")
    lines.append("## Metadata Proxy Signals")
    lines.append("")
    for category, count in metadata_summary["category_counts"].items():
        rate = metadata_summary["category_rates"][category]
        examples = ", ".join(f"`{path}`" for path in metadata_summary["example_paths"].get(category, []))
        lines.append(f"- `{category}`: {count} rows ({rate:.2%}); examples: {examples}")
    lines.append("")
    lines.append("## Backup Stack-v2 API Usage Evidence")
    lines.append("")
    lines.append("The backup quantitative analysis remains the strongest local evidence about training-data skew between deprecated and replacement APIs:")
    lines.append("")
    for row in library_rows:
        lines.append(
            f"- `{row['library']}`: deprecated `{row['deprecated_occurrences']}`, replacement `{row['replacing_occurrences']}`, ratio `{float(row['replacing_to_deprecated_ratio']):.2f}x`"
        )
    lines.append("")
    lines.append("## Backup Report Excerpt")
    lines.append("")
    lines.append("```text")
    lines.append(backup_report_excerpt.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Local data supports a limited proxy conclusion only: the current Stack v2 sample exposes file-path carriers where version information often lives (`setup.py`, version-named modules, docs config), but not the version strings themselves.")
    lines.append("- The backup API-usage analysis still supports the paper’s broader RQ2 narrative that replacement APIs are generally more exposed than deprecated APIs in Stack v2, with strong library-level variation.")
    lines.append("- A faithful completion of the original Task 5 still requires source text, not only metadata.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze path-based version-signal proxies from local Stack v2 metadata.")
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--backup-library-csv", type=Path, required=True)
    parser.add_argument("--backup-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    metadata_rows = load_jsonl(args.metadata_file.resolve())
    metadata_summary = summarize_version_proxy_rows(metadata_rows)
    library_rows = load_library_usage_csv(args.backup_library_csv.resolve())
    backup_report_text = args.backup_report.resolve().read_text(encoding="utf-8")
    backup_report_excerpt = "\n".join(backup_report_text.splitlines()[:40])

    payload = {
        "metadata_file": str(args.metadata_file.resolve()),
        "metadata_summary": metadata_summary,
        "backup_library_statistics": library_rows,
        "backup_report_excerpt": backup_report_excerpt,
        "limitations": [
            "local Stack v2 file contains metadata only",
            "no source-code content field is available locally",
            "exact textual version-declaration patterns cannot be recovered from this file",
        ],
    }

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    output_markdown = args.output_markdown.resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        build_markdown_report(
            metadata_path=args.metadata_file.resolve(),
            metadata_summary=metadata_summary,
            library_rows=library_rows,
            backup_report_excerpt=backup_report_excerpt,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
