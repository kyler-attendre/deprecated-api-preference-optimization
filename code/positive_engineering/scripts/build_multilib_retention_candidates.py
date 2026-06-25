#!/usr/bin/env python3
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ALIAS_TO_LIBRARY = {
    "torch": "pytorch",
    "np": "numpy",
    "numpy": "numpy",
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "sns": "seaborn",
    "seaborn": "seaborn",
    "sklearn": "sklearn",
    "scipy": "scipy",
    "pd": "pandas",
    "pandas": "pandas",
    "transformers": "transformers",
}

CALL_RE = re.compile(
    r"((?:torch|np|numpy|tf|tensorflow|sns|seaborn|sklearn|scipy|pd|pandas|transformers)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.)([A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonicalize_full_api(prefix: str, target: str) -> Optional[Tuple[str, str, str]]:
    prefix = prefix.rstrip(".")
    parts = prefix.split(".")
    if not parts:
        return None
    root = parts[0]
    library = ALIAS_TO_LIBRARY.get(root)
    if library is None:
        return None

    canonical_root = {
        "np": "numpy",
        "tf": "tensorflow",
        "sns": "seaborn",
        "pd": "pandas",
    }.get(root, root)
    canonical_prefix = ".".join([canonical_root] + parts[1:]) + "."
    return library, canonical_prefix, canonical_prefix + target


def collect_training_apis(dataset_paths: Iterable[Path]) -> Set[str]:
    apis: Set[str] = set()
    for path in dataset_paths:
        for row in load_jsonl(path):
            for key in ("replacement_api", "deprecated_api"):
                value = row.get(key)
                values = value if isinstance(value, list) else [value]
                for api_name in values:
                    if isinstance(api_name, str) and api_name.strip():
                        apis.add(api_name.strip())
    return apis


def iter_function_nodes(tree: ast.AST) -> Iterable[Tuple[str, ast.AST]]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def slice_node_text(source: str, node: ast.AST) -> Optional[Tuple[str, int]]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    lines = source.splitlines(keepends=True)
    start_line = max(1, int(node.lineno))
    end_line = min(len(lines), int(node.end_lineno))
    start_col = int(getattr(node, "col_offset", 0))
    end_col = int(getattr(node, "end_col_offset", len(lines[end_line - 1])))

    prefix_chars = sum(len(line) for line in lines[: start_line - 1]) + start_col
    start_char = prefix_chars

    body = lines[start_line - 1 : end_line]
    if not body:
        return None
    body[0] = body[0][start_col:]
    body[-1] = body[-1][:end_col]
    return "".join(body), start_char


def build_rows_for_source(
    *,
    source: str,
    source_name: str,
    repo_label: str,
    excluded_training_apis: Set[str],
    min_function_chars: int,
) -> List[Dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    rows: List[Dict] = []
    for function_name, node in iter_function_nodes(tree):
        sliced = slice_node_text(source, node)
        if sliced is None:
            continue
        function_text, node_start_char = sliced
        if len(function_text) < min_function_chars:
            continue

        for match in CALL_RE.finditer(function_text):
            raw_prefix = match.group(1)
            target_api = match.group(2)
            canonical = canonicalize_full_api(raw_prefix, target_api)
            if canonical is None:
                continue
            library, canonical_prefix, full_api = canonical
            if full_api in excluded_training_apis:
                continue

            prompt = function_text[: match.start() + len(raw_prefix)]
            suffix = function_text[match.start() + len(raw_prefix) :]
            rows.append(
                {
                    "source_repo": repo_label,
                    "source_file": source_name,
                    "function_name": function_name,
                    "library": library,
                    "raw_prefix": raw_prefix,
                    "canonical_prefix": canonical_prefix,
                    "target_api": target_api,
                    "full_api": full_api,
                    "prompt": prompt,
                    "function_text": function_text,
                    "suffix_after_api": suffix,
                    "function_chars": len(function_text),
                    "prompt_chars": len(prompt),
                    "source_char_offset": node_start_char + match.start(),
                }
            )
    return rows


def build_rows_for_file(
    *,
    path: Path,
    repo_label: str,
    excluded_training_apis: Set[str],
    min_function_chars: int,
) -> List[Dict]:
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return build_rows_for_source(
        source=source,
        source_name=str(path),
        repo_label=repo_label,
        excluded_training_apis=excluded_training_apis,
        min_function_chars=min_function_chars,
    )


def build_rows_for_notebook(
    *,
    path: Path,
    repo_label: str,
    excluded_training_apis: Set[str],
    min_function_chars: int,
) -> List[Dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

    rows: List[Dict] = []
    for idx, cell in enumerate(payload.get("cells") or []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        if not source.strip():
            continue
        rows.extend(
            build_rows_for_source(
                source=source,
                source_name=f"{path}::cell_{idx}",
                repo_label=repo_label,
                excluded_training_apis=excluded_training_apis,
                min_function_chars=min_function_chars,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-library function-level API retention candidates from public code repos."
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--training-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--min-function-chars", type=int, default=80)
    parser.add_argument("--min-api-count", type=int, default=3)
    parser.add_argument("--max-per-api", type=int, default=0)
    args = parser.parse_args()

    excluded_training_apis = collect_training_apis(args.training_jsonl)
    raw_rows: List[Dict] = []
    for source_root in args.source_root:
        repo_label = source_root.name
        for path in sorted(source_root.rglob("*.py")):
            raw_rows.extend(
                build_rows_for_file(
                    path=path,
                    repo_label=repo_label,
                    excluded_training_apis=excluded_training_apis,
                    min_function_chars=args.min_function_chars,
                )
            )
        for path in sorted(source_root.rglob("*.ipynb")):
            raw_rows.extend(
                build_rows_for_notebook(
                    path=path,
                    repo_label=repo_label,
                    excluded_training_apis=excluded_training_apis,
                    min_function_chars=args.min_function_chars,
                )
            )

    api_counts = Counter(row["full_api"] for row in raw_rows)
    kept_rows = [row for row in raw_rows if api_counts[row["full_api"]] >= args.min_api_count]
    kept_rows.sort(
        key=lambda row: (
            row["library"],
            row["full_api"],
            row["source_repo"],
            row["source_file"],
            row["function_name"],
            row["source_char_offset"],
        )
    )

    final_rows: List[Dict] = []
    per_api_counter: Counter[str] = Counter()
    seen = set()
    for row in kept_rows:
        key = (row["source_file"], row["function_name"], row["full_api"], row["source_char_offset"])
        if key in seen:
            continue
        if args.max_per_api > 0 and per_api_counter[row["full_api"]] >= args.max_per_api:
            continue
        seen.add(key)
        per_api_counter[row["full_api"]] += 1
        row = dict(row)
        row["id"] = f"multilib_retention_candidate_{len(final_rows):05d}"
        final_rows.append(row)

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_file": str(output_file),
        "samples": len(final_rows),
        "unique_full_apis": len(set(row["full_api"] for row in final_rows)),
        "unique_libraries": sorted(set(row["library"] for row in final_rows)),
        "top_full_apis": Counter(row["full_api"] for row in final_rows).most_common(120),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
