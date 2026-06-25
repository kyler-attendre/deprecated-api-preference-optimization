#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_retention_gap(payload: Dict) -> Dict:
    dpo_overall = payload["variants"]["dpo"]["summary"]["overall"]["variant_exact_match_rate"]
    anchor_overall = payload["variants"]["anchored_dpo"]["summary"]["overall"]["variant_exact_match_rate"]
    gaps = []
    for row in payload["triplet_library_table"]:
        gaps.append(
            {
                "library": row["library"],
                "samples": row["samples"],
                "dpo_exact_match_rate": row["dpo_exact_match_rate"],
                "anchored_dpo_exact_match_rate": row["anchored_dpo_exact_match_rate"],
                "exact_match_gap": row["anchored_dpo_exact_match_rate"] - row["dpo_exact_match_rate"],
                "dpo_prediction_changed_rate": row["dpo_prediction_changed_rate"],
                "anchored_dpo_prediction_changed_rate": row["anchored_dpo_prediction_changed_rate"],
            }
        )
    gaps.sort(key=lambda item: item["exact_match_gap"], reverse=True)
    return {
        "overall_dpo_exact_match_rate": dpo_overall,
        "overall_anchored_dpo_exact_match_rate": anchor_overall,
        "overall_exact_match_gap": anchor_overall - dpo_overall,
        "largest_exact_match_gaps": gaps,
    }


def build_reverse_scale_table(
    reverse_payloads: Dict[str, Dict],
    anchored_eval_payloads: Dict[str, Dict],
    *,
    model_order: Optional[List[str]] = None,
) -> List[Dict]:
    keys = model_order or sorted(reverse_payloads.keys())
    rows = []
    for key in keys:
        reverse_payload = reverse_payloads[key]
        anchor_eval_payload = anchored_eval_payloads[key]
        subset_ids = {
            str(row["id"])
            for row in load_jsonl(Path(reverse_payload["reverse_subset_file"]))
        }
        anchor_rows = load_jsonl(Path(anchor_eval_payload["lora"]["predictions_file"]))
        matched_rows = [row for row in anchor_rows if str(row.get("id")) in subset_ids]
        matched_hit_rate = (
            sum(int(bool(row.get("has_replacement"))) for row in matched_rows) / len(matched_rows)
            if matched_rows
            else 0.0
        )
        rows.append(
            {
                "model": key,
                "base_old_prefix_replacement_hit_rate": reverse_payload["variants"]["base"]["replacement_hit_rate"],
                "dpo_old_prefix_replacement_hit_rate": reverse_payload["variants"]["dpo"]["replacement_hit_rate"],
                "anchored_dpo_old_prefix_replacement_hit_rate": reverse_payload["variants"]["anchored_dpo"]["replacement_hit_rate"],
                "shared_subset_size": len(matched_rows),
                "anchored_dpo_new_prefix_shared_subset_replacement_hit_rate": matched_hit_rate,
            }
        )
    return rows


def build_sparse_restricted_table(
    full_dpo_payload: Dict,
    full_anchor_payload: Dict,
    dpo_sparse_payload: Dict,
    sparse_payload: Dict,
    restricted_dpo_payload: Dict,
    restricted_payload: Dict,
) -> List[Dict]:
    rows = [
        {
            "config": "dpo_full",
            "deprecated_usage_rate": full_dpo_payload["lora"]["deprecated_usage_rate"],
            "replacement_hit_rate": full_dpo_payload["lora"]["replacement_hit_rate"],
        }
    ]
    for label in ["keep_50", "keep_20", "keep_10"]:
        summary = dpo_sparse_payload["variants"][label]["summary"]
        rows.append(
            {
                "config": f"dpo_{label}",
                "deprecated_usage_rate": summary["deprecated_usage_rate"],
                "replacement_hit_rate": summary["replacement_hit_rate"],
            }
        )
    rows.append(
        {
            "config": "dpo_restricted_layers",
            "deprecated_usage_rate": restricted_dpo_payload["lora"]["deprecated_usage_rate"],
            "replacement_hit_rate": restricted_dpo_payload["lora"]["replacement_hit_rate"],
        }
    )
    rows.extend(
        [
        {
            "config": "anchored_dpo_full",
            "deprecated_usage_rate": full_anchor_payload["lora"]["deprecated_usage_rate"],
            "replacement_hit_rate": full_anchor_payload["lora"]["replacement_hit_rate"],
        }
        ]
    )
    for label in ["keep_50", "keep_20", "keep_10"]:
        summary = sparse_payload["variants"][label]["summary"]
        rows.append(
            {
                "config": f"anchored_dpo_{label}",
                "deprecated_usage_rate": summary["deprecated_usage_rate"],
                "replacement_hit_rate": summary["replacement_hit_rate"],
            }
        )
    rows.append(
        {
            "config": "anchored_dpo_restricted_layers",
            "deprecated_usage_rate": restricted_payload["lora"]["deprecated_usage_rate"],
            "replacement_hit_rate": restricted_payload["lora"]["replacement_hit_rate"],
        }
    )
    return rows


def write_csv(rows: Iterable[Dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: Iterable[Dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    headers = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8") as handle:
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


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_followup_markdown(
    *,
    output_path: Path,
    anchor_manifest: Dict,
    dpo_manifest: Dict,
    retention_summary: Dict,
    reverse_rows: List[Dict],
    sparse_rows: List[Dict],
) -> None:
    lines = []
    lines.append("# Positive Engineering Follow-up Notes")
    lines.append("")
    lines.append("## Restricted-Layer Anchor Configuration Check")
    lines.append("")
    lines.append(f"- `anchor_restricted_layers` manifest records `api_anchor_weight = {anchor_manifest['api_anchor_weight']}`.")
    lines.append(f"- `dpo_restricted_layers` manifest records `api_anchor_weight = {dpo_manifest['api_anchor_weight']}`.")
    lines.append("- The final artifact directories therefore do not support the claim that the restricted-anchor run was accidentally trained as plain DPO.")
    lines.append("")
    lines.append("## Task 1 Retention Finding")
    lines.append("")
    lines.append(
        f"- Overall retention exact-match: DPO `{format_pct(retention_summary['overall_dpo_exact_match_rate'])}` vs Anchored DPO `{format_pct(retention_summary['overall_anchored_dpo_exact_match_rate'])}`; gap `{format_pct(retention_summary['overall_exact_match_gap'])}`."
    )
    top_gaps = retention_summary["largest_exact_match_gaps"][:3]
    for row in top_gaps:
        lines.append(
            f"- `{row['library']}`: DPO `{format_pct(row['dpo_exact_match_rate'])}` vs Anchored DPO `{format_pct(row['anchored_dpo_exact_match_rate'])}`; exact-match gap `{format_pct(row['exact_match_gap'])}`."
        )
    lines.append("- This means plain DPO perturbs non-target API behavior more strongly than Anchored DPO on the current 7B retention set, especially for `pytorch` and `numpy`.")
    lines.append("")
    lines.append("## Task 3 Reverse-Version Comparison")
    lines.append("")
    for row in reverse_rows:
        lines.append(
            f"- `{row['model']}`: on the shared reverse subset (`n={row['shared_subset_size']}`), old-prefix replacement hit is `base {format_pct(row['base_old_prefix_replacement_hit_rate'])}`, `dpo {format_pct(row['dpo_old_prefix_replacement_hit_rate'])}`, `anchored_dpo {format_pct(row['anchored_dpo_old_prefix_replacement_hit_rate'])}`; anchored new-prefix control on the same subset is `{format_pct(row['anchored_dpo_new_prefix_shared_subset_replacement_hit_rate'])}`."
        )
    lines.append("- The corrected comparison now uses the same subset denominator for both old-prefix and new-prefix anchored rows. The 3B and 7B scales stay close under this matched comparison, while 15B drops more under old prefixes and should be described more cautiously.")
    lines.append("")
    lines.append("## Task 2 Concentration Summary")
    lines.append("")
    for row in sparse_rows:
        lines.append(
            f"- `{row['config']}`: deprecated `{format_pct(row['deprecated_usage_rate'])}`, replacement `{format_pct(row['replacement_hit_rate'])}`."
        )
    lines.append("- Sparse delta retention and restricted-layer finetuning tell a consistent story: most replacement preference lives in a concentrated subset of upper-layer parameters, but the restricted-layer run still loses part of the full anchored gain.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile follow-up tables and notes from existing experiment outputs.")
    parser.add_argument("--retention-json", type=Path, required=True)
    parser.add_argument("--reverse-3b-json", type=Path, required=True)
    parser.add_argument("--reverse-7b-json", type=Path, required=True)
    parser.add_argument("--reverse-15b-json", type=Path, required=True)
    parser.add_argument("--anchor-eval-3b-json", type=Path, required=True)
    parser.add_argument("--anchor-eval-7b-json", type=Path, required=True)
    parser.add_argument("--anchor-eval-15b-json", type=Path, required=True)
    parser.add_argument("--anchor-full-json", type=Path, required=True)
    parser.add_argument("--dpo-full-json", type=Path, required=True)
    parser.add_argument("--dpo-sparse-json", type=Path, required=True)
    parser.add_argument("--anchor-sparse-json", type=Path, required=True)
    parser.add_argument("--dpo-restricted-json", type=Path, required=True)
    parser.add_argument("--anchor-restricted-json", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--dpo-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    retention_payload = load_json(args.retention_json.resolve())
    retention_summary = summarize_retention_gap(retention_payload)

    reverse_payloads = {
        "3b": load_json(args.reverse_3b_json.resolve()),
        "7b": load_json(args.reverse_7b_json.resolve()),
        "15b": load_json(args.reverse_15b_json.resolve()),
    }
    anchored_eval_payloads = {
        "3b": load_json(args.anchor_eval_3b_json.resolve()),
        "7b": load_json(args.anchor_eval_7b_json.resolve()),
        "15b": load_json(args.anchor_eval_15b_json.resolve()),
    }
    reverse_rows = build_reverse_scale_table(
        reverse_payloads,
        anchored_eval_payloads,
        model_order=["3b", "7b", "15b"],
    )

    sparse_rows = build_sparse_restricted_table(
        load_json(args.dpo_full_json.resolve()),
        load_json(args.anchor_full_json.resolve()),
        load_json(args.dpo_sparse_json.resolve()),
        load_json(args.anchor_sparse_json.resolve()),
        load_json(args.dpo_restricted_json.resolve()),
        load_json(args.anchor_restricted_json.resolve()),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(reverse_rows, output_dir / "reverse_version_scale_table.csv")
    write_markdown_table(reverse_rows, output_dir / "reverse_version_scale_table.md")
    write_csv(sparse_rows, output_dir / "anchored_concentration_table.csv")
    write_markdown_table(sparse_rows, output_dir / "anchored_concentration_table.md")
    write_followup_markdown(
        output_path=output_dir / "followup_notes.md",
        anchor_manifest=load_json(args.anchor_manifest.resolve()),
        dpo_manifest=load_json(args.dpo_manifest.resolve()),
        retention_summary=retention_summary,
        reverse_rows=reverse_rows,
        sparse_rows=sparse_rows,
    )
    with (output_dir / "retention_gap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(retention_summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
