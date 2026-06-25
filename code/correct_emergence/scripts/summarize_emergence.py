#!/usr/bin/env python3
"""Step 2 (part 1): cross-sample statistics + case-study bucket selection.

Joins the decision-token layerwise traces (`topk_trajectory.jsonl`, produced by
`run_topk_trajectory.py`) with the Step 0 selection records
(`*_correct_examples.jsonl`, which carry `prompt_text`), then per PLAN.md §4.3:

  - tags each example with `api_in_context`: does the ground-truth API's textual
    form (or a known alias, e.g. `torch.nn.functional.X` <-> `F.X`) already
    appear in the prompt the model has seen? This is the operational proxy for
    "copy"-type emergence (induction-head-like) vs. "recall"-type emergence.
  - buckets examples into shallow / mid / deep emergence (data-driven terciles
    of `emergence_layer_top10` under the logit-lens, used as the canonical
    bucketing lens since it requires no learned translator)
  - cross-tabulates `emergence_layer_top10` / `saturation_layer` against
    library / category / task_family / api_in_context
  - selects representative case-study candidates per bucket — examples with the
    sharpest rank transition right at the emergence layer make the clearest
    circuit narratives (cf. Hanna et al. 2023's "this is where candidate X drops
    out and Y enters" framing), with a per-library cap for diversity.

Outputs (no GPU needed — pure post-hoc aggregation over the persisted traces):
  output/starcoder2_7b/cross_sample_analysis.json
  output/starcoder2_7b/case_study_candidates.json
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_07 = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_ROOT_07.parent
MECH_SRC_DIR = PROJECT_ROOT / "06_mechanism" / "src"

for path in (MECH_SRC_DIR, PROJECT_ROOT_07):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.correct_selection import load_jsonl_rows  # noqa: E402
from lens_analysis import first_alias_hit, write_json  # noqa: E402

SOURCES = ["repair", "consistency", "edapibench"]
BUCKET_ORDER = ["shallow", "mid", "deep"]
CANONICAL_BUCKET_LENS = "logit_lens"
CASE_STUDY_PER_BUCKET = 5
MAX_PER_LIBRARY_IN_CASE_STUDY = 2
TRANSITION_WINDOW = 2  # layers before/after the emergence layer to snapshot for narrative material


# ---------------------------------------------------------------------------
# Loading + joining
# ---------------------------------------------------------------------------


def load_correct_example_index(correct_examples_dir: Path, sources: List[str]) -> Dict[Tuple[str, str], Dict]:
    index: Dict[Tuple[str, str], Dict] = {}
    for source in sources:
        path = correct_examples_dir / f"{source}_correct_examples.jsonl"
        for row in load_jsonl_rows(path):
            index[(row["source"], row["row_id"])] = row
    return index


def detect_api_in_context(prompt_text: str, ground_truth_api: str, ground_truth_form: str) -> Optional[str]:
    """Return the textual form of the API that is already present in the
    prompt context (an alias hit or the literal ground-truth span), or None."""
    hit = first_alias_hit(prompt_text, ground_truth_api)
    if hit:
        return hit
    if ground_truth_form and ground_truth_form in prompt_text:
        return ground_truth_form
    return None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def summary_stats(values: List[float]) -> Dict:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "count": n,
        "mean": sum(values_sorted) / n,
        "median": values_sorted[n // 2],
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def compute_tercile_thresholds(values: List[int]) -> Tuple[float, float]:
    values_sorted = sorted(values)
    n = len(values_sorted)
    return values_sorted[n // 3], values_sorted[(2 * n) // 3]


def bucket_for_value(value: int, low: float, high: float) -> str:
    """Bucket boundaries are order statistics (so they can land exactly on a
    mode of the distribution -- which they do here: ~42% of examples emerge at
    exactly layer 18). Use `< low` / `<= high` so tied values fall on the `mid`
    side of the low boundary -- otherwise a single dominant value would get
    split arbitrarily or swallowed whole into `shallow`."""
    if value < low:
        return "shallow"
    if value <= high:
        return "mid"
    return "deep"


def cross_tabulate(records: List[Dict], group_key_fn) -> Dict[str, Dict]:
    """Group decision-token records (already filtered to one lens) by
    `group_key_fn(record)` and report emergence_layer_top10 / saturation_layer
    summary stats per group."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        groups[group_key_fn(record)].append(record)

    result = {}
    for key, group_records in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        emergence = [r["emergence_layer_top10"] for r in group_records if r["emergence_layer_top10"] is not None]
        saturation = [r["saturation_layer"] for r in group_records if r["saturation_layer"] is not None]
        result[key] = {
            "count": len(group_records),
            "emergence_layer_top10": summary_stats(emergence),
            "saturation_layer": summary_stats(saturation),
        }
    return result


# ---------------------------------------------------------------------------
# Case-study candidate selection
# ---------------------------------------------------------------------------


def rank_at_layer(layerwise: List[Dict], layer_idx: int) -> Optional[float]:
    if 0 <= layer_idx < len(layerwise):
        return layerwise[layer_idx]["ground_truth_rank"]
    return None


def transition_snapshot(layerwise: List[Dict], emergence_layer_idx: int) -> List[Dict]:
    """Layer-by-layer snapshot around the emergence layer: the correct token's
    own rank plus the layer's top-3 candidates — the raw material a case
    narrative draws on to say "candidate X drops out, Y enters here"."""
    start = max(0, emergence_layer_idx - TRANSITION_WINDOW)
    end = min(len(layerwise), emergence_layer_idx + TRANSITION_WINDOW + 1)
    snapshot = []
    for entry in layerwise[start:end]:
        snapshot.append(
            {
                "layer": entry["layer"],
                "ground_truth_rank": entry["ground_truth_rank"],
                "ground_truth_prob": entry["ground_truth_prob"],
                "jsd_to_final": entry["jsd_to_final"],
                "top3_candidates": [
                    {"rank": c["rank"], "token_text": c["token_text"], "prob": c["prob"]}
                    for c in entry["top_candidates"][:3]
                ],
            }
        )
    return snapshot


def select_case_study_candidates(
    bucketed_records: Dict[str, List[Dict]],
    *,
    per_bucket: int,
    max_per_library: int,
) -> Dict[str, List[Dict]]:
    """For each bucket, rank examples by how sharply the correct token's rank
    improves right at the emergence layer (a big one-layer jump makes for the
    clearest "this is where it clicks" narrative), then greedily pick the
    sharpest ones while capping how many come from the same library."""
    selected: Dict[str, List[Dict]] = {}
    for bucket, records in bucketed_records.items():
        scored = []
        for record in records:
            layerwise = record["layerwise"]
            emergence_idx = record["emergence_layer_top10"]
            if emergence_idx is None:
                continue
            rank_before = rank_at_layer(layerwise, emergence_idx - 1)
            rank_at = rank_at_layer(layerwise, emergence_idx)
            if rank_before is None or rank_at is None:
                continue
            sharpness = rank_before - rank_at
            scored.append((sharpness, record, emergence_idx))
        scored.sort(key=lambda item: -item[0])

        picked = []
        library_counts: Dict[str, int] = defaultdict(int)
        for sharpness, record, emergence_idx in scored:
            layerwise = record["layerwise"]  # must re-bind here; the outer loop's binding has gone stale
            library = record["library"]
            if library_counts[library] >= max_per_library:
                continue
            picked.append(
                {
                    "row_id": record["row_id"],
                    "source": record["source"],
                    "library": library,
                    "category": record["category"],
                    "task_family": record["task_family"],
                    "ground_truth_api": record["ground_truth_api"],
                    "ground_truth_form": record["ground_truth_form"],
                    "ground_truth_token_text": record["ground_truth_token_text"],
                    "api_in_context": record["api_in_context"],
                    "emergence_layer_top5": record["emergence_layer_top5"],
                    "emergence_layer_top10": record["emergence_layer_top10"],
                    "saturation_layer": record["saturation_layer"],
                    "rank_jump_at_emergence": sharpness,
                    "transition_snapshot": transition_snapshot(layerwise, emergence_idx),
                    "pre_emergence_competitors": record["pre_emergence_competitors"][:5],
                }
            )
            library_counts[library] += 1
            if len(picked) >= per_bucket:
                break
        selected[bucket] = picked
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Step 2: cross-sample statistics + case-study candidate selection.")
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT_07 / "output" / "starcoder2_7b")
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    parser.add_argument("--case-study-per-bucket", type=int, default=CASE_STUDY_PER_BUCKET)
    parser.add_argument("--max-per-library", type=int, default=MAX_PER_LIBRARY_IN_CASE_STUDY)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir if args.analysis_dir.is_absolute() else (Path.cwd() / args.analysis_dir)
    analysis_dir = analysis_dir.resolve()

    trajectory_records = load_jsonl_rows(analysis_dir / "topk_trajectory.jsonl")
    correct_example_index = load_correct_example_index(analysis_dir, args.sources)
    print(f"[summarize_emergence] loaded {len(trajectory_records)} trajectory records, "
          f"{len(correct_example_index)} correct examples", flush=True)

    # --- join: attach api_in_context to every record ---------------------------------
    api_in_context_cache: Dict[Tuple[str, str], Optional[str]] = {}
    for record in trajectory_records:
        join_key = (record["source"], record["row_id"])
        if join_key not in api_in_context_cache:
            example = correct_example_index.get(join_key)
            api_in_context_cache[join_key] = (
                detect_api_in_context(example["prompt_text"], record["ground_truth_api"], record["ground_truth_form"])
                if example is not None
                else None
            )
        record["api_in_context"] = api_in_context_cache[join_key]

    # --- bucket by the canonical lens's emergence_layer_top10 -------------------------
    canonical_records = [r for r in trajectory_records if r["lens"] == CANONICAL_BUCKET_LENS]
    canonical_emergence = [r["emergence_layer_top10"] for r in canonical_records if r["emergence_layer_top10"] is not None]
    low, high = compute_tercile_thresholds(canonical_emergence)
    bucket_by_join_key: Dict[Tuple[str, str], str] = {}
    for record in canonical_records:
        if record["emergence_layer_top10"] is not None:
            bucket_by_join_key[(record["source"], record["row_id"])] = bucket_for_value(record["emergence_layer_top10"], low, high)
    for record in trajectory_records:
        record["emergence_bucket"] = bucket_by_join_key.get((record["source"], record["row_id"]))

    bucket_counts = {bucket: sum(1 for v in bucket_by_join_key.values() if v == bucket) for bucket in BUCKET_ORDER}

    # --- cross-tabulations (one block per lens variant) -------------------------------
    by_lens: Dict[str, Dict] = {}
    for lens_label in sorted({r["lens"] for r in trajectory_records}):
        lens_records = [r for r in trajectory_records if r["lens"] == lens_label]
        in_context_records = [r for r in lens_records if r["api_in_context"] is not None]
        not_in_context_records = [r for r in lens_records if r["api_in_context"] is None]
        by_lens[lens_label] = {
            "by_library": cross_tabulate(lens_records, lambda r: r["library"]),
            "by_category": cross_tabulate(lens_records, lambda r: r["category"]),
            "by_task_family": cross_tabulate(lens_records, lambda r: r["task_family"]),
            "by_api_in_context": {
                "in_context": {
                    "count": len(in_context_records),
                    "emergence_layer_top10": summary_stats([r["emergence_layer_top10"] for r in in_context_records if r["emergence_layer_top10"] is not None]),
                    "saturation_layer": summary_stats([r["saturation_layer"] for r in in_context_records if r["saturation_layer"] is not None]),
                },
                "not_in_context": {
                    "count": len(not_in_context_records),
                    "emergence_layer_top10": summary_stats([r["emergence_layer_top10"] for r in not_in_context_records if r["emergence_layer_top10"] is not None]),
                    "saturation_layer": summary_stats([r["saturation_layer"] for r in not_in_context_records if r["saturation_layer"] is not None]),
                },
            },
        }

    n_examples = len(bucket_by_join_key)
    n_in_context = sum(1 for v in api_in_context_cache.values() if v is not None)
    cross_sample_analysis = {
        "n_examples": n_examples,
        "canonical_bucket_lens": CANONICAL_BUCKET_LENS,
        "bucket_thresholds_emergence_layer_top10": {"low": low, "high": high},
        "bucket_counts": bucket_counts,
        "api_in_context": {
            "hit_count": n_in_context,
            "hit_rate": n_in_context / n_examples if n_examples else 0.0,
            "note": "API textual form (or known alias, e.g. F.x / tf.x / np.x / pd.x) already present in the prompt context -- proxy for 'copy'-type vs 'recall'-type emergence (cf. induction-head literature).",
        },
        "by_lens": by_lens,
    }
    write_json(analysis_dir / "cross_sample_analysis.json", cross_sample_analysis)
    print(f"[summarize_emergence] wrote cross_sample_analysis.json "
          f"(n={n_examples}, bucket_thresholds=({low},{high}), bucket_counts={bucket_counts})", flush=True)

    # --- case-study candidates (canonical lens only -- bucket assignment is canonical-lens-based) ---
    canonical_by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for record in canonical_records:
        bucket = record.get("emergence_bucket")
        if bucket is not None:
            canonical_by_bucket[bucket].append(record)
    case_study_candidates = select_case_study_candidates(
        canonical_by_bucket,
        per_bucket=args.case_study_per_bucket,
        max_per_library=args.max_per_library,
    )
    write_json(
        analysis_dir / "case_study_candidates.json",
        {
            "lens": CANONICAL_BUCKET_LENS,
            "selection_method": "sharpest one-layer rank improvement at the emergence layer, capped per library for diversity",
            "candidates_by_bucket": case_study_candidates,
        },
    )
    picked_counts = {bucket: len(items) for bucket, items in case_study_candidates.items()}
    print(f"[summarize_emergence] wrote case_study_candidates.json (picked={picked_counts})", flush=True)


if __name__ == "__main__":
    main()
