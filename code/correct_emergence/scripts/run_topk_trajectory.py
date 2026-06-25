#!/usr/bin/env python3
"""Step 1 entry point: for every "correct" example selected in Step 0, run a
teacher-forced forward pass and extract layer-wise top-k trajectories (logit
lens + tuned lens) for each token position in the ground-truth API span.

Usage example (smoke test, 10 samples):
    python scripts/run_topk_trajectory.py \
        --model-key starcoder2_7b \
        --correct-examples-dir output/smoke_starcoder2_7b \
        --max-samples 10 \
        --output-dir output/smoke_starcoder2_7b

Full run:
    python scripts/run_topk_trajectory.py \
        --model-key starcoder2_7b \
        --correct-examples-dir output/starcoder2_7b \
        --output-dir output/starcoder2_7b
"""
import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_07 = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_ROOT_07.parent
MECH_SRC_DIR = PROJECT_ROOT / "06_mechanism" / "src"

for path in (MECH_SRC_DIR, PROJECT_ROOT_07):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.correct_selection import encode_decision_input, load_jsonl_rows, split_at_namespace_dot  # noqa: E402
from src.topk_trace import (  # noqa: E402
    collect_pre_emergence_competitors,
    emergence_layer,
    layerwise_topk_trace,
    saturation_layer,
)
from lens_analysis import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
    get_decoder_layers,
    get_final_norm,
    get_output_projection,
    load_tuned_lens,
    write_json,
    write_jsonl,
)

SOURCES = ["repair", "consistency", "edapibench"]
EMERGENCE_K_VALUES = (5, 10)
TOP_K_PERSISTED = 10

DEFAULT_TUNED_LENS_PATHS = {
    "starcoder2_7b": PROJECT_ROOT
    / "06_mechanism/output/full_mechanism_20260427/starcoder2_7b/tuned_lens/starcoder2_7b_official_base.pt",
}


def load_correct_examples(correct_examples_dir: Path, sources: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for source in sources:
        path = correct_examples_dir / f"{source}_correct_examples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"correct examples file not found: {path}")
        rows.extend(load_jsonl_rows(path))
    return rows


def build_decision_record(
    *,
    row: Dict,
    lens_label: str,
    namespace_prefix: str,
    decision_form: str,
    suffix_ids: List[int],
    layerwise: List[Dict],
    tokenizer,
) -> Dict:
    """Build one record for the *decision token* — the first token of the
    ground-truth API name *after* its namespace/import-alias prefix (e.g. the
    "random" in "tf.random.categorical", not the "tf"), where the model
    commits to which API to call (cf. PLAN.md §4.2 "在决策位置做
    teacher-forcing 前向" and `06_mechanism/src/lens_analysis.py`'s
    `shared_api_decision_prefix`, which strips the same kind of shared
    namespace prefix). Forms with no '.' (e.g. "GaussianMixture") have an empty
    `namespace_prefix` and are unaffected."""
    ground_truth_token_id = suffix_ids[0]
    em_layers = {f"emergence_layer_top{k}": emergence_layer(layerwise, k) for k in EMERGENCE_K_VALUES}
    sat_layer = saturation_layer(layerwise)
    competitors = collect_pre_emergence_competitors(
        layerwise,
        ground_truth_token_id=ground_truth_token_id,
        emergence_layer_idx=em_layers[f"emergence_layer_top{TOP_K_PERSISTED}"],
    )
    return {
        "row_id": row["row_id"],
        "source": row["source"],
        "library": row["library"],
        "category": row["category"],
        "task_family": row["task_family"],
        "ground_truth_api": row["ground_truth_api"],
        "ground_truth_form": row["ground_truth_form"],
        "namespace_prefix": namespace_prefix,
        "decision_form": decision_form,
        "span_token_count": len(suffix_ids),
        "ground_truth_token_id": ground_truth_token_id,
        "ground_truth_token_text": tokenizer.decode([ground_truth_token_id]),
        "lens": lens_label,
        "num_layers_traced": len(layerwise),
        **em_layers,
        "saturation_layer": sat_layer,
        "layerwise": layerwise,
        "pre_emergence_competitors": competitors[:10],
    }


def run_full(
    *,
    model,
    tokenizer,
    final_norm,
    output_projection,
    rows: List[Dict],
    max_length: int,
    output_dir: Path,
    tuned_lens=None,
) -> Dict:
    lens_variants = [("logit_lens", None)]
    if tuned_lens is not None:
        lens_variants.append(("tuned_lens", tuned_lens))

    records: List[Dict] = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        # Trace the *decision position* per PLAN.md §4.2 ("在决策位置做
        # teacher-forcing 前向"): the first token of the ground-truth API name
        # *after* its namespace/import-alias prefix (e.g. "random" in
        # "tf.random.categorical", not "tf"), mirroring
        # `06_mechanism/src/lens_analysis.py`'s `shared_api_decision_prefix`
        # stripping. Forms with no '.' (e.g. "GaussianMixture") are unaffected.
        namespace_prefix, decision_form = split_at_namespace_dot(row["ground_truth_form"])
        encoded = encode_decision_input(
            tokenizer=tokenizer,
            prefix_text=row["decision_prefix"] + namespace_prefix,
            suffix_text=decision_form,
            max_length=max_length,
        )
        if encoded is None:
            skipped += 1
            continue
        decision_position = [encoded["predict_positions"][0]]
        decision_suffix_ids = [encoded["suffix_ids"][0]]
        for lens_label, lens_obj in lens_variants:
            position_traces = layerwise_topk_trace(
                model=model,
                tokenizer=tokenizer,
                final_norm=final_norm,
                output_projection=output_projection,
                input_ids=encoded["input_ids"],
                suffix_ids=decision_suffix_ids,
                predict_positions=decision_position,
                top_k=TOP_K_PERSISTED,
                tuned_lens=lens_obj,
            )
            records.append(
                build_decision_record(
                    row=row,
                    lens_label=lens_label,
                    namespace_prefix=namespace_prefix,
                    decision_form=decision_form,
                    suffix_ids=encoded["suffix_ids"],
                    layerwise=position_traces[0],
                    tokenizer=tokenizer,
                )
            )
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"[run_topk_trajectory] processed {i + 1}/{len(rows)} examples, elapsed={time.time() - t0:.1f}s", flush=True)

    write_jsonl(output_dir / "topk_trajectory.jsonl", records)
    return {
        "examples": len(rows),
        "skipped_examples": skipped,
        "lens_variants": [label for label, _ in lens_variants],
        "trajectory_records": len(records),
        "elapsed_seconds": round(time.time() - t0, 1),
        "output_file": str((output_dir / "topk_trajectory.jsonl").resolve()),
    }


def summarize_emergence(records: List[Dict]) -> Dict:
    """Aggregate emergence/saturation indicators and competitor profiles for
    the decision-token trajectories (one record per example x lens variant)."""

    def layer_stats(values: List[int]) -> Dict:
        if not values:
            return {"count": 0}
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            "count": n,
            "mean": sum(values_sorted) / n,
            "min": values_sorted[0],
            "p25": values_sorted[n // 4],
            "median": values_sorted[n // 2],
            "p75": values_sorted[(3 * n) // 4],
            "max": values_sorted[-1],
        }

    summary: Dict = {"by_lens": {}}
    for lens_label in sorted({r["lens"] for r in records}):
        lens_records = [r for r in records if r["lens"] == lens_label]

        emergence_top5 = [r["emergence_layer_top5"] for r in lens_records if r["emergence_layer_top5"] is not None]
        emergence_top10 = [r["emergence_layer_top10"] for r in lens_records if r["emergence_layer_top10"] is not None]
        saturation = [r["saturation_layer"] for r in lens_records if r["saturation_layer"] is not None]

        never_emerged_top10 = sum(1 for r in lens_records if r["emergence_layer_top10"] is None)
        never_saturated = sum(1 for r in lens_records if r["saturation_layer"] is None)

        competitor_class_counts: Counter = Counter()
        competitor_text_counts: Counter = Counter()
        for r in lens_records:
            for comp in r["pre_emergence_competitors"]:
                competitor_class_counts[comp["token_class"]] += 1
                competitor_text_counts[comp["token_text"]] += 1

        by_source = defaultdict(list)
        by_library = defaultdict(list)
        for r in lens_records:
            if r["emergence_layer_top10"] is not None:
                by_source[r["source"]].append(r["emergence_layer_top10"])
                by_library[r["library"]].append(r["emergence_layer_top10"])

        summary["by_lens"][lens_label] = {
            "decision_token_examples": len(lens_records),
            "emergence_layer_top5": layer_stats(emergence_top5),
            "emergence_layer_top10": layer_stats(emergence_top10),
            "saturation_layer": layer_stats(saturation),
            "never_entered_top10_count": never_emerged_top10,
            "never_saturated_count": never_saturated,
            "competitor_token_class_distribution": dict(competitor_class_counts.most_common()),
            "top_competitor_tokens": [
                {"token_text": text, "appearances_across_examples": count}
                for text, count in competitor_text_counts.most_common(20)
            ],
            "emergence_layer_top10_by_source": {src: layer_stats(vals) for src, vals in sorted(by_source.items())},
            "emergence_layer_top10_by_library": {
                lib: layer_stats(vals) for lib, vals in sorted(by_library.items(), key=lambda kv: -len(kv[1]))[:10]
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 1: layer-wise top-k trajectory extraction for correct examples.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default="starcoder2_7b")
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    parser.add_argument("--correct-examples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = no cap (full run)")
    parser.add_argument("--tuned-lens-path", type=Path, default=None, help="omit to auto-pick the registered base-model tuned lens; pass empty string to disable")
    args = parser.parse_args()

    correct_examples_dir = args.correct_examples_dir
    if not correct_examples_dir.is_absolute():
        correct_examples_dir = (Path.cwd() / correct_examples_dir).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_path = MODEL_REGISTRY[args.model_key]["path"]
    max_length = args.max_length or MODEL_REGISTRY[args.model_key]["max_length"]

    rows = load_correct_examples(correct_examples_dir, args.sources)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"[run_topk_trajectory] loaded {len(rows)} correct examples from {correct_examples_dir}", flush=True)

    print(f"[run_topk_trajectory] loading model: {model_name_or_path}", flush=True)
    model, tokenizer = build_model(model_name_or_path)
    final_norm = get_final_norm(model)
    output_projection = get_output_projection(model)

    tuned_lens = None
    tuned_lens_metadata = None
    tuned_lens_path = args.tuned_lens_path
    if tuned_lens_path is None:
        tuned_lens_path = DEFAULT_TUNED_LENS_PATHS.get(args.model_key)
    if tuned_lens_path is not None and str(tuned_lens_path) and Path(tuned_lens_path).exists():
        tuned_lens, tuned_lens_metadata = load_tuned_lens(Path(tuned_lens_path), device=next(model.parameters()).device)
        print(f"[run_topk_trajectory] loaded tuned lens: {tuned_lens_path}", flush=True)
    else:
        print(f"[run_topk_trajectory] no tuned lens loaded (path={tuned_lens_path}); running logit-lens only", flush=True)

    run_info = run_full(
        model=model,
        tokenizer=tokenizer,
        final_norm=final_norm,
        output_projection=output_projection,
        rows=rows,
        max_length=max_length,
        output_dir=output_dir,
        tuned_lens=tuned_lens,
    )

    records = load_jsonl_rows(output_dir / "topk_trajectory.jsonl")
    emergence_summary = summarize_emergence(records)

    summary = {
        "model_key": args.model_key,
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "num_decoder_layers": len(get_decoder_layers(model)),
        "top_k_persisted": TOP_K_PERSISTED,
        "emergence_k_values": list(EMERGENCE_K_VALUES),
        "tuned_lens_path": str(tuned_lens_path) if tuned_lens is not None else None,
        "tuned_lens_metadata": tuned_lens_metadata,
        "sources": args.sources,
        "max_samples": args.max_samples,
        "run": run_info,
        "emergence_summary": emergence_summary,
    }
    write_json(output_dir / "emergence_summary.json", summary)
    print(f"[run_topk_trajectory] wrote {run_info['trajectory_records']} trajectory records to {run_info['output_file']}")
    print(f"[run_topk_trajectory] wrote summary to {output_dir / 'emergence_summary.json'}")


if __name__ == "__main__":
    main()
