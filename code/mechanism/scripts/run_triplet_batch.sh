#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/06_mechanism/output/triplet_mechanism_20260506"
PIC_ROOT="${PROJECT_ROOT}/05_positive_engineering/md/pic"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

run_compare() {
  local gpu="$1"
  local model_key="$2"
  local max_samples="${3:-0}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${PROJECT_ROOT}/06_mechanism/scripts/run_lens_compare_variants.py" \
      --model-key "${model_key}" \
      --output-dir "${OUTPUT_ROOT}/${model_key}/compare" \
      --max-samples "${max_samples}"
}

run_summary_and_plots() {
  python "${PROJECT_ROOT}/06_mechanism/scripts/summarize_variant_mechanism_results.py" \
    --result-root "${OUTPUT_ROOT}"

  python "${PROJECT_ROOT}/06_mechanism/scripts/plot_variant_mechanism.py" \
    --summary-csv "${OUTPUT_ROOT}/aggregate/final_layer_summary.csv" \
    --lens logit_lens \
    --output-path "${PIC_ROOT}/fig_triplet_mechanism_logit_20260506.png"

  python "${PROJECT_ROOT}/06_mechanism/scripts/plot_variant_mechanism.py" \
    --summary-csv "${OUTPUT_ROOT}/aggregate/final_layer_summary.csv" \
    --lens tuned_lens \
    --output-path "${PIC_ROOT}/fig_triplet_mechanism_tuned_20260506.png"
}

run_smoke() {
  run_compare 0 starcoder2_3b 4
  run_summary_and_plots
}

run_full() {
  run_compare 0 starcoder2_3b 0
  run_compare 1 starcoder2_7b 0
  run_compare 2 starcoder2_15b 0
  run_compare 3 qwen2_5_coder_7b_instruct 0
  run_compare 4 deepseek_coder_6_7b_instruct 0
  run_summary_and_plots
}

MODE="${1:-all}"
case "${MODE}" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  summary)
    run_summary_and_plots
    ;;
  all)
    run_smoke
    run_full
    ;;
  *)
    echo "Usage: $0 [smoke|full|summary|all]" >&2
    exit 1
    ;;
esac
