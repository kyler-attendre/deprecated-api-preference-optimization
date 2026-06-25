#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/06_mechanism/output/adl_neutral_20260506"
AGG_ROOT="${OUTPUT_ROOT}/aggregate"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

run_model() {
  local model_key="$1"
  python "${PROJECT_ROOT}/06_mechanism/scripts/run_activation_difference_lens.py" \
    --model-key "${model_key}" \
    --output-dir "${OUTPUT_ROOT}/${model_key}"
}

run_model "starcoder2_3b"
run_model "starcoder2_7b"
run_model "starcoder2_15b"
run_model "deepseek_coder_6_7b_instruct"
run_model "qwen2_5_coder_7b_instruct"

python "${PROJECT_ROOT}/06_mechanism/scripts/summarize_adl_results.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${AGG_ROOT}"

python "${PROJECT_ROOT}/06_mechanism/scripts/plot_adl_results.py" \
  --aggregate-json "${AGG_ROOT}/adl_aggregate_summary.json" \
  --output-file "${PROJECT_ROOT}/05_positive_engineering/md/pic/fig_adl_neutral_20260506.png"
