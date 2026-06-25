#!/usr/bin/env bash
# Eval all trained ablation adapters after training completes
# Run after run_module_ablation.sh and run_layer_range_ablation.sh finish
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_test.jsonl"
MODEL="/data/models/StarCoder/starcoder2-7b"
DATE="20260511"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

# All ablation variants: (gpu, label, adapter_dir)
declare -a JOBS=(
    "0 mlp_only     ${PROJECT_ROOT}/output/cerdpo_mlp_only_${DATE}/starcoder2_7b"
    "1 attn_mlp     ${PROJECT_ROOT}/output/cerdpo_attn_mlp_${DATE}/starcoder2_7b"
    "2 o_only       ${PROJECT_ROOT}/output/cerdpo_o_only_${DATE}/starcoder2_7b"
    "3 kv_only      ${PROJECT_ROOT}/output/cerdpo_kv_only_${DATE}/starcoder2_7b"
    "4 qkv_no_o     ${PROJECT_ROOT}/output/cerdpo_qkv_no_o_${DATE}/starcoder2_7b"
    "5 L20_31       ${PROJECT_ROOT}/output/cerdpo_restricted_L20_31_${DATE}/starcoder2_7b"
    "5 L25_31       ${PROJECT_ROOT}/output/cerdpo_restricted_L25_31_${DATE}/starcoder2_7b"
    "6 L28_31       ${PROJECT_ROOT}/output/cerdpo_restricted_L28_31_${DATE}/starcoder2_7b"
)

eval_one() {
    local GPU=$1 LABEL=$2 ADAPTER=$3
    local OUT="${PROJECT_ROOT}/output/ablation_eval_${DATE}/${LABEL}"
    echo "[GPU ${GPU}] Evaluating ${LABEL}..."
    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/eval_compare_lora.py" \
        --model-name-or-path "${MODEL}" \
        --adapter-dir "${ADAPTER}" \
        --test-file "${TEST_FILE}" \
        --output-dir "${OUT}" \
        --max-length 384 \
        --max-new-tokens 64 \
        > "${OUT}_eval.log" 2>&1
    echo "[GPU ${GPU}] Done: ${LABEL}"
}

# Launch in parallel on different GPUs (sequential for same GPU)
for JOB in "${JOBS[@]}"; do
    read -r GPU LABEL ADAPTER <<< "${JOB}"
    eval_one "${GPU}" "${LABEL}" "${ADAPTER}" &
done

wait
echo "All ablation evals done."

# Print summary
echo ""
echo "=== Summary ==="
for JOB in "${JOBS[@]}"; do
    read -r GPU LABEL ADAPTER <<< "${JOB}"
    SUMMARY="${PROJECT_ROOT}/output/ablation_eval_${DATE}/${LABEL}/comparison_summary.json"
    if [ -f "${SUMMARY}" ]; then
        python3 -c "
import json
d = json.load(open('${SUMMARY}'))
dep = d['lora']['deprecated_usage_rate']*100
rep = d['lora']['replacement_hit_rate']*100
print(f'${LABEL:15s}: dep={dep:.1f}%  rep={rep:.1f}%')
"
    else
        echo "${LABEL}: eval not complete"
    fi
done
