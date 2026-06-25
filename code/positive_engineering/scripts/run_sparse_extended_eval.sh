#!/usr/bin/env bash
# Task 3a: extended sparsification eval on GPU 6
# Adds keep_5%, keep_15%, keep_30% to existing DPO and CER-DPO results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_test.jsonl"
MODEL="/data/models/StarCoder/starcoder2-7b"
DATE="20260511"
GPU=6

DPO_ADAPTER="${PROJECT_ROOT}/output/dpo_lora_mixed_sft_v1_20260422/starcoder2_7b"
CERDPO_ADAPTER="${PROJECT_ROOT}/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

for VARIANT_NAME in "dpo" "cerdpo"; do
    case "${VARIANT_NAME}" in
        dpo)    ADAPTER="${DPO_ADAPTER}" ;;
        cerdpo) ADAPTER="${CERDPO_ADAPTER}" ;;
    esac

    OUT_DIR="${PROJECT_ROOT}/output/sparse_delta_extended_${DATE}/starcoder2_7b_${VARIANT_NAME}"
    echo "Running extended sparse eval for ${VARIANT_NAME}..."

    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/eval_sparse_lora_delta.py" \
        --model-name-or-path "${MODEL}" \
        --adapter-dir "${ADAPTER}" \
        --test-file "${TEST_FILE}" \
        --output-dir "${OUT_DIR}" \
        --keep-fraction 0.05 \
        --keep-fraction 0.15 \
        --keep-fraction 0.30 \
        --max-length 384 \
        --max-new-tokens 64

    echo "Done: ${VARIANT_NAME}"
done

echo "Extended sparsification eval done."
