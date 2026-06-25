#!/usr/bin/env bash
# Task 2b: layer range fine-grained ablation on GPU 5 (serial)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_val.jsonl"
MODEL="/data/models/StarCoder/starcoder2-7b"
DATE="20260511"
GPU=5

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

common_args=(
    --train-file "${TRAIN_FILE}"
    --val-file   "${VAL_FILE}"
    --model-name-or-path "${MODEL}"
    --max-length 384
    --learning-rate 5e-5
    --num-train-epochs 1
    --per-device-train-batch-size 1
    --gradient-accumulation-steps 8
    --beta 0.1
    --logprob-reduction sum
    --dpo-scope full
    --api-anchor-weight 0.1
    --lora-r 8
    --lora-alpha 16
    --lora-dropout 0.05
    --target-modules q_proj o_proj
)

for RANGE_NAME in "L20_31" "L25_31" "L28_31"; do
    case "${RANGE_NAME}" in
        L20_31) LAYERS=$(seq 20 31 | tr '\n' ' ') ;;
        L25_31) LAYERS=$(seq 25 31 | tr '\n' ' ') ;;
        L28_31) LAYERS=$(seq 28 31 | tr '\n' ' ') ;;
    esac

    OUT_DIR="${PROJECT_ROOT}/output/cerdpo_restricted_${RANGE_NAME}_${DATE}/starcoder2_7b"
    LOG="${PROJECT_ROOT}/output/cerdpo_restricted_${RANGE_NAME}_${DATE}_train.log"
    echo "Training ${RANGE_NAME}: layers ${LAYERS}..."

    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/train_dpo_lora_restricted.py" \
        "${common_args[@]}" \
        --layers-to-transform ${LAYERS} \
        --output-dir "${OUT_DIR}" \
        > "${LOG}" 2>&1

    echo "Done: ${RANGE_NAME}"
done

echo "All layer-range training jobs done."
