#!/usr/bin/env bash
# Task 1 & Task 2a: module ablation training
# GPU 0: MLP-only
# GPU 1: Attn+MLP
# GPU 2: o_proj-only
# GPU 3: k+v_proj
# GPU 4: q+k+v (no o)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_val.jsonl"
MODEL="/data/models/StarCoder/starcoder2-7b"
DATE="20260511"

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
)

# Task 1: MLP-only (GPU 0)
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_dpo_lora.py" \
    "${common_args[@]}" \
    --target-modules c_fc c_proj \
    --output-dir "${PROJECT_ROOT}/output/cerdpo_mlp_only_${DATE}/starcoder2_7b" \
    > "${PROJECT_ROOT}/output/cerdpo_mlp_only_${DATE}_train.log" 2>&1 &

# Task 1: Attn+MLP (GPU 1)
CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/train_dpo_lora.py" \
    "${common_args[@]}" \
    --target-modules q_proj k_proj v_proj o_proj c_fc c_proj \
    --output-dir "${PROJECT_ROOT}/output/cerdpo_attn_mlp_${DATE}/starcoder2_7b" \
    > "${PROJECT_ROOT}/output/cerdpo_attn_mlp_${DATE}_train.log" 2>&1 &

# Task 2a: o_proj only (GPU 2)
CUDA_VISIBLE_DEVICES=2 python "${SCRIPT_DIR}/train_dpo_lora.py" \
    "${common_args[@]}" \
    --target-modules o_proj \
    --output-dir "${PROJECT_ROOT}/output/cerdpo_o_only_${DATE}/starcoder2_7b" \
    > "${PROJECT_ROOT}/output/cerdpo_o_only_${DATE}_train.log" 2>&1 &

# Task 2a: k+v only (GPU 3)
CUDA_VISIBLE_DEVICES=3 python "${SCRIPT_DIR}/train_dpo_lora.py" \
    "${common_args[@]}" \
    --target-modules k_proj v_proj \
    --output-dir "${PROJECT_ROOT}/output/cerdpo_kv_only_${DATE}/starcoder2_7b" \
    > "${PROJECT_ROOT}/output/cerdpo_kv_only_${DATE}_train.log" 2>&1 &

# Task 2a: q+k+v (no o) (GPU 4)
CUDA_VISIBLE_DEVICES=4 python "${SCRIPT_DIR}/train_dpo_lora.py" \
    "${common_args[@]}" \
    --target-modules q_proj k_proj v_proj \
    --output-dir "${PROJECT_ROOT}/output/cerdpo_qkv_no_o_${DATE}/starcoder2_7b" \
    > "${PROJECT_ROOT}/output/cerdpo_qkv_no_o_${DATE}_train.log" 2>&1 &

echo "Launched 5 training jobs on GPUs 0-4. Logs in output/cerdpo_*_${DATE}_train.log"
wait
echo "All training jobs done."
