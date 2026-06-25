#!/usr/bin/env bash
# Version context ablation experiment
#
# Step A: eval existing CER-DPO models with probing_input (no version context)
# Step B: train CER-DPO-7B without version context, then eval with both prompt types
#
# 2×2 matrix per model:
#   train_with_version   + eval_with_version    → already have (version_prompt)
#   train_with_version   + eval_without_version → Step A
#   train_without_version + eval_with_version   → Step B
#   train_without_version + eval_without_version→ Step B
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATE="20260519"
TEST_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_test.jsonl"
TRAIN_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_val.jsonl"
OUT_ROOT="${PROJECT_ROOT}/output/version_context_ablation_${DATE}"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

mkdir -p "${OUT_ROOT}"

# ─────────────────────────────────────────────
# STEP A: eval existing models with probing_input
# ─────────────────────────────────────────────

eval_no_version() {
    local GPU=$1 MODEL_PATH=$2 ADAPTER_DIR=$3 LABEL=$4
    local OUT="${OUT_ROOT}/${LABEL}/train_with_version__eval_no_version"
    if [[ -f "${OUT}/comparison_summary.json" ]]; then
        echo "Skip (exists): ${LABEL} no_version eval"
        return
    fi
    mkdir -p "${OUT}"
    echo "[GPU ${GPU}] Step-A eval: ${LABEL} (probing_input) ..."
    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/eval_compare_lora.py" \
        --model-name-or-path "${MODEL_PATH}" \
        --adapter-dir "${ADAPTER_DIR}" \
        --test-file "${TEST_FILE}" \
        --output-dir "${OUT}" \
        --prompt-field probing_input \
        --max-length 384 \
        --max-new-tokens 64 \
        > "${OUT}/eval.log" 2>&1
    echo "[GPU ${GPU}] Done: ${LABEL} no_version eval"
}

# All CER-DPO adapters: (gpu, model_path, adapter_dir, label)
declare -a STEP_A_JOBS=(
    "0 /data/models/StarCoder/starcoder2-7b      ${PROJECT_ROOT}/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b          starcoder2_7b_cerdpo"
    "0 /data/models/StarCoder/starcoder2-7b      ${PROJECT_ROOT}/output/dpo_lora_mixed_sft_v1_20260422/starcoder2_7b               starcoder2_7b_dpo"
    "1 /data/models/StarCoder/starcoder2-3b      ${PROJECT_ROOT}/output/dpo_anchor_full01_20260423/starcoder2_3b                   starcoder2_3b_cerdpo"
    "2 /data/models/StarCoder/starcoder2-15b     ${PROJECT_ROOT}/output/dpo_anchor_full01_20260423/starcoder2_15b                  starcoder2_15b_cerdpo"
    "3 /data/models/deepseek-ai/deepseek-coder-6.7b-instruct ${PROJECT_ROOT}/output/dpo_anchor_full01_20260423/deepseek_coder_6_7b_instruct deepseek_6_7b_cerdpo"
    "1 /data/models/Qwen/Qwen2.5-Coder-3B-Instruct ${PROJECT_ROOT}/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_3b_instruct qwen_3b_cerdpo"
    "3 /data/models/Qwen/Qwen2.5-Coder-7B-Instruct ${PROJECT_ROOT}/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_7b_instruct qwen_7b_cerdpo"
    "2 /data/models/Qwen/Qwen2.5-Coder-14B-Instruct ${PROJECT_ROOT}/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_14b_instruct qwen_14b_cerdpo"
)

echo "=== Step A: eval existing models without version context ==="
for JOB in "${STEP_A_JOBS[@]}"; do
    read -r GPU MODEL_PATH ADAPTER_DIR LABEL <<< "${JOB}"
    eval_no_version "${GPU}" "${MODEL_PATH}" "${ADAPTER_DIR}" "${LABEL}" &
done
wait
echo "Step A complete."

# ─────────────────────────────────────────────
# STEP B: train CER-DPO-7B with probing_input, then eval
# (only StarCoder2-7B as representative model)
# ─────────────────────────────────────────────

echo ""
echo "=== Step B: train CER-DPO-7B without version context ==="

NO_VER_ADAPTER="${PROJECT_ROOT}/output/cerdpo_no_version_${DATE}/starcoder2_7b"

if [[ ! -f "${NO_VER_ADAPTER}/adapter_config.json" ]]; then
    echo "Training CER-DPO-7B (no version) on GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_dpo_lora.py" \
        --model-name-or-path /data/models/StarCoder/starcoder2-7b \
        --train-file "${TRAIN_FILE}" \
        --val-file "${VAL_FILE}" \
        --output-dir "${NO_VER_ADAPTER}" \
        --prompt-field probing_input \
        --api-anchor-weight 0.1 \
        --num-train-epochs 1 \
        --per-device-train-batch-size 1 \
        --gradient-accumulation-steps 8 \
        --learning-rate 5e-5 \
        --max-length 384 \
        --lora-r 8 \
        --lora-alpha 16 \
        2>&1 | tee "${OUT_ROOT}/train_no_version_7b.log"
    echo "Training done."
else
    echo "Adapter exists, skip training."
fi

echo "Evaluating CER-DPO-7B (no version) with version_prompt ..."
eval_no_version_b() {
    local GPU=$1 PROMPT_FIELD=$2 LABEL=$3
    local OUT="${OUT_ROOT}/starcoder2_7b_cerdpo_no_version/${LABEL}"
    if [[ -f "${OUT}/comparison_summary.json" ]]; then
        echo "Skip (exists): starcoder2_7b_cerdpo_no_version ${LABEL}"
        return
    fi
    mkdir -p "${OUT}"
    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/eval_compare_lora.py" \
        --model-name-or-path /data/models/StarCoder/starcoder2-7b \
        --adapter-dir "${NO_VER_ADAPTER}" \
        --test-file "${TEST_FILE}" \
        --output-dir "${OUT}" \
        --prompt-field "${PROMPT_FIELD}" \
        --max-length 384 \
        --max-new-tokens 64 \
        > "${OUT}/eval.log" 2>&1
    echo "Done: starcoder2_7b_cerdpo_no_version ${LABEL}"
}

# Run both eval conditions in parallel on different GPUs
eval_no_version_b 0 version_prompt  "eval_with_version" &
eval_no_version_b 1 probing_input   "eval_no_version" &
wait

echo ""
echo "=== All done. Summary ==="
echo ""
echo "Results in: ${OUT_ROOT}"
echo ""

# Print summary table
python3 - << 'EOF'
import json
from pathlib import Path
import os

out_root = Path(os.environ.get("OUT_ROOT", ""))
DATE = os.environ.get("DATE", "20260519")
out_root = Path(f"/workspace/version-control-study/05_positive_engineering/output/version_context_ablation_{DATE}")

print(f"{'Model':<35s}  {'Train prompt':<18s}  {'Eval prompt':<18s}  dep%   rep%")
print("-" * 95)

# Step A results
for p in sorted(out_root.glob("*/train_with_version__eval_no_version/comparison_summary.json")):
    label = p.parent.parent.name
    d = json.loads(p.read_text())
    dep = d["lora"]["deprecated_usage_rate"] * 100
    rep = d["lora"]["replacement_hit_rate"] * 100
    print(f"  {label:<33s}  {'WITH version':<18s}  {'WITHOUT version':<18s}  {dep:5.1f}  {rep:5.1f}")

# Step B results
for subdir, train_label, eval_label in [
    ("starcoder2_7b_cerdpo_no_version/eval_with_version", "WITHOUT version", "WITH version"),
    ("starcoder2_7b_cerdpo_no_version/eval_no_version", "WITHOUT version", "WITHOUT version"),
]:
    p = out_root / subdir / "comparison_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        dep = d["lora"]["deprecated_usage_rate"] * 100
        rep = d["lora"]["replacement_hit_rate"] * 100
        print(f"  {'starcoder2_7b_cerdpo_no_version':<33s}  {train_label:<18s}  {eval_label:<18s}  {dep:5.1f}  {rep:5.1f}")
EOF
