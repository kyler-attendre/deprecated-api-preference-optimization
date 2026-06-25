#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MECH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$MECH_ROOT/.." && pwd)

CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lkl_llm}

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_id> <model_key>"
  exit 1
fi

GPU_ID=$1
MODEL_KEY=$2

case "$MODEL_KEY" in
  starcoder2_3b)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_20260423/starcoder2_3b"
    BATCH_SIZE=4
    MAX_LENGTH=384
    ;;
  starcoder2_7b)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo7b_screen_full_anchor01_20260423/starcoder2_7b"
    BATCH_SIZE=2
    MAX_LENGTH=384
    ;;
  starcoder2_15b)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_20260423/starcoder2_15b"
    BATCH_SIZE=1
    MAX_LENGTH=256
    ;;
  deepseek_coder_6_7b_instruct)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_20260423/deepseek_coder_6_7b_instruct"
    BATCH_SIZE=2
    MAX_LENGTH=384
    ;;
  qwen2_5_coder_3b_instruct)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_3b_instruct"
    BATCH_SIZE=4
    MAX_LENGTH=384
    ;;
  qwen2_5_coder_7b_instruct)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_7b_instruct"
    BATCH_SIZE=2
    MAX_LENGTH=384
    ;;
  qwen2_5_coder_14b_instruct)
    ADAPTER_DIR="$PROJECT_ROOT/05_positive_engineering/output/dpo_anchor_full01_qwen_20260423/qwen2_5_coder_14b_instruct"
    BATCH_SIZE=1
    MAX_LENGTH=256
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY"
    exit 1
    ;;
esac

OUT_ROOT="$MECH_ROOT/output/full_mechanism_20260427/$MODEL_KEY"
BASE_LENS="$OUT_ROOT/tuned_lens/${MODEL_KEY}_official_base.pt"
ADAPTER_LENS="$OUT_ROOT/tuned_lens/${MODEL_KEY}_anchored_dpo.pt"
COMPARE_DIR="$OUT_ROOT/compare"

mkdir -p "$OUT_ROOT/tuned_lens" "$COMPARE_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$MECH_ROOT"

echo "[$(date '+%F %T')] Training base tuned lens for $MODEL_KEY on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_tuned_lens.py \
  --model-key "$MODEL_KEY" \
  --output-file "$BASE_LENS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH"

echo "[$(date '+%F %T')] Training anchored DPO tuned lens for $MODEL_KEY on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train_tuned_lens.py \
  --model-key "$MODEL_KEY" \
  --adapter-dir "$ADAPTER_DIR" \
  --output-file "$ADAPTER_LENS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH"

echo "[$(date '+%F %T')] Running lens comparison for $MODEL_KEY on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_lens_compare.py \
  --model-key "$MODEL_KEY" \
  --adapter-dir "$ADAPTER_DIR" \
  --base-tuned-lens "$BASE_LENS" \
  --adapter-tuned-lens "$ADAPTER_LENS" \
  --output-dir "$COMPARE_DIR" \
  --max-length "$MAX_LENGTH"

echo "[$(date '+%F %T')] Completed $MODEL_KEY"
