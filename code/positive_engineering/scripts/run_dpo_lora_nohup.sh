#!/usr/bin/env bash
set -euo pipefail

GPU=""
MODEL_KEY="starcoder2_3b"
TAG="dpo_lora_mixed_sft_v1_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS="-1"
EPOCHS="1"
LR="5e-5"
BETA="0.1"
API_ANCHOR_WEIGHT="0.0"
DPO_SCOPE="full"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
TARGET_MODULES=()
LAYERS_TO_TRANSFORM=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --model-key)
      MODEL_KEY="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --learning-rate)
      LR="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --api-anchor-weight)
      API_ANCHOR_WEIGHT="$2"
      shift 2
      ;;
    --dpo-scope)
      DPO_SCOPE="$2"
      shift 2
      ;;
    --target-modules)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TARGET_MODULES+=("$1")
        shift
      done
      ;;
    --layers-to-transform)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        LAYERS_TO_TRANSFORM+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GPU" ]]; then
  echo "--gpu is required" >&2
  exit 2
fi

case "$MODEL_KEY" in
  starcoder2_3b)
    MODEL_PATH="${MODEL_PATH_STARCODER2_3B:-${MODEL_ROOT}/StarCoder/starcoder2-3b}"
    MAX_LENGTH="384"
    ;;
  starcoder2_7b)
    MODEL_PATH="${MODEL_PATH_STARCODER2_7B:-${MODEL_ROOT}/StarCoder/starcoder2-7b}"
    MAX_LENGTH="384"
    ;;
  starcoder2_15b)
    MODEL_PATH="${MODEL_PATH_STARCODER2_15B:-${MODEL_ROOT}/StarCoder/starcoder2-15b}"
    MAX_LENGTH="256"
    ;;
  deepseek_coder_6_7b_instruct)
    MODEL_PATH="${MODEL_PATH_DEEPSEEK_CODER_6_7B_INSTRUCT:-${MODEL_ROOT}/deepseek-ai/deepseek-coder-6.7b-instruct}"
    MAX_LENGTH="384"
    ;;
  qwen2_5_coder_3b_instruct)
    MODEL_PATH="${MODEL_PATH_QWEN2_5_CODER_3B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-3B-Instruct}"
    MAX_LENGTH="384"
    ;;
  qwen2_5_coder_7b_instruct)
    MODEL_PATH="${MODEL_PATH_QWEN2_5_CODER_7B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-7B-Instruct}"
    MAX_LENGTH="384"
    ;;
  qwen2_5_coder_14b_instruct)
    MODEL_PATH="${MODEL_PATH_QWEN2_5_CODER_14B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-14B-Instruct}"
    MAX_LENGTH="256"
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_DIR/output/$TAG"
OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_KEY"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/${MODEL_KEY}_gpu${GPU}.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ -f "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}" ]]; then
  source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-lkl_llm}"
fi
cd "$PROJECT_DIR"

echo "[$(date)] Starting DPO LoRA: model=$MODEL_KEY gpu=$GPU output=$OUTPUT_DIR dpo_scope=$DPO_SCOPE api_anchor_weight=$API_ANCHOR_WEIGHT" | tee "$LOG_FILE"

EXTRA_ARGS=()
if [[ ${#TARGET_MODULES[@]} -gt 0 ]]; then
  EXTRA_ARGS+=(--target-modules "${TARGET_MODULES[@]}")
fi
if [[ ${#LAYERS_TO_TRANSFORM[@]} -gt 0 ]]; then
  EXTRA_ARGS+=(--layers-to-transform "${LAYERS_TO_TRANSFORM[@]}")
fi

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 scripts/train_dpo_lora.py \
  --train-file data/mixed_sft_v1/mixed_sft_train.jsonl \
  --val-file data/mixed_sft_v1/mixed_sft_val.jsonl \
  --model-name-or-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "$MAX_LENGTH" \
  --learning-rate "$LR" \
  --num-train-epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --logging-steps 10 \
  --save-strategy epoch \
  --eval-strategy epoch \
  --save-total-limit 2 \
  --beta "$BETA" \
  --dpo-scope "$DPO_SCOPE" \
  --api-anchor-weight "$API_ANCHOR_WEIGHT" \
  --logprob-reduction sum \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1 &

echo "$!" | tee "$OUTPUT_DIR/pid.txt"
echo "Log: $LOG_FILE"
