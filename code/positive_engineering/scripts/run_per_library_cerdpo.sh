#!/usr/bin/env bash
# run_per_library_cerdpo.sh
#
# 为每个库单独训一个 CER-DPO LoRA，然后评测。
# 与现有 run_dpo_lora_nohup.sh 使用相同的 train_dpo_lora.py，
# 仅把 --train-file / --val-file 换成各库的子集文件。
#
# Usage:
#   bash run_per_library_cerdpo.sh --gpu GPU --model-key MODEL_KEY [--library LIB] [--mode train|eval|all] [--tag TAG]
#
# Examples:
#   # 跑某个模型的全部8个库（顺序执行）
#   bash run_per_library_cerdpo.sh --gpu 0 --model-key starcoder2_3b
#
#   # 只跑某个库
#   bash run_per_library_cerdpo.sh --gpu 1 --model-key deepseek_coder_6_7b_instruct --library pytorch
#
#   # 只评测（已有训练结果）
#   bash run_per_library_cerdpo.sh --gpu 0 --model-key starcoder2_3b --mode eval --tag per_library_cerdpo_20260601
set -euo pipefail

GPU=""
MODEL_KEY=""
LIBRARY=""          # 空 = 跑全部库
MODE="all"          # train | eval | all
TAG="per_library_cerdpo_$(date +%Y%m%d_%H%M%S)"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"

# CER-DPO 超参（沿用全局 CER-DPO 配置）
EPOCHS="3"
LR="5e-5"
BETA="0.1"
API_ANCHOR_WEIGHT="0.1"
DPO_SCOPE="full"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
GRAD_ACCUM="8"
LOGGING_STEPS="10"
MAX_NEW_TOKENS="64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)              GPU="$2";        shift 2 ;;
    --model-key)        MODEL_KEY="$2";  shift 2 ;;
    --library)          LIBRARY="$2";   shift 2 ;;
    --mode)             MODE="$2";      shift 2 ;;
    --tag)              TAG="$2";       shift 2 ;;
    --epochs)           EPOCHS="$2";    shift 2 ;;
    --lr|--learning-rate) LR="$2";      shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$GPU" ]]       && { echo "--gpu is required" >&2;       exit 2; }
[[ -z "$MODEL_KEY" ]] && { echo "--model-key is required" >&2; exit 2; }

case "$MODEL_KEY" in
  starcoder2_3b)
    MODEL_PATH="${MODEL_ROOT}/StarCoder/starcoder2-3b"
    MAX_LENGTH="384" ;;
  starcoder2_7b)
    MODEL_PATH="${MODEL_ROOT}/StarCoder/starcoder2-7b"
    MAX_LENGTH="384" ;;
  starcoder2_15b)
    MODEL_PATH="${MODEL_ROOT}/StarCoder/starcoder2-15b"
    MAX_LENGTH="256" ;;
  deepseek_coder_6_7b_instruct)
    MODEL_PATH="${MODEL_ROOT}/deepseek-ai/deepseek-coder-6.7b-instruct"
    MAX_LENGTH="384" ;;
  qwen2_5_coder_3b_instruct)
    MODEL_PATH="${MODEL_ROOT}/Qwen/Qwen2.5-Coder-3B-Instruct"
    MAX_LENGTH="384" ;;
  qwen2_5_coder_7b_instruct)
    MODEL_PATH="${MODEL_ROOT}/Qwen/Qwen2.5-Coder-7B-Instruct"
    MAX_LENGTH="384" ;;
  qwen2_5_coder_14b_instruct)
    MODEL_PATH="${MODEL_ROOT}/Qwen/Qwen2.5-Coder-14B-Instruct"
    MAX_LENGTH="256" ;;
  *) echo "Unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BY_LIB_DIR="$PROJECT_DIR/data/mixed_sft_v1/by_library"
OUTPUT_ROOT="$PROJECT_DIR/output/$TAG"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lkl_llm}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

# 收集要跑的库
if [[ -n "$LIBRARY" ]]; then
  LIBRARIES=("$LIBRARY")
else
  LIBRARIES=($(ls "$BY_LIB_DIR"))
fi

echo "======================================================"
echo "Per-library CER-DPO"
echo "  model:     $MODEL_KEY"
echo "  gpu:       $GPU"
echo "  mode:      $MODE"
echo "  tag:       $TAG"
echo "  libraries: ${LIBRARIES[*]}"
echo "  epochs:    $EPOCHS  lr=$LR  beta=$BETA  anchor=$API_ANCHOR_WEIGHT"
echo "======================================================"

for lib in "${LIBRARIES[@]}"; do
  TRAIN_FILE="$BY_LIB_DIR/$lib/mixed_sft_train.jsonl"
  VAL_FILE="$BY_LIB_DIR/$lib/mixed_sft_val.jsonl"
  TEST_FILE="$BY_LIB_DIR/$lib/mixed_sft_test.jsonl"

  if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "[skip] $lib: train file not found"
    continue
  fi

  # val DPO pairs 检查：为0时用 train 文件替代 val（仅针对极小库）
  if [[ ! -f "$VAL_FILE" ]] || [[ ! -s "$VAL_FILE" ]]; then
    echo "   [warn] $lib: val file missing or empty — using train as val"
    VAL_FILE="$TRAIN_FILE"
  else
    val_dpo=$(python3 -c "
import json, sys; sys.path.insert(0, 'src')
from dpo_training import build_dpo_pairs
rows = [json.loads(l) for l in open('$VAL_FILE') if l.strip()]
print(len(build_dpo_pairs(rows)))
" 2>/dev/null || echo "0")
    if [[ "$val_dpo" == "0" ]]; then
      echo "   [warn] $lib: val has 0 DPO pairs — using train file as val"
      VAL_FILE="$TRAIN_FILE"
    fi
  fi

  ADAPTER_DIR="$OUTPUT_ROOT/${MODEL_KEY}/${lib}"
  COMPARE_DIR="$ADAPTER_DIR/compare"
  TRAIN_LOG="$LOG_DIR/${MODEL_KEY}_${lib}_train.log"
  EVAL_LOG="$LOG_DIR/${MODEL_KEY}_${lib}_eval.log"

  echo ""
  echo "── library: $lib ──────────────────────────────────"
  echo "   adapter → $ADAPTER_DIR"

  # ── Training ───────────────────────────────────────────
  if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    train_count=$(wc -l < "$TRAIN_FILE")
    echo "   train samples: $train_count"

    # 预检：确认能构建至少1条 DPO pair，否则跳过
    dpo_count=$(python3 -c "
import json, sys
sys.path.insert(0, 'src')
from dpo_training import build_dpo_pairs
rows = [json.loads(l) for l in open('$TRAIN_FILE') if l.strip()]
print(len(build_dpo_pairs(rows)))
" 2>/dev/null || echo "0")
    if [[ "$dpo_count" == "0" ]]; then
      echo "   [skip] 0 DPO pairs constructable from $train_count samples — skipping $lib"
      continue
    fi
    echo "   dpo pairs: $dpo_count"
    echo "   training..."

    python3 scripts/train_dpo_lora.py \
      --train-file       "$TRAIN_FILE" \
      --val-file         "$VAL_FILE" \
      --model-name-or-path "$MODEL_PATH" \
      --output-dir       "$ADAPTER_DIR" \
      --max-length       "$MAX_LENGTH" \
      --learning-rate    "$LR" \
      --num-train-epochs "$EPOCHS" \
      --per-device-train-batch-size 1 \
      --per-device-eval-batch-size  1 \
      --gradient-accumulation-steps "$GRAD_ACCUM" \
      --logging-steps    "$LOGGING_STEPS" \
      --save-strategy    epoch \
      --eval-strategy    epoch \
      --save-total-limit 2 \
      --beta             "$BETA" \
      --dpo-scope        "$DPO_SCOPE" \
      --api-anchor-weight "$API_ANCHOR_WEIGHT" \
      --logprob-reduction sum \
      --lora-r           "$LORA_R" \
      --lora-alpha       "$LORA_ALPHA" \
      --lora-dropout     "$LORA_DROPOUT" \
      --gradient-checkpointing \
      > "$TRAIN_LOG" 2>&1

    echo "   training done → $TRAIN_LOG"
  fi

  # ── Evaluation ─────────────────────────────────────────
  if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
    if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
      echo "   [skip eval] adapter not found: $ADAPTER_DIR/adapter_config.json"
      continue
    fi
    if [[ ! -f "$TEST_FILE" ]]; then
      echo "   [skip eval] test file not found: $TEST_FILE"
      continue
    fi

    mkdir -p "$COMPARE_DIR"
    test_count=$(wc -l < "$TEST_FILE")
    echo "   test samples: $test_count"
    echo "   evaluating..."

    python3 scripts/eval_compare_lora.py \
      --model-name-or-path "$MODEL_PATH" \
      --adapter-dir        "$ADAPTER_DIR" \
      --test-file          "$TEST_FILE" \
      --output-dir         "$COMPARE_DIR" \
      --max-length         "$MAX_LENGTH" \
      --max-new-tokens     "$MAX_NEW_TOKENS" \
      --max-samples        0 \
      > "$EVAL_LOG" 2>&1

    echo "   eval done → $EVAL_LOG"

    # 打印简要结果
    python3 -c "
import json, pathlib
f = pathlib.Path('$COMPARE_DIR/comparison_summary.json')
if f.exists():
    r = json.load(f.open())
    base = r.get('base', {})
    lora = r.get('lora', {})
    delta = r.get('delta', {})
    print(f'   dep:  base={base.get(\"deprecated_usage_rate\",0)*100:.1f}%  lora={lora.get(\"deprecated_usage_rate\",0)*100:.1f}%  Δ={delta.get(\"deprecated_usage_rate\",0)*100:+.1f}%')
    print(f'   rep:  base={base.get(\"replacement_hit_rate\",0)*100:.1f}%  lora={lora.get(\"replacement_hit_rate\",0)*100:.1f}%  Δ={delta.get(\"replacement_hit_rate\",0)*100:+.1f}%')
" 2>/dev/null || true
  fi
done

echo ""
echo "======================================================"
echo "Done: $MODEL_KEY  tag=$TAG"
echo "Outputs: $OUTPUT_ROOT"
