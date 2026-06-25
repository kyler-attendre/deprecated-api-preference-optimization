#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run Qwen2.5-Coder LoRA/SFT experiments on mixed_sft_v1 with nohup.

Usage:
  scripts/run_qwen_mixed_sft_v1_nohup.sh --gpu <card_id> [options]

Required:
  --gpu <card_id>        GPU card id passed to CUDA_VISIBLE_DEVICES.

Options:
  --mode <all|train|eval>
                         Run training only, evaluation only, or both. Default: all.
  --models <list>        Comma-separated model sizes to run. Default: 3b,7b,14b.
                         Accepted values: 3b, 7b, 14b.
  --tag <name>           Output run tag. Default: qwen2_5_mixed_sft_v1_<timestamp>.
  --max-samples <n>      Evaluation sample cap; 0 means all test samples. Default: 0.
  --foreground           Do not launch with nohup; run in the current shell.
  -h, --help             Show this help.

Environment overrides:
  PYTHON_BIN             Python executable. Default: python3.
  EPOCHS                 LoRA training epochs. Default: 3.
  LEARNING_RATE          Learning rate. Default: 1e-4.
  TRAIN_BATCH_SIZE       Per-device train batch size. Default: 1.
  EVAL_BATCH_SIZE        Per-device eval batch size. Default: 1.
  GRAD_ACCUM_STEPS       Gradient accumulation steps. Default: 16.
  LORA_R                 LoRA rank. Default: 16.
  LORA_ALPHA             LoRA alpha. Default: 32.
  LORA_DROPOUT           LoRA dropout. Default: 0.05.
  LOGGING_STEPS          Training logging steps. Default: 20.
  MAX_NEW_TOKENS         Evaluation generation length. Default: 64.
  MAX_LENGTH_3B          Train/eval max length for 3B. Default: 384.
  MAX_LENGTH_7B          Train/eval max length for 7B. Default: 384.
  MAX_LENGTH_14B         Train/eval max length for 14B. Default: 256.

Examples:
  scripts/run_qwen_mixed_sft_v1_nohup.sh --gpu 0
  scripts/run_qwen_mixed_sft_v1_nohup.sh --gpu 1 --models 3b,7b --tag qwen_card1
  scripts/run_qwen_mixed_sft_v1_nohup.sh --gpu 2 --mode eval --tag qwen_card2
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU=""
MODE="all"
MODELS="3b,7b,14b"
TAG="qwen2_5_mixed_sft_v1_$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES="0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${GPU}" ]]; then
  echo "Missing required argument: --gpu <card_id>" >&2
  usage >&2
  exit 2
fi

case "${MODE}" in
  all|train|eval) ;;
  *)
    echo "Invalid --mode: ${MODE}. Expected all, train, or eval." >&2
    exit 2
    ;;
esac

RUN_ROOT="${PROJECT_ROOT}/output/${TAG}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${FOREGROUND}" != "1" && "${RUN_UNDER_NOHUP:-0}" != "1" ]]; then
  MAIN_LOG="${LOG_DIR}/nohup_main.log"
  nohup env \
    RUN_UNDER_NOHUP=1 \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    bash "$0" \
      --gpu "${GPU}" \
      --mode "${MODE}" \
      --models "${MODELS}" \
      --tag "${TAG}" \
      --max-samples "${MAX_SAMPLES}" \
      --foreground \
    > "${MAIN_LOG}" 2>&1 &

  echo "Started Qwen mixed_sft_v1 job."
  echo "PID: $!"
  echo "GPU: ${GPU}"
  echo "Run root: ${RUN_ROOT}"
  echo "Main log: ${MAIN_LOG}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

TRAIN_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_val.jsonl"
TEST_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_test.jsonl"

for required_file in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required dataset file not found: ${required_file}" >&2
    exit 1
  fi
done

model_path_for() {
  case "$1" in
    3b) echo "${MODEL_PATH_QWEN2_5_CODER_3B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-3B-Instruct}" ;;
    7b) echo "${MODEL_PATH_QWEN2_5_CODER_7B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-7B-Instruct}" ;;
    14b) echo "${MODEL_PATH_QWEN2_5_CODER_14B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-14B-Instruct}" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

max_length_for() {
  case "$1" in
    3b) echo "${MAX_LENGTH_3B:-384}" ;;
    7b) echo "${MAX_LENGTH_7B:-384}" ;;
    14b) echo "${MAX_LENGTH_14B:-256}" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

label_for() {
  case "$1" in
    3b) echo "qwen2_5_coder_3b_instruct" ;;
    7b) echo "qwen2_5_coder_7b_instruct" ;;
    14b) echo "qwen2_5_coder_14b_instruct" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

echo "Run started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Project root: ${PROJECT_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "GPU: ${GPU}"
echo "Mode: ${MODE}"
echo "Models: ${MODELS}"
echo "Python: ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a REQUESTED_MODELS <<< "${MODELS}"

for raw_key in "${REQUESTED_MODELS[@]}"; do
  model_key="$(echo "${raw_key}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  model_path="$(model_path_for "${model_key}")"
  label="$(label_for "${model_key}")"
  max_length="$(max_length_for "${model_key}")"

  if [[ ! -d "${model_path}" ]]; then
    echo "Model directory not found: ${model_path}" >&2
    exit 1
  fi

  adapter_dir="${RUN_ROOT}/${label}_lora"
  compare_dir="${RUN_ROOT}/${label}_compare"
  train_log="${LOG_DIR}/${label}_train.log"
  eval_log="${LOG_DIR}/${label}_eval.log"

  echo "============================================================"
  echo "Model: ${label}"
  echo "Model path: ${model_path}"
  echo "Adapter dir: ${adapter_dir}"
  echo "Compare dir: ${compare_dir}"
  echo "Max length: ${max_length}"

  if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
    echo "Training ${label}; log: ${train_log}"
    "${PYTHON_BIN}" scripts/train_lora.py \
      --train-file "${TRAIN_FILE}" \
      --val-file "${VAL_FILE}" \
      --model-name-or-path "${model_path}" \
      --output-dir "${adapter_dir}" \
      --max-length "${max_length}" \
      --learning-rate "${LEARNING_RATE}" \
      --num-train-epochs "${EPOCHS}" \
      --per-device-train-batch-size "${TRAIN_BATCH_SIZE}" \
      --per-device-eval-batch-size "${EVAL_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
      --logging-steps "${LOGGING_STEPS}" \
      --lora-r "${LORA_R}" \
      --lora-alpha "${LORA_ALPHA}" \
      --lora-dropout "${LORA_DROPOUT}" \
      > "${train_log}" 2>&1
  fi

  if [[ "${MODE}" == "all" || "${MODE}" == "eval" ]]; then
    if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
      echo "Adapter not found for eval: ${adapter_dir}/adapter_config.json" >&2
      echo "Run with --mode train first, or use the same --tag that contains trained adapters." >&2
      exit 1
    fi

    mkdir -p "${compare_dir}"
    echo "Evaluating ${label}; log: ${eval_log}"
    "${PYTHON_BIN}" scripts/eval_compare_lora.py \
      --model-name-or-path "${model_path}" \
      --adapter-dir "${adapter_dir}" \
      --test-file "${TEST_FILE}" \
      --output-dir "${compare_dir}" \
      --max-length "${max_length}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-samples "${MAX_SAMPLES}" \
      > "${eval_log}" 2>&1
  fi

  echo "Finished ${label}: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Outputs: ${RUN_ROOT}"
