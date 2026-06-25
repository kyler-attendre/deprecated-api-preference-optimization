#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run DeepSeek-Coder LoRA/SFT experiments on mixed_sft_v1 with nohup.

Default behavior:
  - run deepseek-coder-1.3b-instruct on GPU 1
  - run deepseek-coder-6.7b-instruct on GPU 4
  - each job runs train_lora.py, then eval_compare_lora.py

Usage:
  scripts/run_deepseek_mixed_sft_v1_nohup.sh [options]

Options:
  --models <list>        Comma-separated model sizes to run. Default: 1.3b,6.7b.
                         Accepted values: 1.3b, 6.7b.
  --gpu-1.3b <card_id>   GPU for DeepSeek-Coder-1.3B. Default: 1.
  --gpu-6.7b <card_id>   GPU for DeepSeek-Coder-6.7B. Default: 4.
  --mode <all|train|eval>
                         Run training only, evaluation only, or both. Default: all.
  --tag <name>           Output run tag. Default: deepseek_mixed_sft_v1_<timestamp>.
  --max-samples <n>      Evaluation sample cap; 0 means all test samples. Default: 0.
  --foreground           Do not launch with nohup; run selected jobs in current shell.
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
  MAX_LENGTH_1_3B        Train/eval max length for 1.3B. Default: 384.
  MAX_LENGTH_6_7B        Train/eval max length for 6.7B. Default: 384.

Examples:
  scripts/run_deepseek_mixed_sft_v1_nohup.sh
  scripts/run_deepseek_mixed_sft_v1_nohup.sh --models 1.3b --gpu-1.3b 1
  scripts/run_deepseek_mixed_sft_v1_nohup.sh --models 6.7b --gpu-6.7b 4 --tag deepseek_6_7b_card4
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODELS="1.3b,6.7b"
GPU_1_3B="1"
GPU_6_7B="4"
MODE="all"
TAG="deepseek_mixed_sft_v1_$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES="0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --gpu-1.3b)
      GPU_1_3B="$2"
      shift 2
      ;;
    --gpu-6.7b)
      GPU_6_7B="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
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
    1.3b) echo "${MODEL_PATH_DEEPSEEK_CODER_1_3B_INSTRUCT:-${MODEL_ROOT}/deepseek-ai/deepseek-coder-1.3b-instruct}" ;;
    6.7b) echo "${MODEL_PATH_DEEPSEEK_CODER_6_7B_INSTRUCT:-${MODEL_ROOT}/deepseek-ai/deepseek-coder-6.7b-instruct}" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

gpu_for() {
  case "$1" in
    1.3b) echo "${GPU_1_3B}" ;;
    6.7b) echo "${GPU_6_7B}" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

max_length_for() {
  case "$1" in
    1.3b) echo "${MAX_LENGTH_1_3B:-384}" ;;
    6.7b) echo "${MAX_LENGTH_6_7B:-384}" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

label_for() {
  case "$1" in
    1.3b) echo "deepseek_coder_1_3b_instruct" ;;
    6.7b) echo "deepseek_coder_6_7b_instruct" ;;
    *)
      echo "Unknown model key: $1" >&2
      return 1
      ;;
  esac
}

run_one_model() {
  local model_key="$1"
  local model_path="$2"
  local gpu="$3"
  local label="$4"
  local max_length="$5"

  local adapter_dir="${RUN_ROOT}/${label}_lora"
  local compare_dir="${RUN_ROOT}/${label}_compare"
  local train_log="${LOG_DIR}/${label}_train.log"
  local eval_log="${LOG_DIR}/${label}_eval.log"

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  echo "============================================================"
  echo "Start model: ${label}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "GPU: ${gpu}"
  echo "Model path: ${model_path}"
  echo "Adapter dir: ${adapter_dir}"
  echo "Compare dir: ${compare_dir}"
  echo "Max length: ${max_length}"
  echo "Mode: ${MODE}"

  if [[ ! -d "${model_path}" ]]; then
    echo "Model directory not found: ${model_path}" >&2
    exit 1
  fi

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

  echo "Finished model: ${label}"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
}

start_one_model() {
  local model_key="$1"
  local model_path
  local gpu
  local label
  local max_length

  model_path="$(model_path_for "${model_key}")"
  gpu="$(gpu_for "${model_key}")"
  label="$(label_for "${model_key}")"
  max_length="$(max_length_for "${model_key}")"

  if [[ "${FOREGROUND}" == "1" ]]; then
    run_one_model "${model_key}" "${model_path}" "${gpu}" "${label}" "${max_length}"
    return
  fi

  local main_log="${LOG_DIR}/${label}_nohup_main.log"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EPOCHS="${EPOCHS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
    LORA_R="${LORA_R}" \
    LORA_ALPHA="${LORA_ALPHA}" \
    LORA_DROPOUT="${LORA_DROPOUT}" \
    LOGGING_STEPS="${LOGGING_STEPS}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    MAX_LENGTH_1_3B="${MAX_LENGTH_1_3B:-384}" \
    MAX_LENGTH_6_7B="${MAX_LENGTH_6_7B:-384}" \
    bash "$0" \
      --models "${model_key}" \
      --gpu-1.3b "${GPU_1_3B}" \
      --gpu-6.7b "${GPU_6_7B}" \
      --mode "${MODE}" \
      --tag "${TAG}" \
      --max-samples "${MAX_SAMPLES}" \
      --foreground \
    > "${main_log}" 2>&1 &

  echo "Started ${label}"
  echo "PID: $!"
  echo "GPU: ${gpu}"
  echo "Main log: ${main_log}"
}

echo "DeepSeek mixed_sft_v1 launcher"
echo "Run root: ${RUN_ROOT}"
echo "Mode: ${MODE}"
echo "Models: ${MODELS}"

IFS=',' read -r -a REQUESTED_MODELS <<< "${MODELS}"
for raw_key in "${REQUESTED_MODELS[@]}"; do
  model_key="$(echo "${raw_key}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  start_one_model "${model_key}"
done

if [[ "${FOREGROUND}" != "1" ]]; then
  echo "All requested jobs submitted."
  echo "Output root: ${RUN_ROOT}"
fi
