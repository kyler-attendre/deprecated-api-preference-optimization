#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run the LoRA prompt-field ablation grid.

For each requested model this script:
  1. trains a LoRA adapter on mixed_sft_v1 with --prompt-field probing_input
     (the no-version training ablation);
  2. evaluates the existing version-trained LoRA adapter and the new no-version
     LoRA adapter under both original prompt and version prompt test conditions.
     The evaluation grid uses --prompt-field probing_input and
     --prompt-field version_prompt.

Usage:
  scripts/run_lora_prompt_ablation_nohup.sh --gpu <card_id> [options]

Options:
  --gpu <card_id>        GPU card id passed to CUDA_VISIBLE_DEVICES.
  --models <list>        Comma-separated model keys. Default: v2_all.
  --mode <name>          all|train|eval|summarize. Default: all.
  --tag <name>           Output run tag. Default: lora_prompt_ablation_<timestamp>.
  --max-samples <n>      Evaluation cap; 0 means all test samples. Default: 0.
  --foreground           Run in current shell instead of nohup.
  -h, --help             Show this help.

Environment overrides:
  CONDA_ENV              Conda env name. Default: lkl_llm.
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
  MAX_TRAIN_SAMPLES      Training cap for smoke/debug. Default: 0.
  MAX_VAL_SAMPLES        Validation cap for smoke/debug. Default: 0.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU=""
MODELS="v2_all"
MODE="all"
TAG="lora_prompt_ablation_$(date +%Y%m%d_%H%M%S)"
MAX_SAMPLES="0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${GPU}" ]]; then
  echo "Missing required --gpu <card_id>" >&2
  usage >&2
  exit 2
fi

case "${MODE}" in all|train|eval|summarize) ;; *) echo "Invalid --mode ${MODE}" >&2; exit 2 ;; esac

if [[ "${MODELS}" == "v2_all" ]]; then
  MODELS="starcoder2_3b,starcoder2_7b,starcoder2_15b,deepseek_coder_6_7b_instruct,qwen2_5_coder_3b_instruct,qwen2_5_coder_7b_instruct,qwen2_5_coder_14b_instruct"
fi

RUN_ROOT="${PROJECT_ROOT}/output/${TAG}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${FOREGROUND}" != "1" && "${RUN_UNDER_NOHUP:-0}" != "1" ]]; then
  MAIN_LOG="${LOG_DIR}/nohup_main_gpu${GPU}.log"
  nohup setsid env \
    RUN_UNDER_NOHUP=1 \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    bash "$0" \
      --gpu "${GPU}" \
      --models "${MODELS}" \
      --mode "${MODE}" \
      --tag "${TAG}" \
      --max-samples "${MAX_SAMPLES}" \
      --foreground \
    > "${MAIN_LOG}" 2>&1 &
  echo "Started LoRA prompt ablation job."
  echo "PID: $!"
  echo "GPU: ${GPU}"
  echo "Run root: ${RUN_ROOT}"
  echo "Main log: ${MAIN_LOG}"
  exit 0
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-lkl_llm}"
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
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"

TRAIN_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_val.jsonl"
TEST_FILE="${PROJECT_ROOT}/data/mixed_sft_v1/mixed_sft_test.jsonl"

model_path_for() {
  case "$1" in
    starcoder2_3b) echo "${MODEL_PATH_STARCODER2_3B:-${MODEL_ROOT}/StarCoder/starcoder2-3b}" ;;
    starcoder2_7b) echo "${MODEL_PATH_STARCODER2_7B:-${MODEL_ROOT}/StarCoder/starcoder2-7b}" ;;
    starcoder2_15b) echo "${MODEL_PATH_STARCODER2_15B:-${MODEL_ROOT}/StarCoder/starcoder2-15b}" ;;
    deepseek_coder_6_7b_instruct) echo "${MODEL_PATH_DEEPSEEK_CODER_6_7B_INSTRUCT:-${MODEL_ROOT}/deepseek-ai/deepseek-coder-6.7b-instruct}" ;;
    qwen2_5_coder_3b_instruct) echo "${MODEL_PATH_QWEN2_5_CODER_3B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-3B-Instruct}" ;;
    qwen2_5_coder_7b_instruct) echo "${MODEL_PATH_QWEN2_5_CODER_7B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-7B-Instruct}" ;;
    qwen2_5_coder_14b_instruct) echo "${MODEL_PATH_QWEN2_5_CODER_14B_INSTRUCT:-${MODEL_ROOT}/Qwen/Qwen2.5-Coder-14B-Instruct}" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

max_length_for() {
  case "$1" in
    starcoder2_15b|qwen2_5_coder_14b_instruct) echo "256" ;;
    *) echo "384" ;;
  esac
}

version_adapter_for() {
  case "$1" in
    starcoder2_3b) echo "${PROJECT_ROOT}/output/mixed_sft_v1_starcoder2_3b_card1_e3" ;;
    starcoder2_7b) echo "${PROJECT_ROOT}/output/mixed_sft_v1_starcoder2_7b_card0_e3" ;;
    starcoder2_15b) echo "${PROJECT_ROOT}/output/mixed_sft_v1_starcoder2_15b_card1_e3" ;;
    deepseek_coder_6_7b_instruct) echo "${PROJECT_ROOT}/output/deepseek_coder_card1_card4/deepseek_coder_6_7b_instruct_lora" ;;
    qwen2_5_coder_3b_instruct) echo "${PROJECT_ROOT}/output/qwen2_5_coder_3b_card1/qwen2_5_coder_3b_instruct_lora" ;;
    qwen2_5_coder_7b_instruct) echo "${PROJECT_ROOT}/output/qwen2_5_coder_7b_card4/qwen2_5_coder_7b_instruct_lora" ;;
    qwen2_5_coder_14b_instruct) echo "${PROJECT_ROOT}/output/qwen2_5_coder_14b_card6/qwen2_5_coder_14b_instruct_lora" ;;
    *) echo "Unknown model key: $1" >&2; return 1 ;;
  esac
}

prompt_field_for_dir() {
  case "$1" in
    original_prompt) echo "probing_input" ;;
    version_prompt) echo "version_prompt" ;;
    *) echo "Unknown prompt dir: $1" >&2; return 1 ;;
  esac
}

run_eval() {
  local model_key="$1"
  local model_path="$2"
  local adapter_dir="$3"
  local prompt_dir="$4"
  local compare_name="$5"
  local max_length="$6"
  local prompt_field
  prompt_field="$(prompt_field_for_dir "${prompt_dir}")"

  local out_dir="${RUN_ROOT}/${model_key}/${prompt_dir}/${compare_name}"
  local log_file="${LOG_DIR}/${model_key}_${prompt_dir}_${compare_name}.log"
  if [[ -f "${out_dir}/comparison_summary.json" ]]; then
    echo "Skip existing eval: ${out_dir}/comparison_summary.json"
    return
  fi
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" scripts/eval_compare_lora.py \
    --model-name-or-path "${model_path}" \
    --adapter-dir "${adapter_dir}" \
    --test-file "${TEST_FILE}" \
    --output-dir "${out_dir}" \
    --prompt-field "${prompt_field}" \
    --max-length "${max_length}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-samples "${MAX_SAMPLES}" \
    > "${log_file}" 2>&1
}

run_model() {
  local model_key="$1"
  local model_path
  local max_length
  local version_adapter
  local no_version_adapter
  model_path="$(model_path_for "${model_key}")"
  max_length="$(max_length_for "${model_key}")"
  version_adapter="$(version_adapter_for "${model_key}")"
  no_version_adapter="${RUN_ROOT}/${model_key}/lora_no_version"

  echo "============================================================"
  echo "Model: ${model_key}"
  echo "GPU: ${GPU}"
  echo "Model path: ${model_path}"
  echo "Max length: ${max_length}"
  echo "Version-trained adapter: ${version_adapter}"
  echo "No-version adapter: ${no_version_adapter}"

  if [[ ! -d "${model_path}" ]]; then
    echo "Model directory not found: ${model_path}" >&2
    exit 1
  fi
  if [[ ! -f "${version_adapter}/adapter_config.json" ]]; then
    echo "Existing version-trained adapter not found: ${version_adapter}" >&2
    exit 1
  fi

  if [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; then
    if [[ -f "${no_version_adapter}/adapter_config.json" ]]; then
      echo "Skip existing no-version adapter: ${no_version_adapter}"
    else
      "${PYTHON_BIN}" scripts/train_lora.py \
        --train-file "${TRAIN_FILE}" \
        --val-file "${VAL_FILE}" \
        --model-name-or-path "${model_path}" \
        --output-dir "${no_version_adapter}" \
        --prompt-field probing_input \
        --max-length "${max_length}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-train-epochs "${EPOCHS}" \
        --per-device-train-batch-size "${TRAIN_BATCH_SIZE}" \
        --per-device-eval-batch-size "${EVAL_BATCH_SIZE}" \
        --max-train-samples "${MAX_TRAIN_SAMPLES}" \
        --max-val-samples "${MAX_VAL_SAMPLES}" \
        --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
        --logging-steps "${LOGGING_STEPS}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --lora-dropout "${LORA_DROPOUT}" \
        > "${LOG_DIR}/${model_key}_train_no_version.log" 2>&1
    fi
  fi

  if [[ "${MODE}" == "all" || "${MODE}" == "eval" ]]; then
    if [[ ! -f "${no_version_adapter}/adapter_config.json" ]]; then
      echo "No-version adapter missing for eval: ${no_version_adapter}" >&2
      exit 1
    fi
    for prompt_dir in original_prompt version_prompt; do
      run_eval "${model_key}" "${model_path}" "${version_adapter}" "${prompt_dir}" "lora_version_compare" "${max_length}"
      run_eval "${model_key}" "${model_path}" "${no_version_adapter}" "${prompt_dir}" "lora_no_version_compare" "${max_length}"
    done
  fi
}

IFS=',' read -r -a REQUESTED_MODELS <<< "${MODELS}"

if [[ "${MODE}" != "summarize" ]]; then
  echo "LoRA prompt ablation started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Run root: ${RUN_ROOT}"
  echo "Models: ${MODELS}"
  for raw_key in "${REQUESTED_MODELS[@]}"; do
    model_key="$(echo "${raw_key}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    run_model "${model_key}"
  done
fi

"${PYTHON_BIN}" scripts/summarize_lora_prompt_ablation.py \
  --run-root "${RUN_ROOT}" \
  --models "${MODELS}" \
  --output-md "${RUN_ROOT}/lora_prompt_ablation_summary.md" \
  --output-json "${RUN_ROOT}/lora_prompt_ablation_summary.json" \
  > "${RUN_ROOT}/lora_prompt_ablation_summary.stdout"

echo "LoRA prompt ablation finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary: ${RUN_ROOT}/lora_prompt_ablation_summary.md"
