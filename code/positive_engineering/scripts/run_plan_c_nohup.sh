#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run Plan C steering-vector experiments on mixed_sft_v1.

This launcher supports:
  vectors         compute per-library steering vectors
  direct_eval     evaluate base model + direct steering
  kts_train       train KL-then-steer LoRA adapter
  kts_eval        evaluate KTS adapter + steering
  all             vectors -> direct_eval -> kts_train -> kts_eval

Usage:
  scripts/run_plan_c_nohup.sh --gpu <card_id> [options]

Options:
  --gpu <card_id>        GPU card id passed to CUDA_VISIBLE_DEVICES.
  --models <list>        Comma-separated model keys. Default: v2_all.
  --phase <name>         all|vectors|direct_eval|kts_train|kts_eval. Default: all.
  --tag <name>           Output run tag. Default: plan_c_<timestamp>.
  --layers <spec>        Steering layers, inclusive. Default: 2:22.
  --coefficient <float>  Steering coefficient. Default: 1.0.
  --max-samples <n>      Eval sample cap for smoke/debug. 0 means all test samples.
  --max-pairs <n>        Vector pair cap per library for smoke/debug. 0 means all pairs.
  --foreground           Run in current shell instead of nohup.
  --overwrite-vectors    Recompute vector files even if present.
  -h, --help             Show this help.

Model keys:
  starcoder2_3b, starcoder2_7b, starcoder2_15b,
  deepseek_coder_6_7b_instruct,
  qwen2_5_coder_3b_instruct, qwen2_5_coder_7b_instruct, qwen2_5_coder_14b_instruct

Environment overrides:
  PYTHON_BIN             Python executable. Default: python3.
  PRECISION              fp32|fp16|bf16. Default: bf16.
  EPOCHS                 KTS epochs. Default: 3.
  LEARNING_RATE          KTS LoRA learning rate. Default: 1e-4.
  GRAD_ACCUM_STEPS       KTS gradient accumulation. Default: 16.
  LORA_R                 KTS LoRA rank. Default: 16.
  LORA_ALPHA             KTS LoRA alpha. Default: 32.
  LORA_DROPOUT           KTS LoRA dropout. Default: 0.05.
  MAX_NEW_TOKENS         mixed_sft_v1 generation length. Default: 64.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU=""
MODELS="v2_all"
PHASE="all"
TAG="plan_c_$(date +%Y%m%d_%H%M%S)"
LAYERS="2:22"
COEFFICIENT="1.0"
MAX_SAMPLES="0"
MAX_PAIRS="0"
FOREGROUND="0"
OVERWRITE_VECTORS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --coefficient) COEFFICIENT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max-pairs) MAX_PAIRS="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    --overwrite-vectors) OVERWRITE_VECTORS="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${GPU}" ]]; then
  echo "Missing required --gpu <card_id>" >&2
  usage >&2
  exit 2
fi

case "${PHASE}" in
  all|vectors|direct_eval|kts_train|kts_eval) ;;
  *) echo "Invalid --phase ${PHASE}" >&2; exit 2 ;;
esac

if [[ "${MODELS}" == "v2_all" ]]; then
  MODELS="starcoder2_3b,starcoder2_7b,starcoder2_15b,deepseek_coder_6_7b_instruct,qwen2_5_coder_3b_instruct,qwen2_5_coder_7b_instruct,qwen2_5_coder_14b_instruct"
fi

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
      --models "${MODELS}" \
      --phase "${PHASE}" \
      --tag "${TAG}" \
      --layers "${LAYERS}" \
      --coefficient "${COEFFICIENT}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-pairs "${MAX_PAIRS}" \
      --foreground \
    > "${MAIN_LOG}" 2>&1 &
  echo "Started Plan C job."
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
PRECISION="${PRECISION:-bf16}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

run_vectors() {
  local model_key="$1"
  local vector_file="${RUN_ROOT}/${model_key}/vectors.pt"
  local log_file="${LOG_DIR}/${model_key}_vectors.log"
  mkdir -p "$(dirname "${vector_file}")"
  if [[ -f "${vector_file}" && "${OVERWRITE_VECTORS}" != "1" ]]; then
    echo "Skip existing vectors: ${vector_file}"
    return
  fi
  "${PYTHON_BIN}" scripts/compute_steering_vectors.py \
    --model-key "${model_key}" \
    --output-file "${vector_file}" \
    --layers "${LAYERS}" \
    --max-pairs-per-library "${MAX_PAIRS}" \
    --precision "${PRECISION}" \
    > "${log_file}" 2>&1
}

run_direct_eval() {
  local model_key="$1"
  local vector_file="${RUN_ROOT}/${model_key}/vectors.pt"
  local out_dir="${RUN_ROOT}/${model_key}/direct_steering_compare"
  local log_file="${LOG_DIR}/${model_key}_direct_eval.log"
  if [[ ! -f "${vector_file}" ]]; then
    echo "Missing vector file for ${model_key}: ${vector_file}" >&2
    exit 1
  fi
  if [[ -f "${out_dir}/comparison_summary.json" ]]; then
    echo "Skip existing direct eval: ${out_dir}/comparison_summary.json"
    return
  fi
  "${PYTHON_BIN}" scripts/eval_compare_steering.py \
    --model-key "${model_key}" \
    --vector-file "${vector_file}" \
    --output-dir "${out_dir}" \
    --layers "${LAYERS}" \
    --coefficient "${COEFFICIENT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-samples "${MAX_SAMPLES}" \
    --precision "${PRECISION}" \
    --allow-missing-vector \
    > "${log_file}" 2>&1
}

run_kts_train() {
  local model_key="$1"
  local vector_file="${RUN_ROOT}/${model_key}/vectors.pt"
  local adapter_dir="${RUN_ROOT}/${model_key}/kts_lora"
  local log_file="${LOG_DIR}/${model_key}_kts_train.log"
  if [[ ! -f "${vector_file}" ]]; then
    echo "Missing vector file for ${model_key}: ${vector_file}" >&2
    exit 1
  fi
  if [[ -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "Resume/update KTS adapter: ${adapter_dir}"
  fi
  "${PYTHON_BIN}" scripts/train_kts_lora.py \
    --model-key "${model_key}" \
    --vector-file "${vector_file}" \
    --output-dir "${adapter_dir}" \
    --layers "${LAYERS}" \
    --coefficient "${COEFFICIENT}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-train-epochs "${EPOCHS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --precision "${PRECISION}" \
    --resume \
    > "${log_file}" 2>&1
}

run_kts_eval() {
  local model_key="$1"
  local vector_file="${RUN_ROOT}/${model_key}/vectors.pt"
  local adapter_dir="${RUN_ROOT}/${model_key}/kts_lora"
  local out_dir="${RUN_ROOT}/${model_key}/kts_steering_compare"
  local log_file="${LOG_DIR}/${model_key}_kts_eval.log"
  if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "Missing KTS adapter for ${model_key}: ${adapter_dir}" >&2
    exit 1
  fi
  if [[ -f "${out_dir}/comparison_summary.json" ]]; then
    echo "Skip existing KTS eval: ${out_dir}/comparison_summary.json"
    return
  fi
  "${PYTHON_BIN}" scripts/eval_compare_steering.py \
    --model-key "${model_key}" \
    --adapter-dir "${adapter_dir}" \
    --vector-file "${vector_file}" \
    --output-dir "${out_dir}" \
    --layers "${LAYERS}" \
    --coefficient "${COEFFICIENT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-samples "${MAX_SAMPLES}" \
    --precision "${PRECISION}" \
    --allow-missing-vector \
    > "${log_file}" 2>&1
}

IFS=',' read -r -a REQUESTED_MODELS <<< "${MODELS}"
echo "Plan C run root: ${RUN_ROOT}"
echo "GPU: ${GPU}"
echo "Phase: ${PHASE}"
echo "Models: ${MODELS}"

for raw_key in "${REQUESTED_MODELS[@]}"; do
  model_key="$(echo "${raw_key}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  echo "===== ${model_key} ====="
  case "${PHASE}" in
    vectors) run_vectors "${model_key}" ;;
    direct_eval) run_direct_eval "${model_key}" ;;
    kts_train) run_kts_train "${model_key}" ;;
    kts_eval) run_kts_eval "${model_key}" ;;
    all)
      run_vectors "${model_key}"
      run_direct_eval "${model_key}"
      run_kts_train "${model_key}"
      run_kts_eval "${model_key}"
      ;;
  esac
done

echo "Plan C run finished: $(date '+%Y-%m-%d %H:%M:%S')"
