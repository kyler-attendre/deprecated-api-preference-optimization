#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run BigCode evaluation harness on HumanEval/MBPP pass@1 for baseline and LoRA.

This launcher intentionally delegates generation, post-processing, and execution
metrics to /workspace/bigcode-evaluation-harness. It only supplies project
model paths, LoRA adapter paths, GPU/nohup handling, and resumable output paths.

Usage:
  scripts/run_official_code_eval_nohup.sh --gpu <card_id> [options]

Options:
  --gpu <card_id>        GPU card id passed to CUDA_VISIBLE_DEVICES.
  --models <list>        Comma-separated model keys. Default: v2_all.
  --benchmark <name>     humaneval|mbpp|both. Default: both.
  --variant <name>       base|lora|both. Default: both.
  --tag <name>           Output run tag. Default: bigcode_code_eval_<timestamp>.
  --limit <n>            Smoke/debug task cap. 0 means full task set.
  --foreground           Run in current shell instead of nohup.
  -h, --help             Show this help.

Environment overrides:
  PYTHON_BIN             Python executable in the active conda env. Default: python3.
  BIGCODE_ROOT           BigCode harness checkout. Default: /workspace/bigcode-evaluation-harness.
  BIGCODE_LOCAL_HUMANEVAL_FILE
                         Local HumanEval JSONL.GZ fallback used by the patched BigCode task.
                         Default: /workspace/evaluation/human-eval/data/HumanEval.jsonl.gz.
  BIGCODE_LOCAL_MBPP_FILE
                         Local MBPP JSONL fallback used by the patched BigCode task.
                         Default: /workspace/evaluation/google-research/mbpp/mbpp.jsonl.
  PRECISION              fp32|fp16|bf16. Default: bf16.
  MAX_LENGTH_GENERATION  BigCode prompt+generation length. Default: 512.
  BATCH_SIZE             BigCode batch size. Default: 1.
  TRUST_REMOTE_CODE      Set to 1 to pass --trust_remote_code. Default: 0.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU=""
MODELS="v2_all"
BENCHMARK="both"
VARIANT="both"
TAG="bigcode_code_eval_$(date +%Y%m%d_%H%M%S)"
LIMIT="0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
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

case "${BENCHMARK}" in humaneval|mbpp|both) ;; *) echo "Invalid --benchmark ${BENCHMARK}" >&2; exit 2 ;; esac
case "${VARIANT}" in base|lora|both) ;; *) echo "Invalid --variant ${VARIANT}" >&2; exit 2 ;; esac

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
      --benchmark "${BENCHMARK}" \
      --variant "${VARIANT}" \
      --tag "${TAG}" \
      --limit "${LIMIT}" \
      --foreground \
    > "${MAIN_LOG}" 2>&1 &
  echo "Started BigCode code eval job."
  echo "PID: $!"
  echo "GPU: ${GPU}"
  echo "Run root: ${RUN_ROOT}"
  echo "Main log: ${MAIN_LOG}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
BIGCODE_ROOT="${BIGCODE_ROOT:-/workspace/bigcode-evaluation-harness}"
BIGCODE_LOCAL_HUMANEVAL_FILE="${BIGCODE_LOCAL_HUMANEVAL_FILE:-/workspace/evaluation/human-eval/data/HumanEval.jsonl.gz}"
BIGCODE_LOCAL_MBPP_FILE="${BIGCODE_LOCAL_MBPP_FILE:-/workspace/evaluation/google-research/mbpp/mbpp.jsonl}"
PRECISION="${PRECISION:-bf16}"
MAX_LENGTH_GENERATION="${MAX_LENGTH_GENERATION:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export BIGCODE_LOCAL_HUMANEVAL_FILE
export BIGCODE_LOCAL_MBPP_FILE

if [[ ! -f "${BIGCODE_ROOT}/main.py" ]]; then
  echo "Missing BigCode harness main.py under BIGCODE_ROOT=${BIGCODE_ROOT}" >&2
  exit 1
fi

model_for() {
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

adapter_for() {
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

bigcode_task_for() {
  case "$1" in
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *) echo "Unknown benchmark: $1" >&2; return 1 ;;
  esac
}

append_common_args() {
  local -n args_ref="$1"
  if [[ "${LIMIT}" != "0" ]]; then
    args_ref+=(--limit "${LIMIT}")
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    args_ref+=(--trust_remote_code)
  fi
}

run_bigcode() {
  local log_file="$1"
  shift
  (
    cd "${BIGCODE_ROOT}"
    "${PYTHON_BIN}" "${BIGCODE_ROOT}/main.py" "$@"
  ) > "${log_file}" 2>&1
}

run_one_eval() {
  local model_key="$1"
  local benchmark="$2"
  local variant="$3"
  local bigcode_task
  bigcode_task="$(bigcode_task_for "${benchmark}")"

  local model_path
  model_path="$(model_for "${model_key}")"
  if [[ ! -e "${model_path}" ]]; then
    echo "Missing model path for ${model_key}: ${model_path}" >&2
    exit 1
  fi

  local out_dir="${RUN_ROOT}/${model_key}/${variant}/${benchmark}"
  local metric_json="${out_dir}/metrics.json"
  local generation_base="${out_dir}/generations.json"
  local generation_json="${out_dir}/generations_${bigcode_task}.json"
  local log_file="${LOG_DIR}/${model_key}_${variant}_${benchmark}.log"
  mkdir -p "${out_dir}"

  if [[ -f "${metric_json}" ]]; then
    echo "Skip existing metric: ${metric_json}"
    return
  fi

  local args=(
    --model "${model_path}"
    --tasks "${bigcode_task}"
    --allow_code_execution
    --metric_output_path "${metric_json}"
    --n_samples 1
    --precision "${PRECISION}"
    --max_length_generation "${MAX_LENGTH_GENERATION}"
    --batch_size "${BATCH_SIZE}"
  )
  append_common_args args

  if [[ "${variant}" == "lora" ]]; then
    local adapter_dir
    adapter_dir="$(adapter_for "${model_key}")"
    if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
      echo "Missing LoRA adapter for ${model_key}: ${adapter_dir}" >&2
      exit 1
    fi
    args+=(--peft_model "${adapter_dir}")
  fi

  if [[ -f "${generation_json}" ]]; then
    echo "Resume evaluation from existing generations: ${generation_json}"
    args+=(--load_generations_path "${generation_json}")
  else
    args+=(
      --save_generations
      --save_generations_path "${generation_base}"
    )
  fi

  echo "BigCode ${model_key} ${variant} ${benchmark}"
  echo "  metric: ${metric_json}"
  echo "  log: ${log_file}"
  run_bigcode "${log_file}" "${args[@]}"
}

IFS=',' read -r -a REQUESTED_MODELS <<< "${MODELS}"
BENCHMARKS=()
VARIANTS=()
if [[ "${BENCHMARK}" == "both" ]]; then BENCHMARKS=(humaneval mbpp); else BENCHMARKS=("${BENCHMARK}"); fi
if [[ "${VARIANT}" == "both" ]]; then VARIANTS=(base lora); else VARIANTS=("${VARIANT}"); fi

echo "BigCode code eval root: ${RUN_ROOT}"
echo "Models: ${MODELS}"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "GPU: ${GPU}"
echo "BigCode root: ${BIGCODE_ROOT}"

for raw_key in "${REQUESTED_MODELS[@]}"; do
  model_key="$(echo "${raw_key}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  for variant in "${VARIANTS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
      run_one_eval "${model_key}" "${benchmark}" "${variant}"
    done
  done
done

echo "BigCode code eval finished: $(date '+%Y-%m-%d %H:%M:%S')"
