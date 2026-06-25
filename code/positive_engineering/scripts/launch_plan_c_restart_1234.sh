#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Launch Plan C steering experiments from a fresh tag, using four GPUs.

Default schedule:
  Wave 1:
    GPU[0] -> starcoder2_3b
    GPU[1] -> starcoder2_7b
    GPU[2] -> starcoder2_15b
    GPU[3] -> deepseek_coder_6_7b_instruct
  Wave 2, after StarCoder2 family finishes:
    GPU[0] -> qwen2_5_coder_3b_instruct
    GPU[1] -> qwen2_5_coder_7b_instruct
    GPU[2] -> qwen2_5_coder_14b_instruct

Usage:
  bash scripts/launch_plan_c_restart_1234.sh [options]

Options:
  --gpus <csv>          Four GPU ids. Default: 1,2,3,4.
  --tag-prefix <name>   Output tag prefix. Default: plan_c_restart_<timestamp>.
  --layers <spec>       Steering layers. Default: 2:22.
  --coefficient <num>   Steering coefficient. Default: 1.0.
  --foreground          Run orchestrator in current shell instead of nohup.
  -h, --help            Show this help.

Environment:
  CONDA_ENV             Conda env name. Default: lkl_llm.

Outputs:
  output/<tag-prefix>_<model>_g<gpu>/
  output/<tag-prefix>_orchestrator.log
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_CSV="1,2,3,4"
TAG_PREFIX="plan_c_restart_$(date +%Y%m%d_%H%M%S)"
LAYERS="2:22"
COEFFICIENT="1.0"
FOREGROUND="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_CSV="$2"; shift 2 ;;
    --tag-prefix) TAG_PREFIX="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --coefficient) COEFFICIENT="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "--gpus must contain exactly four comma-separated GPU ids, got: ${GPU_CSV}" >&2
  exit 2
fi

ORCH_LOG="${PROJECT_ROOT}/output/${TAG_PREFIX}_orchestrator.log"
mkdir -p "${PROJECT_ROOT}/output"

if [[ "${FOREGROUND}" != "1" && "${PLAN_C_RESTART_UNDER_NOHUP:-0}" != "1" ]]; then
  nohup setsid bash "$0" \
    --gpus "${GPU_CSV}" \
    --tag-prefix "${TAG_PREFIX}" \
    --layers "${LAYERS}" \
    --coefficient "${COEFFICIENT}" \
    --foreground \
    > "${ORCH_LOG}" 2>&1 < /dev/null &
  echo "Started Plan C restart orchestrator."
  echo "PID: $!"
  echo "GPUs: ${GPU_CSV}"
  echo "Tag prefix: ${TAG_PREFIX}"
  echo "Orchestrator log: ${ORCH_LOG}"
  exit 0
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # Makes `conda activate` work even when this script is launched by nohup.
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate "${CONDA_ENV:-lkl_llm}"

echo "[orchestrator] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[orchestrator] project=${PROJECT_ROOT}"
echo "[orchestrator] gpus=${GPU_CSV}"
echo "[orchestrator] tag_prefix=${TAG_PREFIX}"
echo "[orchestrator] layers=${LAYERS}"
echo "[orchestrator] coefficient=${COEFFICIENT}"

PIDS=()

start_model() {
  local gpu="$1"
  local model="$2"
  local tag="$3"
  local run_root="${PROJECT_ROOT}/output/${tag}"
  mkdir -p "${run_root}/logs"

  echo "[orchestrator] launch gpu=${gpu} model=${model} tag=${tag}"
  scripts/run_plan_c_nohup.sh \
    --gpu "${gpu}" \
    --models "${model}" \
    --phase all \
    --tag "${tag}" \
    --layers "${LAYERS}" \
    --coefficient "${COEFFICIENT}" \
    --overwrite-vectors \
    --foreground \
    > "${run_root}/logs/nohup_main.log" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  echo "[orchestrator] pid=${pid} gpu=${gpu} model=${model}"
}

wait_group() {
  local status=0
  local pid
  for pid in "$@"; do
    if wait "${pid}"; then
      echo "[orchestrator] pid=${pid} finished"
    else
      local code=$?
      echo "[orchestrator] pid=${pid} failed code=${code}"
      status=1
    fi
  done
  return "${status}"
}

echo "[orchestrator] wave 1: StarCoder2 family plus DeepSeek"
start_model "${GPUS[0]}" starcoder2_3b "${TAG_PREFIX}_starcoder2_3b_g${GPUS[0]}"
start_model "${GPUS[1]}" starcoder2_7b "${TAG_PREFIX}_starcoder2_7b_g${GPUS[1]}"
start_model "${GPUS[2]}" starcoder2_15b "${TAG_PREFIX}_starcoder2_15b_g${GPUS[2]}"
STAR_PIDS=("${PIDS[@]}")
PIDS=()

start_model "${GPUS[3]}" deepseek_coder_6_7b_instruct "${TAG_PREFIX}_deepseek_6_7b_g${GPUS[3]}"
DEEP_PIDS=("${PIDS[@]}")
PIDS=()

if wait_group "${STAR_PIDS[@]}"; then
  echo "[orchestrator] StarCoder2 family finished"
else
  echo "[orchestrator] StarCoder2 family had at least one failure; continuing to Qwen family"
fi

echo "[orchestrator] wave 2: Qwen2.5-Coder family"
start_model "${GPUS[0]}" qwen2_5_coder_3b_instruct "${TAG_PREFIX}_qwen2_5_coder_3b_g${GPUS[0]}"
start_model "${GPUS[1]}" qwen2_5_coder_7b_instruct "${TAG_PREFIX}_qwen2_5_coder_7b_g${GPUS[1]}"
start_model "${GPUS[2]}" qwen2_5_coder_14b_instruct "${TAG_PREFIX}_qwen2_5_coder_14b_g${GPUS[2]}"
QWEN_PIDS=("${PIDS[@]}")
PIDS=()

wait_group "${QWEN_PIDS[@]}" || true
wait_group "${DEEP_PIDS[@]}" || true

echo "[orchestrator] all scheduled Plan C jobs finished $(date '+%Y-%m-%d %H:%M:%S')"
