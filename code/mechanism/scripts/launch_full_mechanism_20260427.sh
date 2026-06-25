#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MECH_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
LOG_DIR="$MECH_ROOT/output/full_mechanism_20260427/logs"
mkdir -p "$LOG_DIR"

MODELS=(
  starcoder2_3b
  starcoder2_7b
  starcoder2_15b
  deepseek_coder_6_7b_instruct
  qwen2_5_coder_3b_instruct
  qwen2_5_coder_7b_instruct
  qwen2_5_coder_14b_instruct
)

GPUS=(0 1 2 3 4 5 6)

for idx in "${!MODELS[@]}"; do
  model=${MODELS[$idx]}
  gpu=${GPUS[$idx]}
  log_file="$LOG_DIR/${model}.log"
  echo "Launching $model on GPU $gpu -> $log_file"
  setsid bash "$SCRIPT_DIR/run_mechanism_model.sh" "$gpu" "$model" >"$log_file" 2>&1 < /dev/null &
  echo "$model pid=$!"
done

echo "Logs: $LOG_DIR"
