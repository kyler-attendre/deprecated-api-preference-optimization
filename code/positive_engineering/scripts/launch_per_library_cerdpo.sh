#!/usr/bin/env bash
# launch_per_library_cerdpo.sh
#
# 多卡并行启动 per-library CER-DPO 全量训练。
# 分卡规则：
#   GPU 1 → starcoder2_3b 跑完后接着跑 qwen2_5_coder_3b_instruct（小模型共卡）
#   GPU 2 → starcoder2_7b
#   GPU 3 → deepseek_coder_6_7b_instruct
#   GPU 4 → qwen2_5_coder_7b_instruct
#   GPU 5 → starcoder2_15b
#   GPU 6 → qwen2_5_coder_14b_instruct
#   GPU 7 → 空闲备用
#
# Usage:
#   bash launch_per_library_cerdpo.sh [--tag TAG] [--epochs N] [--dry-run]
set -euo pipefail

TAG="per_library_cerdpo_$(date +%Y%m%d_%H%M%S)"
EPOCHS="3"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)     TAG="$2";    shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1;  shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PROJECT_DIR/scripts/run_per_library_cerdpo.sh"
LOG_DIR="$PROJECT_DIR/output/$TAG/launcher_logs"
mkdir -p "$LOG_DIR"

source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lkl_llm}"

echo "============================================================"
echo "Per-library CER-DPO — parallel launch"
echo "  tag:    $TAG"
echo "  epochs: $EPOCHS"
echo "  log:    $LOG_DIR"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  | awk -F', ' '{printf "  GPU %s: %s/%s  util=%s\n", $1, $2, $3, $4}'
echo ""

launch_model() {
  local gpu="$1"
  local model_key="$2"
  local log="$LOG_DIR/${model_key}_gpu${gpu}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] GPU $gpu → $model_key  log=$log"
    return
  fi

  # setsid + nohup 双重保证：脱离父进程组，launcher 退出后继续运行
  setsid nohup bash "$SCRIPT" \
    --gpu "$gpu" \
    --model-key "$model_key" \
    --mode all \
    --tag "$TAG" \
    --epochs "$EPOCHS" \
    > "$log" 2>&1 &
  echo "  GPU $gpu → $model_key  PID=$!  log=$log"
}

# GPU 1：生成一个顺序跑两个小模型的包装脚本，再用 setsid nohup 启动
GPU1_WRAPPER="$LOG_DIR/gpu1_sequential.sh"
cat > "$GPU1_WRAPPER" <<WRAPPER
#!/usr/bin/env bash
source "${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lkl_llm}"
echo "[gpu1] start starcoder2_3b: \$(date)"
bash "$SCRIPT" --gpu 1 --model-key starcoder2_3b \\
  --mode all --tag "$TAG" --epochs "$EPOCHS" \\
  >> "$LOG_DIR/starcoder2_3b_gpu1.log" 2>&1
echo "[gpu1] starcoder2_3b done, starting qwen_3b: \$(date)"
bash "$SCRIPT" --gpu 1 --model-key qwen2_5_coder_3b_instruct \\
  --mode all --tag "$TAG" --epochs "$EPOCHS" \\
  >> "$LOG_DIR/qwen2_5_coder_3b_instruct_gpu1.log" 2>&1
echo "[gpu1] all done: \$(date)"
WRAPPER
chmod +x "$GPU1_WRAPPER"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] GPU 1 → starcoder2_3b then qwen2_5_coder_3b_instruct"
else
  setsid nohup bash "$GPU1_WRAPPER" > "$LOG_DIR/gpu1_wrapper.log" 2>&1 &
  echo "  GPU 1 → starcoder2_3b → qwen_3b  PID=$!  log=$LOG_DIR/gpu1_wrapper.log"
fi

# GPU 2-6：各自单独跑一个模型
launch_model 2 starcoder2_7b
launch_model 3 deepseek_coder_6_7b_instruct
launch_model 4 qwen2_5_coder_7b_instruct
launch_model 5 starcoder2_15b
launch_model 6 qwen2_5_coder_14b_instruct

echo ""
echo "============================================================"
echo "All jobs launched (running detached). Tag: $TAG"
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  watch -n5 nvidia-smi"
echo "============================================================"
