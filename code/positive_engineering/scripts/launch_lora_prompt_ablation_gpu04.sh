#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/version-control-study/05_positive_engineering"
cd "${PROJECT_ROOT}"

source /opt/conda/etc/profile.d/conda.sh
conda activate lkl_llm

mode="${1:-full}"

case "${mode}" in
  smoke)
    EPOCHS=1 \
    MAX_TRAIN_SAMPLES=4 \
    MAX_VAL_SAMPLES=2 \
    MAX_NEW_TOKENS=8 \
    TRAIN_BATCH_SIZE=1 \
    EVAL_BATCH_SIZE=1 \
    GRAD_ACCUM_STEPS=1 \
    scripts/run_lora_prompt_ablation_nohup.sh \
      --gpu 0 \
      --models starcoder2_3b \
      --tag lora_prompt_ablation_smoke_20260422 \
      --max-samples 2 \
      --foreground
    ;;
  full)
    scripts/run_lora_prompt_ablation_nohup.sh \
      --gpu 0 \
      --models starcoder2_15b,qwen2_5_coder_3b_instruct,qwen2_5_coder_7b_instruct \
      --tag lora_prompt_ablation_20260422_full

    scripts/run_lora_prompt_ablation_nohup.sh \
      --gpu 4 \
      --models qwen2_5_coder_14b_instruct,starcoder2_3b,starcoder2_7b,deepseek_coder_6_7b_instruct \
      --tag lora_prompt_ablation_20260422_full
    ;;
  summarize)
    python3 scripts/summarize_lora_prompt_ablation.py \
      --run-root output/lora_prompt_ablation_20260422_full \
      --output-md output/lora_prompt_ablation_20260422_full/lora_prompt_ablation_summary.md \
      --output-json output/lora_prompt_ablation_20260422_full/lora_prompt_ablation_summary.json
    ;;
  *)
    echo "Usage: $0 [smoke|full|summarize]" >&2
    exit 2
    ;;
esac
