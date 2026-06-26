#!/usr/bin/env bash
# Smoke test for this replication package.
# Usage: bash reproduce.sh --quick --model-name-or-path <path-to-a-local-causal-LM>
#
# This does not require downloading the paper's base models — pass any local
# causal LM to exercise the eval code path end-to-end on 5 samples. It does
# NOT reproduce paper numbers (see README.md "Reproduction" for that); it
# only confirms the dataset, adapter files, and environment are wired up
# correctly. See README.md for the full per-RQ reproduction walkthrough.
set -euo pipefail

MODEL=""
for arg in "$@"; do
  case "$arg" in
    --model-name-or-path=*) MODEL="${arg#*=}" ;;
  esac
done
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--model-name-or-path" ]; then MODEL="$2"; fi
  shift
done

if [ -z "$MODEL" ]; then
  echo "Usage: bash reproduce.sh --quick --model-name-or-path <path-to-a-local-causal-LM>" >&2
  exit 1
fi

OUT="$(mktemp -d)"
echo "Running smoke test (5 samples) with model=$MODEL ..."
python code/positive_engineering/scripts/eval_compare_lora.py \
  --model-name-or-path "$MODEL" \
  --adapter-dir results/rq1_effectiveness/sftdpo_starcoder2-7b/starcoder2_7b \
  --test-file code/positive_engineering/data/mixed_sft_v1/mixed_sft_test.jsonl \
  --max-samples 5 --output-dir "$OUT"

echo
echo "Smoke test complete. Output: $OUT/comparison_summary.json"
cat "$OUT/comparison_summary.json"
