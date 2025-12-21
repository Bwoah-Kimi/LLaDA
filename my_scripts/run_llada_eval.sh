#!/usr/bin/env bash
set -euo pipefail

# Find the directory of this script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Sweep LLaDA evaluation settings using eval_llada.py (profiled_generate)
# Configure by editing the variables below.

TASKS=humaneval
STEPS=1024
BLOCK_SIZES="128"
GEN_LENGTHS="128 256"
BATCH_SIZE=1
OTHER_ARGS="--confirm_run_unsafe_code true"
RESULTS_DIR="$SCRIPT_DIR/../eval_results"
mkdir -p "$RESULTS_DIR"

timestamp=$(date +%Y%m%d-%H%M%S)

for block in $BLOCK_SIZES; do
  for gen in $GEN_LENGTHS; do
    run_id="tasks-${TASKS}_block-${block}_gen-${gen}_steps-${STEPS}_${timestamp}"
    echo "=== Running: $run_id ==="
    python eval_llada.py \
      --tasks "$TASKS" \
      --batch_size "$BATCH_SIZE" \
      --steps "$STEPS" \
      --gen_length "$gen" \
      --block_length "$block" \
      --enable_dllm_cache false \
      --other_args="$OTHER_ARGS" | tee "$RESULTS_DIR/$run_id.log"
  done
done

echo "\nDone. Logs in: $RESULTS_DIR"
