#!/bin/bash

# Configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
DATASET_NAME="humaneval"
DATASET_PATH="/root/LLaDA/hf_models/datasets/openai_humaneval"
OUTPUT_BASE_DIR="eval_results_custom"

# Parallelism
NUM_SHARDS=1  # Set to 1 to avoid OOM if using single GPU
BATCH_SIZE=8  # Increase batch size to accelerate

# Sweep Parameters
GEN_LENGTH_LIST=(512 1024)
STEPS_LIST=(512)
BLOCK_SIZES=(32 64)
CACHE_ENABLED=false

for gen_length in "${GEN_LENGTH_LIST[@]}"; do
    for steps in "${STEPS_LIST[@]}"; do
        for block in "${BLOCK_SIZES[@]}"; do
            
            RUN_ID="${DATASET_NAME}_length${gen_length}_steps${steps}_block${block}_cache${CACHE_ENABLED}"
            OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_ID}"
            mkdir -p "$OUTPUT_DIR"
            
            echo "Starting run: $RUN_ID with $NUM_SHARDS shards..."
            
            pids=()
            for ((i=0; i<NUM_SHARDS; i++)); do
                python my_scripts/run_custom_eval.py \
                    --model_path "$MODEL_PATH" \
                    --dataset_name "$DATASET_NAME" \
                    --dataset_path "$DATASET_PATH" \
                    --output_dir "$OUTPUT_DIR" \
                    --shard_id "$i" \
                    --num_shards "$NUM_SHARDS" \
                    --batch_size "$BATCH_SIZE" \
                    --gen_length "$gen_length" \
                    --steps "$steps" \
                    --block_length "$block" \
                    $( [ "$CACHE_ENABLED" = true ] && echo "--enable_dllm_cache" ) \
                    > "${OUTPUT_DIR}/shard_${i}.log" 2>&1 &
                
                pids+=($!)
            done
            
            # Wait for all shards
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
            
            echo "Run $RUN_ID completed. Merging results..."
            cat "${OUTPUT_DIR}"/*.jsonl > "${OUTPUT_DIR}/merged_results.jsonl"
            echo "Merged results saved to ${OUTPUT_DIR}/merged_results.jsonl"
            
        done
    done
done
