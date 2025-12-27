#!/bin/bash

# Configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"

# Get script directory (where the script is located)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Select Benchmark (uncomment one)
BENCHMARK="humaneval"
# BENCHMARK="gsm8k"

if [ "$BENCHMARK" == "humaneval" ]; then
    DATASET_NAME="humaneval"
    DATASET_PATH="/root/LLaDA/hf_models/datasets/openai_humaneval"
    EVAL_SCRIPT="$SCRIPT_DIR/evaluate_humaneval.py"
elif [ "$BENCHMARK" == "gsm8k" ]; then
    DATASET_NAME="gsm8k"
    DATASET_PATH="/root/LLaDA/hf_models/datasets/gsm8k"
    EVAL_SCRIPT="$SCRIPT_DIR/evaluate_gsm8k.py"
elif [ "$BENCHMARK" == "mmlu" ]; then
    DATASET_NAME="mmlu"
    DATASET_PATH="/root/LLaDA/hf_models/datasets/mmlu"
    EVAL_SCRIPT="$SCRIPT_DIR/evaluate_mmlu.py"
else
    echo "Unknown benchmark: $BENCHMARK"
    exit 1
fi

OUTPUT_BASE_DIR="eval_results_custom"

# Parallelism
NUM_SHARDS=1  # Set to 1 to avoid OOM if using single GPU
BATCH_SIZE=1  # Increase batch size to accelerate

# Define Runs: "GEN_LENGTH STEPS BLOCK_SIZE CACHE_ENABLED TRANSFER_RATIO SIM_THRESHOLD CACHE_STRATEGY"

RUNS=(
    "512 128 32 false 0.0 0.0 ratio"
    "512 128 32 true 0.25 0.0 ratio"
    "512 128 32 true 0.0 0.998 threshold"
    "512 128 32 true 0.0 0.999 threshold"
    "512 256 32 false 0.0 0.0 ratio"
    "512 256 32 true 0.25 0.0 ratio"
    "512 256 32 true 0.0 0.998 threshold"
    "512 256 32 true 0.0 0.999 threshold"
    "512 512 32 true 0.25 0.0 ratio"
    "512 512 32 true 0.0 0.998 threshold"
    "512 512 32 true 0.0 0.999 threshold"
    "1024 256 32 false 0.0 0.0 ratio"
    "1024 256 32 true 0.25 0.0 ratio"
    "1024 256 32 true 0.0 0.998 threshold"
    "1024 256 32 true 0.0 0.999 threshold"
    "1024 512 32 false 0.0 0.0 ratio"
    "1024 512 32 true 0.25 0.0 ratio"
    "1024 512 32 true 0.0 0.998 threshold"
    "1024 512 32 true 0.0 0.999 threshold"
    "1024 1024 32 false 0.0 0.0 ratio"
    "1024 1024 32 true 0.25 0.0 ratio"
    "1024 1024 32 true 0.0 0.998 threshold"
    "1024 1024 32 true 0.0 0.999 threshold"
)

for run_config in "${RUNS[@]}"; do
    # Parse configuration
    read -r gen_length steps block cache_enabled t_r s_t cache_strategy <<< "$run_config"

    if [ "$cache_enabled" = true ]; then
        if [ "$cache_strategy" = "ratio" ]; then
            RUN_ID="${DATASET_NAME}_length${gen_length}_steps${steps}_block${block}_cache${cache_enabled}_tr${t_r}"
        elif [ "$cache_strategy" = "threshold" ]; then
            RUN_ID="${DATASET_NAME}_length${gen_length}_steps${steps}_block${block}_cache${cache_enabled}_st${s_t}"
        else
            echo "Unknown CACHE_STRATEGY: $cache_strategy"
            exit 1
        fi
    else
        RUN_ID="${DATASET_NAME}_length${gen_length}_steps${steps}_block${block}_cache${cache_enabled}"
    fi
    
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
            $( [ "$cache_enabled" = true ] && echo "--enable_dllm_cache" ) \
            --cache_transfer_ratio "$t_r" \
            --cache_similarity_threshold "$s_t" \
            --cache_strategy "$cache_strategy" \
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
    
    echo "Evaluating results..."
    python "$EVAL_SCRIPT" \
        --results_file "${OUTPUT_DIR}/merged_results.jsonl" \
        --dataset_path "$DATASET_PATH" | tee -a "${OUTPUT_DIR}/evaluation.log"
    
done
