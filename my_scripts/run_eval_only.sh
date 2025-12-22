#!/bin/bash

# Configuration
DATASET_PATH=""
OUTPUT_BASE_DIR="eval_results_custom"

# Find all merged_results.jsonl files
find "$OUTPUT_BASE_DIR" -name "merged_results.jsonl" | while read result_file; do
    dir_path=$(dirname "$result_file")
    echo "========================================================"
    echo "Evaluating: $dir_path"
    echo "========================================================"
    
    # Infer benchmark from directory name
    if [[ "$dir_path" == *"humaneval"* ]]; then
        EVAL_SCRIPT="my_scripts/evaluate_humaneval.py"
        DATASET_PATH="/root/LLaDA/hf_models/datasets/openai_humaneval"
    elif [[ "$dir_path" == *"gsm8k"* ]]; then
        EVAL_SCRIPT="my_scripts/evaluate_gsm8k.py"
        DATASET_PATH="/root/LLaDA/hf_models/datasets/gsm8k"
    elif [[ "$dir_path" == *"mmlu"* ]]; then
        EVAL_SCRIPT="my_scripts/evaluate_mmlu.py"
        DATASET_PATH="/root/LLaDA/hf_models/datasets/mmlu"
    else
        echo "Unknown benchmark for $dir_path, stopping..."
        exit 1
    fi
    
    python "$EVAL_SCRIPT" \
        --results_file "$result_file" \
        --dataset_path "$DATASET_PATH" | tee "${dir_path}/evaluation.log"
        
    echo ""
done
