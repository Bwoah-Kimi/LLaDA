#!/bin/bash

# Configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
CACHE_DIR="/root/LLaDA/hf_models/hub"
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
SPLIT="test"
MC_NUM=64  # Higher means more accurate but slower
MAX_SAMPLES=100 # Limit samples for faster testing

OUTPUT_DIR="eval_results_custom/ppl"

echo "Starting PPL evaluation on $DATASET_NAME..."

python my_scripts/eval_ppl.py \
    --model_path "$MODEL_PATH" \
    --cache_dir "$CACHE_DIR" \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --mc_num "$MC_NUM" \
    --max_samples "$MAX_SAMPLES"

echo "Evaluation complete. Results in $OUTPUT_DIR"
