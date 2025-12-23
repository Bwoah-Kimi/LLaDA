# LLaDA Custom Evaluation & Profiling with dLLM-Cache

This repository contains custom scripts and tools for evaluating and profiling the [LLaDA](https://github.com/ML-GSAI/LLaDA) model, incorporating acceleration features from [dLLM-Cache](https://github.com/Gen-Verse/dLLM-Cache).

## Overview

This project builds upon the original LLaDA repository and dLLM-Cache to provide:
- Automated downloading of datasets and models.
- Inference profiling tools.
- A flexible evaluation pipeline supporting multiple benchmarks (HumanEval, GSM8K, MMLU).
- Scripts for batch processing and result analysis.

## Prerequisites

Ensure you have the necessary dependencies installed.

```bash
pip install -r requirements.txt
pip install -r dLLM-cache/requirements.txt
```

## Setup

### 1. Download Models and Datasets

Before running evaluations, you need to download the LLaDA model and the required datasets.

**Download Model:**
This script downloads `GSAI-ML/LLaDA-8B-Instruct` to the `hf_models/` directory.
```bash
python my_scripts/download_model.py
```

**Download Datasets:**
This script downloads supported benchmarks (HumanEval, GSM8K, MMLU) to `hf_models/datasets`.
```bash
python my_scripts/download_dataset.py
```

## Usage

### Inference Profiling

Use the provided Jupyter Notebook to profile model inference performance.
- Open `my_scripts/inference_profiling.ipynb` in VS Code or Jupyter Lab.
- Run the cells to analyze memory usage, latency, and other metrics.

### Running Evaluations

The main evaluation pipeline is managed by `my_scripts/run_custom_eval.sh`. This script supports parameter sweeps (generation length, steps, block sizes) and handles sharding for parallel processing.

1.  **Configure the Script**: Edit `my_scripts/run_custom_eval.sh` to select your benchmark and sweep parameters.
    ```bash
    # Select Benchmark (uncomment one)
    BENCHMARK="humaneval"
    # BENCHMARK="gsm8k"
    
    # Sweep Parameters
    GEN_LENGTH_LIST=(512 1024)
    STEPS_LIST=(512)
    BLOCK_SIZES=(32 64)
    CACHE_ENABLED=false
    ```

2.  **Run the Evaluation**:
    ```bash
    bash my_scripts/run_custom_eval.sh
    ```
    This will:
    - Run inference using `my_scripts/run_custom_eval.py`.
    - Save results to `eval_results_custom/<RUN_ID>/`.
    - Merge sharded results into `merged_results.jsonl`.
    - Automatically calculate scores using the appropriate evaluation script (e.g., `evaluate_humaneval.py`).

### Re-evaluating Existing Results

If you have already generated results and want to re-calculate scores (e.g., after fixing a metric script), use `run_eval_only.sh`.

```bash
bash my_scripts/run_eval_only.sh
```
This script scans `eval_results_custom` for `merged_results.jsonl` files and runs the corresponding evaluation logic.

## Results

Evaluation results are stored in the `eval_results_custom` directory, organized by run configuration:

```
eval_results_custom/
└── humaneval_length512_steps512_block32_cachefalse/
    ├── shard_0.log
    ├── shard_0.jsonl
    ├── merged_results.jsonl
    └── evaluation.log
```

## TODO

- [ ] Add quantization evaluation.
