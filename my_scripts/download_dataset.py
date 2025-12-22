import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/LLaDA/hf_models/'
from datasets import load_dataset

BENCHMARK_NAME = "mmlu"

if BENCHMARK_NAME == "humaneval":
    print("Loading HumanEval dataset...")
    ds = load_dataset("openai_humaneval", split="test")
elif BENCHMARK_NAME == "gsm8k":
    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main", split="test")
elif BENCHMARK_NAME == 'mmlu':
    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")
else:
    raise ValueError(f"Unsupported benchmark: {BENCHMARK_NAME}")
print(f"Loaded {len(ds)} problems.")

# Display a sample
print("\nSample Problem:")
if BENCHMARK_NAME == "humaneval":
    print(ds[0]['prompt'])
elif BENCHMARK_NAME == "gsm8k":
    print(ds[0]['question'])
elif BENCHMARK_NAME == "mmlu":
    print(f"Question: {ds[0]['question']}\nChoices: {ds[0]['choices']}\nAnswer: {ds[0]['answer']}")
