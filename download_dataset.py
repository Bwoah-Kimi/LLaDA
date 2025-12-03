import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/LLaDA/hf_models/'
from datasets import load_dataset

# Load HumanEval Dataset
print("Loading HumanEval dataset...")
ds = load_dataset("openai_humaneval", split="test")
print(f"Loaded {len(ds)} problems.")

# Display a sample
print("\nSample Problem:")
print(ds[0]['prompt'])