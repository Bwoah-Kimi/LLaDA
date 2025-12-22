import argparse
import json
import re
import os
from datasets import load_dataset
from tqdm import tqdm

def load_dataset_local(path):
    print(f"Loading dataset from {path}...")
    # GSM8K usually has 'main' config
    try:
        return load_dataset(path=path, name="main", split="test")
    except:
        return load_dataset(path=path, split="test")

def extract_answer(text):
    # Extract the last number in the text
    # This is a simplified extractor. 
    # Often GSM8K solutions end with "#### <number>"
    # But model output might just be the reasoning + answer.
    # We look for the last number.
    
    # If the model follows the format "#### number", use that.
    if "####" in text:
        text = text.split("####")[-1]
    
    # Find all numbers (integers or floats)
    # Remove commas from numbers like 1,000
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        return numbers[-1]
    return None

def is_correct(completion, target):
    pred = extract_answer(completion)
    gold = extract_answer(target)
    
    if pred is None or gold is None:
        return False
    
    try:
        return float(pred) == float(gold)
    except ValueError:
        return False

def evaluate_gsm8k(results_file, dataset_path):
    # Load Results
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # GSM8K doesn't have task_id in standard dataset, usually we use index or question as key
            # run_custom_eval.py needs to ensure task_id is consistent.
            results[entry['task_id']] = entry['completion']

    # Load Dataset
    ds = load_dataset_local(dataset_path)
    
    passed = 0
    total = 0
    
    print(f"Evaluating {len(results)} samples...")
    
    for i, sample in enumerate(ds):
        # We assume task_id in results corresponds to index or some ID
        # In run_custom_eval.py, we should set task_id to string index if not present
        task_id = f"GSM8K/{i}" 
        
        if task_id not in results:
            continue
            
        completion = results[task_id]
        target = sample['answer']
        
        if is_correct(completion, target):
            passed += 1
        total += 1

    if total == 0:
        print("No matching tasks found!")
        return

    acc = passed / total
    print("-" * 30)
    print(f"Total Evaluated: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/root/LLaDA/hf_models/datasets/gsm8k")
    args = parser.parse_args()

    evaluate_gsm8k(args.results_file, args.dataset_path)
