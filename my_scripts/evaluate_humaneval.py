import argparse
import json
import multiprocessing
import os
import sys
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm

# Add path to import dLLM-cache metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dLLM-cache/metrics')))
# Import sanitize from the reference script (renamed to avoid import issues if needed, but direct import works)
# We need to import the module dynamically or just copy the functions if the filename is tricky (humaneval_pass@1.py has @)
import importlib.util
spec = importlib.util.spec_from_file_location("humaneval_pass_at_1", "/root/LLaDA/dLLM-cache/metrics/humaneval_pass@1.py")
humaneval_module = importlib.util.module_from_spec(spec)
sys.modules["humaneval_pass_at_1"] = humaneval_module
# Mock hf_evaluate to avoid loading issues when executing the module
import sys
from unittest.mock import MagicMock
sys.modules["evaluate"] = MagicMock()
spec.loader.exec_module(humaneval_module)
sanitize = humaneval_module.sanitize

# Timeout for executing a single problem
TIMEOUT = 5.0

def load_dataset_local(path):
    print(f"Loading dataset from {path}...")
    return load_dataset(path=path, split="test")

def extract_code(text, entry_point):
    """
    Robustly extract code from the model output using the reference script's sanitize function.
    """
    # The sanitize function expects the full text including prompt, but here we usually have just the completion.
    # However, the reference script usage is: sanitize(prompt + completion, entry_point)
    # So we should adapt check_correctness to pass the prompt as well.
    
    # For now, let's try to use sanitize on the completion directly if it's self-contained, 
    # but the robust way is to pass the full context.
    
    # Let's just return the text here and handle the sanitization in check_correctness 
    # where we have access to the prompt and entry_point.
    return text

def unsafe_execute(check_program, result_queue):
    # Redirect stdout/stderr to suppress output from generated code
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        exec_globals = {}
        exec(check_program, exec_globals)
        result_queue.put("passed")
    except Exception as e:
        result_queue.put(f"failed: {str(e)}")

def check_correctness(problem, completion, timeout):
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """
    # Use the reference script's sanitize function
    # It expects the full code (prompt + completion) and the entry point
    
    # First, do a basic cleanup of markdown blocks if present, as sanitize expects raw code or text
    # The reference script does: sample["doc"]["prompt"] + "\n" + sample["resps"][0][0].split("```python\n", 1)[-1].split("```")[0]
    # So we should replicate that pre-processing
    
    raw_completion = completion
    if "```python" in raw_completion:
        raw_completion = raw_completion.split("```python", 1)[-1]
    if "```" in raw_completion:
        raw_completion = raw_completion.split("```")[0]
        
    full_code = problem["prompt"] + "\n" + raw_completion
    
    try:
        clean_code = sanitize(full_code, problem['entry_point'])
    except Exception as e:
        # Fallback if sanitization fails
        print(f"Sanitization failed for task {problem['task_id']}: {e}")
        clean_code = full_code

    check_program = (
        clean_code + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    )

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return False # Timeout

    if not result_queue.empty():
        result = result_queue.get()
        return result == "passed"
    else:
        return False # Process crashed or didn't return

def evaluate_functional_correctness(results_file, dataset_path):
    # Load Results
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            results[entry['task_id']] = entry['completion']

    # Load Dataset
    ds = load_dataset_local(dataset_path)
    
    passed = 0
    total = 0
    
    print(f"Evaluating {len(results)} samples...")
    
    # Prepare tasks
    tasks = []
    for sample in ds:
        task_id = sample['task_id']
        if task_id in results:
            tasks.append((sample, results[task_id]))

    # Run in parallel
    # Use fewer workers than CPUs to avoid overhead issues
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_correctness, t[0], t[1], TIMEOUT) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                passed += 1
            total += 1

    if total == 0:
        print("No matching tasks found!")
        return

    pass_at_1 = passed / total
    print("-" * 30)
    print(f"Total Evaluated: {total}")
    print(f"Passed: {passed}")
    print(f"Pass@1: {pass_at_1:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/root/LLaDA/hf_models/datasets/openai_humaneval")
    args = parser.parse_args()

    evaluate_functional_correctness(args.results_file, args.dataset_path)