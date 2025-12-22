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

# Timeout for executing a single problem
TIMEOUT = 5.0

def load_dataset_local(path):
    print(f"Loading dataset from {path}...")
    return load_dataset(path=path, split="test")

def extract_code(text):
    """
    Robustly extract code from the model output.
    """
    # 1. Look for a complete code block ```python ... ``` or ``` ... ```
    #    This handles cases where the model wraps the code entirely.
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # 2. Look for a closing backtick that terminates the code.
    #    This handles your specific case: "    code...\n```\n### Explanation"
    if "```" in text:
        return text.split("```")[0]
        
    # 3. Fallback: return original text
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
    # Extract pure code from the completion
    clean_completion = extract_code(completion)
        
    check_program = (
        problem["prompt"] + "\n" + 
        clean_completion + "\n" +
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