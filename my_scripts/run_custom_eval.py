import argparse
import os
import sys
import json
import torch
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_utils.eval_common import load_model_and_tokenizer, load_dataset_shard
from my_utils.run_inference import profiled_generate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dLLM-cache')))
from dllm_cache.cache import dLLMCache


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaDA evaluation.")
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='/root/LLaDA/hf_models/hub')
    parser.add_argument('--dataset_name', type=str, default='humaneval')
    parser.add_argument('--dataset_path', type=str, default='/root/LLaDA/hf_models/datasets/openai_humaneval')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Generation params
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--mask_id', type=int, default=126336)
    
    # dLLM Cache params
    parser.add_argument('--enable_dllm_cache', action='store_true')
    parser.add_argument('--cache_prompt_interval', type=int, default=50)
    parser.add_argument('--cache_gen_interval', type=int, default=7)
    parser.add_argument('--cache_transfer_ratio', type=float, default=0.25)
    parser.add_argument('--cache_similarity_threshold', type=float, default=None)
    parser.add_argument('--cache_strategy', type=str, default='ratio')
    return parser.parse_args()


def main():
    args = parse_args()
    
    cache_config = {
        'prompt_interval_steps': args.cache_prompt_interval,
        'gen_interval_steps': args.cache_gen_interval,
        'transfer_ratio': args.cache_transfer_ratio,
        'similarity_threshold': args.cache_similarity_threshold,
        'cache_strategy': args.cache_strategy,
    }
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.cache_dir, 
        enable_dllm_cache=args.enable_dllm_cache,
        cache_config_dict=cache_config
    )
    
    ds = load_dataset_shard(args.dataset_name, args.dataset_path, "test", args.shard_id, args.num_shards)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}_shard_{args.shard_id}.jsonl")
    
    # Clear file if exists to avoid duplicates on rerun
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Batch processing
    batch_data = []
    
    for sample in tqdm(ds, desc=f"Shard {args.shard_id}"):
        # Handle different dataset formats
        if args.dataset_name == 'humaneval':
            task_id = sample['task_id']
            prompt_text = sample['prompt']
        elif args.dataset_name == 'gsm8k':
             task_id = f"GSM8K/{sample['global_index']}"
             # Standard zero-shot prompt for GSM8K
             prompt_text = f"Question: {sample['question']}\nAnswer:"
        elif args.dataset_name == 'mmlu':
            task_id = f"MMLU/{sample['global_index']}"
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample['choices'])])
            prompt_text = f"Question: {sample['question']}\nChoices:\n{choices_str}\nAnswer:"
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")
        
        batch_data.append({"task_id": task_id, "prompt_text": prompt_text})
        
        if len(batch_data) >= args.batch_size:
            process_batch(batch_data, model, tokenizer, args, output_file)
            batch_data = []
            
    # Process remaining
    if batch_data:
        process_batch(batch_data, model, tokenizer, args, output_file)


def process_batch(batch_data, model, tokenizer, args, output_file):
    prompts = [item['prompt_text'] for item in batch_data]
    
    # Tokenize with padding
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    prompt_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Reset dLLM-Cache if enabled
    if args.enable_dllm_cache:
        cache = dLLMCache()
        cache.reset_cache(prompt_ids.shape[1])
        cache.step_logs = {}

    generated_tensor, _ = profiled_generate(
        model,
        prompt_ids,
        attention_mask=attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=0,
        cfg_scale=0.0,
        remasking='low_confidence',
        mask_id=args.mask_id,
    )
    
    # Decode and save
    for i, item in enumerate(batch_data):
        # Extract generation part
        gen_ids = generated_tensor[i][prompt_ids.shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        result_entry = {
            "task_id": item['task_id'],
            "completion": completion
        }
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(result_entry) + "\n")


if __name__ == "__main__":
    main()
