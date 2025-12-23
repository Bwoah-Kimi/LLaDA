import argparse
import os
import sys
import torch
import numpy as np
import json
from tqdm import tqdm
from datasets import load_dataset

# Add path to import my_utils and root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_utils.eval_common import load_model_and_tokenizer

# Import get_log_likelihood from root
try:
    from get_log_likelihood import get_log_likelihood
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from get_log_likelihood import get_log_likelihood

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL/Log-Likelihood on LLaDA.")
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='/root/LLaDA/hf_models/hub')
    parser.add_argument('--dataset_name', type=str, default='wikitext')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--output_dir', type=str, default='eval_results_custom/ppl')
    parser.add_argument('--mc_num', type=int, default=16, help="Monte Carlo samples for estimation")
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, cache_dir=args.cache_dir, device=device)
    
    print(f"Loading dataset {args.dataset_name} ({args.dataset_config})...")
    try:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.max_samples:
        ds = ds.select(range(min(len(ds), args.max_samples)))
        
    nlls = []
    results = []
    
    print(f"Evaluating on {len(ds)} samples...")
    for i, item in enumerate(tqdm(ds)):
        text = item[args.text_column]
        if not text.strip():
            continue
            
        # Tokenize
        # We treat the whole text as the 'answer' and prompt as empty
        prompt_ids = torch.tensor([], dtype=torch.long, device=device)
        answer_ids = torch.tensor(tokenizer(text)['input_ids'], device=device)
        if len(answer_ids) == 0:
            continue
            
        # Calculate Log Likelihood
        # get_log_likelihood returns the TOTAL log likelihood (sum) for the sequence
        ll_sum = get_log_likelihood(model, prompt_ids, answer_ids, mc_num=args.mc_num)
        
        if isinstance(ll_sum, torch.Tensor):
            ll_sum = ll_sum.item()

        # Normalize by length to get average log likelihood per token
        # PPL = exp(-1/N * sum(log P(x)))
        seq_len = len(answer_ids)
        nll = -ll_sum / seq_len
        
        nlls.append(nll)
        
        # Safe PPL calculation
        if nll > 100:
            ppl_val = float('inf')
        else:
            ppl_val = np.exp(nll)

        results.append({
            "id": i,
            "text_len": seq_len,
            "nll": nll,
            "ppl": ppl_val
        })
        
    if not nlls:
        print("No valid samples found.")
        return

    mean_nll = np.mean(nlls)
    
    if mean_nll > 100:
        ppl = float('inf')
    else:
        ppl = np.exp(mean_nll)
    
    print(f"\nResults:")
    print(f"Mean NLL: {mean_nll:.4f}")
    print(f"PPL: {ppl:.4f}")
    
    output_file = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "config": vars(args),
            "mean_nll": mean_nll,
            "ppl": ppl,
            "details": results
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
