import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from dataclasses import asdict

# Add path to import dLLM-Cache
cur_dir = os.path.dirname(os.path.abspath(__file__))
dllm_cache_dir = os.path.join(cur_dir, '../dLLM-cache')
sys.path.append(dllm_cache_dir)

from dllm_cache.cache import dLLMCache, dLLMCacheConfig
from dllm_cache.hooks import register_cache_LLaDA

def load_model_and_tokenizer(model_path, cache_dir, device='cuda', enable_dllm_cache=False, cache_config_dict=None):
    print(f"Loading model from {model_path} (cache: {cache_dir})...")
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        local_files_only=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True
    )
    
    if enable_dllm_cache and cache_config_dict:
        cache_config = dLLMCacheConfig(**cache_config_dict)
        dLLMCache.new_instance(**asdict(cache_config))
        register_cache_LLaDA(model, "model.transformer.blocks")
        print(f"dLLM-Cache enabled with config: {cache_config}")
        
    return model, tokenizer

def load_dataset_shard(dataset_name, dataset_path, split, shard_id, num_shards):
    from datasets import load_dataset
    print(f"Loading {dataset_name} dataset from {dataset_path}...")
    
    try:
        # Try loading with default config
        ds = load_dataset(path=dataset_path, split=split)
    except:
        # Fallback for GSM8K which often needs 'main'
        ds = load_dataset(path=dataset_path, name="main", split=split)
    
    # Add global index to track samples across shards
    if 'global_index' not in ds.column_names:
        ds = ds.map(lambda _, idx: {'global_index': idx}, with_indices=True)
    
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_id)
        print(f"Loaded shard {shard_id}/{num_shards} with {len(ds)} samples.")
    else:
        print(f"Loaded full dataset with {len(ds)} samples.")
        
    return ds
