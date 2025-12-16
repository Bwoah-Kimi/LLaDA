import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
dllm_cache_dir = os.path.join(cur_dir, '../dLLM-cache')
sys.path.append(dllm_cache_dir)
from dllm_cache.cache import dLLMCache

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


class ActivationProfiler:
    def __init__(self, model, target_layers, save_dir):
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.save_dir = save_dir
        self.current_step = 0
        self.buffer = {} # name -> tensor
        os.makedirs(self.save_dir, exist_ok=True)

    def register_hooks(self):
        self.clear()
        for name, module in self.model.named_modules():
            if any(name.endswith(t) for t in self.target_layers):
                hook = module.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)
        print(f"Registered hooks on {len(self.hooks)} layers.")

    def get_hook(self, name):
        def hook(module, input, output):
            self.buffer[name] = output.detach().cpu()
        return hook
    
    def step(self, current_step_index):
        self.current_step = current_step_index

    def save_buffer(self):
        if not self.buffer:
            return
        step_dir = os.path.join(self.save_dir, f"step_{self.current_step}")
        os.makedirs(step_dir, exist_ok=True)
        for name, tensor in self.buffer.items():
            safe_name = name.replace('.', '_')
            file_path = os.path.join(step_dir, f"{safe_name}.pt")
            torch.save(tensor, file_path)
        self.buffer = {}

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.buffer = {}
        self.current_step = 0
        
    def get_collected_data(self):
        return self.save_dir


def save_token_state(save_dir, step_idx, prompt_mask, mask_index, transfer_index, block_idx, inblock_step):
    """Persist token-role metadata for later visualization."""
    os.makedirs(save_dir, exist_ok=True)
    step_dir = os.path.join(save_dir, f"step_{step_idx}")
    os.makedirs(step_dir, exist_ok=True)
    payload = {
        "prompt_mask": prompt_mask.to(dtype=torch.bool).cpu(),
        "mask_index": mask_index.to(dtype=torch.bool).cpu(),
        "transfer_index": transfer_index.to(dtype=torch.bool).cpu(),
        "block_idx": block_idx,
        "inblock_step": inblock_step,
    }
    torch.save(payload, os.path.join(step_dir, "token_state.pt"))


def save_cache_state(save_dir, step_idx, cache_data):
    """Persist dLLM-Cache state for later visualization."""
    step_dir = os.path.join(save_dir, f"step_{step_idx}")
    torch.save(cache_data, os.path.join(step_dir, "cache_state.pt"))


@torch.no_grad()
def profiled_generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, profiler=None, with_dllm_cache=False):
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    step_details = []
    
    print(f'Generation length: {gen_length}')
    print(f'Denoising steps per block: {steps}')
    print(f'Number of blocks: {num_blocks}')

    def synchronize():
        if x.device.type == 'cuda':
            torch.cuda.synchronize()

    synchronize()
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            global_step = num_block * steps + i
            if profiler:
                profiler.step(global_step)
            
            synchronize()
            step_start = time.perf_counter()
            
            mask_index = (x == mask_id)
            
            # 1. Model Forward Pass
            t_forward_start = time.perf_counter()
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            synchronize()
            t_forward_end = time.perf_counter()

            if profiler:
                profiler.save_buffer()

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            # 2. Sampling
            t_sample_start = time.perf_counter()
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) 
            synchronize()
            t_sample_end = time.perf_counter()

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # 3. Remasking Strategy
            t_remask_start = time.perf_counter()
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if profiler is not None:
                save_token_state(
                    save_dir=profiler.save_dir,
                    step_idx=global_step,
                    prompt_mask=prompt_index,
                    mask_index=mask_index,
                    transfer_index=transfer_index,
                    block_idx=num_block,
                    inblock_step=i,
                )

                if with_dllm_cache:
                    cache = dLLMCache()
                    cache_step = global_step + 1 # 1-based step index
                    if hasattr(cache, 'step_logs') and cache_step in cache.step_logs:
                        cache_data = cache.step_logs[cache_step]
                        save_cache_state(
                            save_dir=profiler.save_dir,
                            step_idx=global_step,
                            cache_data=cache_data,
                        )
                        del cache.step_logs[cache_step] # Clean up to avoid issues in next steps
            
            synchronize()
            t_remask_end = time.perf_counter()
            
            step_end = time.perf_counter()
            step_duration = step_end - step_start
            
            step_details.append({
                "block_idx": num_block,
                "step_idx": i,
                "duration": step_duration,
                "forward_time": t_forward_end - t_forward_start,
                "sampling_time": t_sample_end - t_sample_start,
                "remasking_time": t_remask_end - t_remask_start,
                "num_masks": mask_index.sum().item()
            })

    return x, step_details


def run_inference(model, tokenizer, prompt_text, steps=64, gen_length=64, block_length=32, profiler=None, device='cuda', with_dllm_cache=False):
    """
    Runs inference and measures wall time with detailed profiling.
    """
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    if with_dllm_cache:
        cache = dLLMCache()
        cache.reset_cache(input_ids.shape[1])
        cache.step_logs = {}

    start_time = time.perf_counter()
    
    with torch.no_grad():
        out, step_details = profiled_generate(
            model, 
            input_ids,
            attention_mask=attention_mask,
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length, 
            temperature=0., 
            cfg_scale=0., 
            remasking='low_confidence',
            profiler=profiler,
            with_dllm_cache=with_dllm_cache
        )
    
    end_time = time.perf_counter()
    wall_time = end_time - start_time
    
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    return generated_text, wall_time, step_details