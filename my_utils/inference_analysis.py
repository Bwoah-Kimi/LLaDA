import os
import glob
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

_layer_re = re.compile(
    r".*_(?:layers|transformer_blocks)_(\d+).*_(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)\.pt$"
)

def load_token_state(base_dir, step):
    """Load token state data for a specific step."""
    step_dir = os.path.join(base_dir, f"step_{step}")
    token_state_path = os.path.join(step_dir, "token_state.pt")
    if os.path.exists(token_state_path):
        return torch.load(token_state_path, map_location="cpu")
    return None

def collect_token_evolution(base_dir, steps=None):
    """
    Collect token state evolution across all steps.
    Returns DataFrame with columns: step, token_idx, state, block_idx, inblock_step
    """
    if steps is None:
        step_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("step_")],
                          key=lambda x: int(x.split('_')[1]))
        steps = [int(d.split('_')[1]) for d in step_dirs]
    
    rows = []
    for step in steps:
        token_state = load_token_state(base_dir, step)
        if token_state is None:
            continue
            
        prompt_mask = token_state["prompt_mask"][0]
        mask_index = token_state["mask_index"][0]
        transfer_index = token_state["transfer_index"][0]
        block_idx = token_state["block_idx"]
        inblock_step = token_state["inblock_step"]
        
        seq_len = prompt_mask.shape[0]
        for token_idx in range(seq_len):
            if prompt_mask[token_idx]:
                state = "prompt"
            elif mask_index[token_idx]:
                if transfer_index[token_idx]:
                    state = "transferred"
                else:
                    state = "masked"
            else:
                state = "generated"
            
            rows.append({
                "step": step,
                "token_idx": token_idx,
                "state": state,
                "block_idx": block_idx,
                "inblock_step": inblock_step,
                "is_prompt": prompt_mask[token_idx].item(),
                "is_masked": mask_index[token_idx].item(),
                "is_transferred": transfer_index[token_idx].item()
            })
    
    return pd.DataFrame(rows)

def analyze_token_transitions(df_tokens):
    """Analyze how tokens transition between states."""
    transitions = []
    for token_idx in df_tokens['token_idx'].unique():
        token_history = df_tokens[df_tokens['token_idx'] == token_idx].sort_values('step')
        prev_state = None
        for _, row in token_history.iterrows():
            current_state = row['state']
            if prev_state is not None and prev_state != current_state:
                transitions.append({
                    'token_idx': token_idx,
                    'step': row['step'],
                    'from_state': prev_state,
                    'to_state': current_state,
                    'block_idx': row['block_idx']
                })
            prev_state = current_state
    return pd.DataFrame(transitions)

def get_token_coverage_stats(df_tokens):
    """Get statistics about token coverage at each step."""
    coverage_stats = []
    for step in sorted(df_tokens['step'].unique()):
        step_data = df_tokens[df_tokens['step'] == step]
        total_tokens = len(step_data)
        masked_tokens = len(step_data[step_data['state'] == 'masked'])
        generated_tokens = len(step_data[step_data['state'] == 'generated'])
        transferred_tokens = len(step_data[step_data['state'] == 'transferred'])
        prompt_tokens = len(step_data[step_data['state'] == 'prompt'])
        
        coverage_stats.append({
            'step': step,
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'masked_tokens': masked_tokens,
            'generated_tokens': generated_tokens,
            'transferred_tokens': transferred_tokens,
            'mask_ratio': masked_tokens / total_tokens if total_tokens > 0 else 0,
            'generated_ratio': generated_tokens / total_tokens if total_tokens > 0 else 0,
            'block_idx': step_data['block_idx'].iloc[0] if len(step_data) > 0 else None,
            'inblock_step': step_data['inblock_step'].iloc[0] if len(step_data) > 0 else None
        })
    return pd.DataFrame(coverage_stats)

def load_step_tensors(base_dir, step, selector_substrings):
    step_dir = os.path.join(base_dir, f"step_{step}")
    tensors = {}
    if not os.path.isdir(step_dir):
        return tensors
    for f in glob.glob(os.path.join(step_dir, "*.pt")):
        base = os.path.basename(f)
        if any(s in base for s in selector_substrings):
            tensors[base] = torch.load(f, map_location="cpu")
    return tensors

def parse_layer_and_kind(layer_file_basename):
    m = _layer_re.match(layer_file_basename)
    if m:
        return int(m.group(1)), m.group(2)
    for k in ['q_proj','k_proj','v_proj','attn_out','ff_proj','up_proj','ff_out']:
        if k in layer_file_basename:
            return None, k
    return None, None

def reduce_stat(t, how="mean_abs", sample=None):
    x = t
    if sample:
        b, s, h = x.shape
        if 'seq' in sample:
            x = x[:, torch.linspace(0, s-1, steps=min(sample['seq'], s), dtype=torch.long), :]
        if 'hidden' in sample:
            x = x[:, :, torch.linspace(0, h-1, steps=min(sample['hidden'], h), dtype=torch.long)]
    x = x.float()
    if how == "mean_abs":
        return x.abs().mean().item()
    if how == "max_abs":
        return x.abs().amax().item()
    if how == "std":
        return x.std().item()
    if how == "l2":
        return x.pow(2).mean().sqrt().item()
    raise ValueError(how)

def collect_series(base_dir, selectors=('q_proj','k_proj'), how="mean_abs", sample=None, steps=None):
    rows = []
    if steps is None:
        step_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("step_")],
                           key=lambda x: int(x.split('_')[1]))
        steps = [int(d.split('_')[1]) for d in step_dirs]
    for step in steps:
        tensors = load_step_tensors(base_dir, step, selectors)
        for name, t in tensors.items():
            layer_idx, kind = parse_layer_and_kind(name)
            if kind is None: 
                continue
            val = reduce_stat(t, how=how, sample=sample)
            rows.append({"step": step, "layer_idx": layer_idx, "kind": kind, "value": val, "name": name})
    return pd.DataFrame(rows)

def flatten_activation(t, mode="batch_seq_mean"):
    x = t.float()
    if mode == "batch_seq_mean":
        return x.mean(dim=(0, 1))
    if mode == "batch_mean":
        return x.mean(dim=0).reshape(-1, x.size(-1))
    if mode == "none":
        return x.reshape(-1, x.size(-1))
    raise ValueError(mode)

def _token_matrix(x):
    x = x.float()
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.view(-1, x.size(-1))
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)} for token-wise mode")

def cosine_similarity_tensors(t_a, t_b, mode="batch_seq_mean"):
    if mode == "token-wise":
        a = _token_matrix(t_a)
        b = _token_matrix(t_b)
        sims = F.cosine_similarity(a, b, dim=-1)
        return sims.cpu().numpy()
    elif mode == 'batch_seq_mean':
        a = flatten_activation(t_a, 'batch_seq_mean')
        b = flatten_activation(t_b, 'batch_seq_mean')
        if a.ndim == 1:
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        sims = F.cosine_similarity(a, b, dim=-1)
        return sims.mean().item()
    elif mode == 'batch_mean':
        a = flatten_activation(t_a, 'batch_mean')
        b = flatten_activation(t_b, 'batch_mean')
        sims = F.cosine_similarity(a, b, dim=-1)
        return sims.mean().item()
    else:
        raise ValueError(mode)

def collect_similarity_for_pairs(base_dir, step_pairs, selector="attn_out", tensor_names=None, mode="batch_seq_mean"):
    rows = []
    for step_a, step_b in step_pairs:
        tensors_a = load_step_tensors(base_dir, step_a, [selector])
        tensors_b = load_step_tensors(base_dir, step_b, [selector])
        candidates = tensors_a.keys() if not tensor_names else [
            name for name in tensors_a if any(tag in name for tag in tensor_names)
        ]
        for name in candidates:
            if name not in tensors_b:
                continue
            layer_idx, kind = parse_layer_and_kind(name)
            sim = cosine_similarity_tensors(tensors_a[name], tensors_b[name], mode=mode)
            rows.append({
                "step_a": step_a,
                "step_b": step_b,
                "layer_idx": layer_idx,
                "kind": kind,
                "name": name,
                "similarity": sim,
            })
    return pd.DataFrame(rows)

def expand_tokenwise_similarity(df_sim):
    rows = []
    for _, row in df_sim.iterrows():
        sims = row["similarity"]
        if isinstance(sims, np.ndarray) and row["layer_idx"] is not None:
            for token_idx, val in enumerate(sims):
                rows.append({
                    "step_pair": f"{row['step_a']}→{row['step_b']}",
                    "layer_idx": row["layer_idx"],
                    "name": row["name"],
                    "token_idx": token_idx,
                    "token_similarity": float(val),
                })
    return pd.DataFrame(rows)