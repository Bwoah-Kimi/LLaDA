import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_utils.inference_analysis import collect_similarity_for_pairs, expand_tokenwise_similarity


def plot_latency_distribution(df_results):
    if df_results.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.histplot(df_results['wall_time'], kde=True, bins=10)
    plt.title('Inference Wall Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.axvline(df_results['wall_time'].mean(), color='r', linestyle='--', label=f"Mean: {df_results['wall_time'].mean():.2f}s")
    plt.legend()
    plt.show()


def plot_step_breakdown(df_steps):
    if df_steps.empty:
        return
    plt.figure(figsize=(10, 4))
    steps_idx = np.arange(len(df_steps))
    fwd = df_steps['forward_time'].values
    smp = df_steps['sampling_time'].values
    rmk = df_steps['remasking_time'].values
    plt.bar(steps_idx, fwd, label='forward')
    plt.bar(steps_idx, smp, bottom=fwd, label='sampling')
    plt.bar(steps_idx, rmk, bottom=fwd + smp, label='remasking')
    plt.title('Per-step Latency Breakdown')
    plt.xlabel('Global step')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_token_evolution(df_token_evolution, df_coverage):
    if df_token_evolution.empty:
        return
        
    # 1. Token state heatmap
    plt.figure(figsize=(15, 8))
    steps = sorted(df_token_evolution['step'].unique())
    tokens = sorted(df_token_evolution['token_idx'].unique())
    state_map = {'prompt': 0, 'masked': 1, 'transferred': 2, 'generated': 3}
    state_matrix = np.full((len(steps), len(tokens)), -1)
    
    for i, step in enumerate(steps):
        step_data = df_token_evolution[df_token_evolution['step'] == step]
        for _, row in step_data.iterrows():
            j = row['token_idx']
            state_matrix[i, j] = state_map[row['state']]
            
    colors = ['blue', 'red', 'orange', 'green']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    plt.imshow(state_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    plt.xlabel('Token Index')
    plt.ylabel('Denoising Step')
    plt.title('Token State Evolution During Denoising')
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Prompt', 'Masked', 'Transferred', 'Generated'])
    plt.tight_layout()
    plt.show()
    
    # 2. Coverage evolution
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(df_coverage['step'], df_coverage['mask_ratio'], 'r-', label='Masked Ratio')
    plt.plot(df_coverage['step'], df_coverage['generated_ratio'], 'g-', label='Generated Ratio')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.title('Token Coverage Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df_coverage['step'], df_coverage['masked_tokens'], 'r-', label='Masked')
    plt.plot(df_coverage['step'], df_coverage['generated_tokens'], 'g-', label='Generated')
    plt.plot(df_coverage['step'], df_coverage['transferred_tokens'], 'orange', label='Transferred')
    plt.xlabel('Step')
    plt.ylabel('Token Count')
    plt.title('Token Count by State')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_cache_state_evolution(df_cache):
    if df_cache.empty:
        print("No cache state data found.")
        return

    # Helper to normalize indices
    def get_indices(x):
        if x is None:
            return np.array([])
        if isinstance(x, float) and np.isnan(x): return np.array([]) # Handle NaN
        if hasattr(x, 'cpu'): x = x.cpu().numpy()
        if isinstance(x, list): x = np.array(x)
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 0:
            if x.size == 1: x = np.array([x.item()])
            else: x = np.array([])
        else:
            x = x.flatten()
        return x

    # Infer generation length from transfer indices
    max_idx = 0
    has_indices = False
    if 'transfer_index' in df_cache.columns:
        for idxs in df_cache['transfer_index']:
            indices = get_indices(idxs)
            if indices.size > 0:
                has_indices = True
                max_idx = max(max_idx, np.max(indices))

    gen_len = int(max_idx + 1) if has_indices else 0
    print(f"Inferred generation length from transfer indices: {gen_len}")

    # Define states for visualization
    # 0: Reuse All
    # 1: Refresh Prompt Only
    # 2: Refresh Gen Only
    # 3: Full Refresh
    def get_state(row):
        g = row['refresh_gen']
        p = row['refresh_prompt']
        if g and p: return 3
        if g: return 2
        if p: return 1
        return 0
    
    # Work on a copy to avoid side effects
    df_plot = df_cache.copy()
    df_plot['state_code'] = df_plot.apply(get_state, axis=1)
    
    # Update transfer_count for refresh_gen steps
    if gen_len > 0 and 'transfer_count' in df_plot.columns:
        def adjust_count(row):
            if row['refresh_gen']:
                return gen_len
            return row['transfer_count']
        df_plot['transfer_count'] = df_plot.apply(adjust_count, axis=1)

    # 1. Cache State Heatmap
    pivot_state = df_plot.pivot_table(index='layer_id', columns='step', values='state_code')
    
    plt.figure(figsize=(15, 8))
    
    # Colors: Grey (Reuse), Blue (Prompt), Orange (Gen), Red (Full)
    cmap = ListedColormap(['#e0e0e0', '#2196f3', '#ff9800', '#f44336']) 
    
    ax = sns.heatmap(pivot_state, cmap=cmap, cbar=True, linewidths=0.5, linecolor='white')
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(['Reuse', 'Prompt Only', 'Gen Only', 'Full Refresh'])
    
    plt.title('dLLM-Cache State Evolution (Layer vs Step)')
    plt.xlabel('Denoising Step')
    plt.ylabel('Layer ID')
    plt.tight_layout()
    plt.show()
    
    # 2. Transfer Count Heatmap
    if 'transfer_count' in df_plot.columns and df_plot['transfer_count'].sum() > 0:
        pivot_transfer = df_plot.pivot_table(index='layer_id', columns='step', values='transfer_count', fill_value=0)
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_transfer, cmap='Greens', annot=False, cbar_kws={'label': 'Transferred Tokens'})
        plt.title('Token Transfer Count (Layer vs Step)')
        plt.xlabel('Denoising Step')
        plt.ylabel('Layer ID')
        plt.tight_layout()
        plt.show()

        # 3. Detailed Token Transfer Analysis
        if 'transfer_index' in df_plot.columns:
            print("Analyzing detailed token transfers...")
            transfer_data = []

            # Pre-compute full range indices for refresh steps
            full_indices = np.arange(gen_len) if gen_len > 0 else np.array([])

            for _, row in df_plot.iterrows():
                indices = np.array([])
                
                if row['refresh_gen']:
                    # If refreshing generation, all tokens are effectively transferred/recomputed
                    indices = full_indices
                elif row['transfer']:
                    # Partial transfer
                    indices = get_indices(row['transfer_index'])
                
                if indices.size > 0:
                    # Create rows for DataFrame construction
                    transfer_data.extend([
                        {'step': row['step'], 'layer_id': row['layer_id'], 'token_idx': int(idx)}
                        for idx in indices
                    ])
            
            if transfer_data:
                df_transfer = pd.DataFrame(transfer_data)
                
                # Plot 3a: Step vs Token (Aggregated Layers)
                # Shows which tokens are hot (transferred by many layers) at each step
                pivot_token_step = df_transfer.pivot_table(
                    index='token_idx', 
                    columns='step', 
                    values='layer_id', 
                    aggfunc='nunique', 
                    fill_value=0
                )
                
                plt.figure(figsize=(15, 8))
                sns.heatmap(pivot_token_step, cmap='Blues', cbar_kws={'label': 'Layer Count'})
                plt.title('Token Transfer Frequency (Step vs Token)\nColor: Number of layers transferring this token')
                plt.xlabel('Denoising Step')
                plt.ylabel('Token Index')
                plt.tight_layout()
                plt.show()

                # Plot 3b: Layer vs Token (Aggregated Steps)
                # Shows which tokens each layer tends to transfer
                pivot_token_layer = df_transfer.pivot_table(
                    index='layer_id', 
                    columns='token_idx', 
                    values='step', 
                    aggfunc='nunique',
                    fill_value=0
                )
                
                plt.figure(figsize=(15, 8))
                sns.heatmap(pivot_token_layer, cmap='Reds', cbar_kws={'label': 'Step Count'})
                plt.title('Token Transfer Frequency (Layer vs Token)\nColor: Number of steps where this token was transferred')
                plt.xlabel('Token Index')
                plt.ylabel('Layer ID')
                plt.tight_layout()
                plt.show()
    else:
        print("No token transfers detected.")


def plot_layer_token_evolution(df_cache, layer_ids):
    """
    Plots token state evolution for specific layers.
    X-axis: Token Index
    Y-axis: Denoising Step
    Color: State (Reuse, Refresh, Transfer)
    """
    if df_cache.empty:
        print("No cache state data found.")
        return

    # Helper to normalize indices
    def get_indices(x):
        if x is None: return np.array([])
        if isinstance(x, float) and np.isnan(x): return np.array([])
        if hasattr(x, 'cpu'): x = x.cpu().numpy()
        if isinstance(x, list): x = np.array(x)
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 0:
            if x.size == 1: x = np.array([x.item()])
            else: x = np.array([])
        else:
            x = x.flatten()
        return x

    # Infer generation length
    max_idx = 0
    has_indices = False
    if 'transfer_index' in df_cache.columns:
        for idxs in df_cache['transfer_index']:
            arr = get_indices(idxs)
            if arr.size > 0:
                max_idx = max(max_idx, np.max(arr))
                has_indices = True
    
    gen_len = int(max_idx + 1) if has_indices else 0
    if gen_len == 0:
        print("Could not infer generation length. Cannot plot token evolution.")
        return

    full_indices = np.arange(gen_len)

    # Define states
    # 0: Reuse (No Update)
    # 1: Transfer (Partial Update)
    # 2: Refresh (Full Update - Prompt or Gen)
    
    for layer_id in layer_ids:
        layer_data = df_cache[df_cache['layer_id'] == layer_id].sort_values('step')
        if layer_data.empty:
            print(f"No data for Layer {layer_id}")
            continue
            
        steps = layer_data['step'].unique()
        state_matrix = np.zeros((len(steps), gen_len))
        
        for i, step in enumerate(steps):
            row = layer_data[layer_data['step'] == step].iloc[0]
            
            # Check for full refresh (Prompt or Gen)
            if row['refresh_gen'] or row['refresh_prompt']:
                state_matrix[i, :] = 2 # Refresh
            elif row['transfer']:
                indices = get_indices(row['transfer_index'])
                if indices.size > 0:
                    # Ensure indices are within bounds
                    valid_indices = indices[indices < gen_len].astype(int)
                    state_matrix[i, valid_indices] = 1 # Transfer
            # Else remains 0 (Reuse)

        plt.figure(figsize=(15, 6))
        from matplotlib.colors import ListedColormap
        # Colors: Grey (Reuse), Orange (Transfer), Red (Refresh)
        cmap = ListedColormap(['#e0e0e0', '#ff9800', '#f44336'])
        
        plt.imshow(state_matrix.T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)
        
        plt.title(f'Cache State Evolution - Layer {layer_id}')
        plt.ylabel('Token Index')
        plt.xlabel('Denoising Step')
        
        # Set X-ticks to show actual step numbers
        step_indices = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
        plt.xticks(step_indices, [steps[i] for i in step_indices])
        
        # Custom Legend
        cbar = plt.colorbar(ticks=[0.33, 1, 1.66])
        cbar.set_ticklabels(['Reuse', 'Transfer', 'Refresh'])
        
        plt.tight_layout()
        plt.show()


def plot_similarity_heatmap(df_tok, step_pair_str, t_name, block_boundaries, prompt_len, seq_len, ax_heatmap, show_x_label=True):
    """Helper function to plot a single similarity heatmap."""
    # Pivot for heatmap
    pivot = (
        df_tok.pivot_table(index="layer_idx", columns="token_idx", values="token_similarity", aggfunc="mean")
        .sort_index()
    )
    
    im = ax_heatmap.imshow(
        pivot.values,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        aspect='auto',
        interpolation='nearest'
    )
    
    # Add boundaries
    for boundary in block_boundaries:
        if boundary < seq_len:
            ax_heatmap.axvline(x=boundary-0.5, color='black', linestyle='-', linewidth=2, alpha=0.7)
    ax_heatmap.axvline(x=prompt_len-0.5, color='purple', linestyle='-', linewidth=3, alpha=0.8)
    
    ax_heatmap.set_title(f"{step_pair_str}: {t_name} Cosine Similarity")
    ax_heatmap.set_ylabel("Layer Index")
    
    if not show_x_label:
        ax_heatmap.set_xticks([])
    else:
        ax_heatmap.set_xlabel("Token Index")

    if len(pivot.index) <= 20:
        ax_heatmap.set_yticks(range(len(pivot.index)))
        ax_heatmap.set_yticklabels(pivot.index)
    
    plt.colorbar(im, ax=ax_heatmap, label="Sim")


def plot_token_state_strips(token_states_a, token_states_b, block_idx_a, block_idx_b, step_pair_str, ax_state_a, ax_state_b):
    """Helper function to plot token state strips."""
    # Plot State Strips
    state_colors = {'prompt': 0, 'masked': 1, 'transferred': 2, 'generated': 3, 'unknown': 4}
    
    # State A
    state_array_a = [state_colors.get(state, 4) for state in token_states_a]
    ax_state_a.imshow([state_array_a], cmap=plt.cm.Set3, aspect='auto', interpolation='nearest', vmin=0, vmax=11)
    ax_state_a.set_title(f"Token States at Step {step_pair_str.split('→')[0]} (Block {block_idx_a})")
    ax_state_a.set_yticks([])
    ax_state_a.set_xticks([]) 
    
    # State B
    state_array_b = [state_colors.get(state, 4) for state in token_states_b]
    ax_state_b.imshow([state_array_b], cmap=plt.cm.Set3, aspect='auto', interpolation='nearest', vmin=0, vmax=11)
    ax_state_b.set_title(f"Token States at Step {step_pair_str.split('→')[1]} (Block {block_idx_b})")
    ax_state_b.set_xlabel("Token Index")
    ax_state_b.set_yticks([])
    
    # Legend        
    legend_elements = [
        Patch(facecolor=plt.cm.Set3(0), label='Prompt'),
        Patch(facecolor=plt.cm.Set3(1), label='Masked'),
        Patch(facecolor=plt.cm.Set3(2), label='Transferred'),
        Patch(facecolor=plt.cm.Set3(3), label='Generated')
    ]
    ax_state_b.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))


def analyze_similarity_statistics(full_df, save_dir=None):
    """Performs detailed statistical analysis on similarity data."""
    print("\n" + "="*80)
    print("Cosine Similarity Distribution Analysis")
    print("="*80)
    
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for t_type in full_df['tensor_type'].unique():
        subset = full_df[full_df['tensor_type'] == t_type].copy()
        print(f"\nTensor Type: {t_type}")
        print(f"Total data points: {len(subset)}")
        
        # 1. Global Statistics
        print("\n--- Global Statistics ---")
        stats = subset['token_similarity'].quantile(percentiles)
        print(f"{'Percentile':<12} | {'Similarity':<12}")
        print("-" * 30)
        for p in percentiles:
            print(f"{p*100:>5.1f}%       | {stats[p]:.4f}")

        # 2. Evolution over Steps (Step Pairs)
        print("\n--- Evolution over Steps (Median Similarity) ---")
        # Extract step number from step_pair string "X→Y" -> X
        subset['step_num'] = subset['step_pair'].apply(lambda x: int(x.split('→')[0]))
        step_stats = subset.groupby('step_num')['token_similarity'].quantile(percentiles).unstack()
        print(step_stats.round(4))

        # 3. Evolution over Layers
        print("\n--- Evolution over Layers (Median Similarity) ---")
        layer_stats = subset.groupby('layer_idx')['token_similarity'].quantile(percentiles).unstack()
        print(layer_stats.round(4))
        
        # Optional: Heatmap of Median Similarity (Layer vs Step)
        print("\n--- Median Similarity Matrix (Layer vs Step) ---")
        pivot_median = subset.pivot_table(index='layer_idx', columns='step_num', values='token_similarity', aggfunc='median')
        print(pivot_median.round(3))
        
        if save_dir:
            print(f"\nSaving statistics to {save_dir}...")
            step_stats.to_csv(os.path.join(save_dir, f'{t_type}_similarity_by_step.csv'))
            layer_stats.to_csv(os.path.join(save_dir, f'{t_type}_similarity_by_layer.csv'))
            pivot_median.to_csv(os.path.join(save_dir, f'{t_type}_similarity_matrix.csv'))


def enhanced_cosine_similarity_with_full_evolution(df_token_evolution, tensor_dir, step_pairs, tensor_name, show_fig=True, save_dir=None):
    """
    Analyze cosine similarity using the pre-computed token evolution DataFrame.
    tensor_name: str or list of str, e.g. ['attn_out', 'ff_out']
    """
    TENSOR_NAMES = None
    MODE = "token-wise"
    
    if df_token_evolution.empty:
        print("No token evolution data provided.")
        return

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    seq_len = df_token_evolution['token_idx'].max() + 1
    first_step = df_token_evolution['step'].min()
    prompt_mask = df_token_evolution[df_token_evolution['step'] == first_step]['state'] == 'prompt'
    prompt_len = prompt_mask.sum()
    gen_len = seq_len - prompt_len
    block_length = 32
    
    block_boundaries = []
    for i in range(0, gen_len, block_length):
        block_boundaries.append(prompt_len + i)
    if prompt_len + gen_len not in block_boundaries:
        block_boundaries.append(prompt_len + gen_len)

    def get_states_for_step(step):
        step_data = df_token_evolution[df_token_evolution['step'] == step].sort_values('token_idx')
        if step_data.empty:
            return None, None
        return step_data['state'].tolist(), step_data['block_idx'].iloc[0]

    # Handle single string or list input
    target_tensors = [tensor_name] if isinstance(tensor_name, str) else tensor_name

    # Accumulator for statistics
    all_similarity_records = []
    
    for step_pair_tuple in step_pairs:
        step_a, step_b = step_pair_tuple
        step_pair_str = f"{step_a}→{step_b}"
        
        token_states_a, block_idx_a = get_states_for_step(step_a)
        token_states_b, block_idx_b = get_states_for_step(step_b)
        
        if token_states_a is None or token_states_b is None:
            print(f"Missing token states for steps {step_a} or {step_b}")
            continue

        n_tensors = len(target_tensors)
        height_ratios = [3] * n_tensors + [0.2, 0.2]
        total_height = 4 * n_tensors + 2 
        
        fig, axes = plt.subplots(n_tensors + 2, 1, figsize=(15, total_height), 
                                gridspec_kw={'height_ratios': height_ratios})
        
        # Ensure axes is always a list/array even if n_tensors=1 (total=3)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        for idx, t_name in enumerate(target_tensors):
            ax = axes[idx]
            
            df_sim = collect_similarity_for_pairs(
                tensor_dir,
                [step_pair_tuple],
                selector=t_name,
                tensor_names=TENSOR_NAMES,
                mode=MODE,
            )
            
            if df_sim.empty:
                ax.text(0.5, 0.5, f"No data for {t_name}", ha='center', va='center')
                continue

            if MODE == "token-wise":
                df_tok = expand_tokenwise_similarity(df_sim)
                if df_tok.empty:
                    ax.text(0.5, 0.5, f"No token-wise data for {t_name}", ha='center', va='center')
                    continue

                # Collect data for statistics
                df_tok['tensor_type'] = t_name
                all_similarity_records.append(df_tok)
                
                # Use helper function for plotting heatmap only
                plot_similarity_heatmap(
                    df_tok, step_pair_str, t_name, block_boundaries, prompt_len, seq_len,
                    ax, show_x_label=(idx == n_tensors - 1)
                )

        # Plot state strips once at the bottom
        plot_token_state_strips(
            token_states_a, token_states_b, block_idx_a, block_idx_b, 
            step_pair_str, axes[-2], axes[-1]
        )

        plt.tight_layout()
        if save_dir is not None:
            fig_path = os.path.join(save_dir, f"cosine_similarity_{step_pair_str.replace('→', '_to_')}.png")
            fig.savefig(fig_path)
            plt.close(fig)
        
        # Show figure if required
        if show_fig:
            plt.show()
        else:
            plt.close(fig)
        
    # --- Statistical Analysis ---
    if all_similarity_records:
        full_df = pd.concat(all_similarity_records, ignore_index=True)
        analyze_similarity_statistics(full_df, save_dir=save_dir)