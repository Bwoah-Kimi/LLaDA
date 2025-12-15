import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

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

    # Pre-fetch data for all tensors to avoid re-reading inside the loop if possible,
    # or just iterate step pairs first.
    
    for step_pair_tuple in step_pairs:
        step_a, step_b = step_pair_tuple
        step_pair_str = f"{step_a}→{step_b}"
        
        token_states_a, block_idx_a = get_states_for_step(step_a)
        token_states_b, block_idx_b = get_states_for_step(step_b)
        
        if token_states_a is None or token_states_b is None:
            print(f"Missing token states for steps {step_a} or {step_b}")
            continue

        # Calculate layout: 
        # For N tensors, we need N heatmaps + 2 state strips (top and bottom)
        # Actually, usually we want:
        # [Tensor 1 Heatmap]
        # [Tensor 2 Heatmap]
        # ...
        # [State Strip A]
        # [State Strip B]
        
        n_tensors = len(target_tensors)
        # Height ratios: Heatmaps get 3 units, State strips get 0.2 units
        height_ratios = [3] * n_tensors + [0.2, 0.2]
        total_height = 4 * n_tensors + 2 # Adjust figure height based on number of tensors
        
        fig, axes = plt.subplots(n_tensors + 2, 1, figsize=(15, total_height), 
                                gridspec_kw={'height_ratios': height_ratios})
        
        # If only 1 tensor, axes is a list of 3. If multiple, it's length N+2.
        # Let's ensure axes is indexable
        if n_tensors + 2 == 1: axes = [axes]
        
        # Plot Heatmaps for each tensor
        for idx, t_name in enumerate(target_tensors):
            ax = axes[idx]
            
            # Collect similarity for this specific tensor and step pair
            # Note: collect_similarity_for_pairs expects a list of pairs
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
                
                # Pivot for heatmap
                pivot = (
                    df_tok.pivot_table(index="layer_idx", columns="token_idx", values="token_similarity", aggfunc="mean")
                    .sort_index()
                )
                
                im = ax.imshow(
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
                        ax.axvline(x=boundary-0.5, color='black', linestyle='-', linewidth=2, alpha=0.7)
                ax.axvline(x=prompt_len-0.5, color='purple', linestyle='-', linewidth=3, alpha=0.8)
                
                ax.set_title(f"{step_pair_str}: {t_name} Cosine Similarity")
                ax.set_ylabel("Layer Index")
                
                # Only show x-labels on the last heatmap
                if idx < n_tensors - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Token Index")

                if len(pivot.index) <= 20:
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_yticklabels(pivot.index)
                
                plt.colorbar(im, ax=ax, label="Sim")

        # Plot State Strips (shared for all tensors in this step pair)
        ax_state_a = axes[-2]
        ax_state_b = axes[-1]
        
        state_colors = {'prompt': 0, 'masked': 1, 'transferred': 2, 'generated': 3, 'unknown': 4}
        
        # State A
        state_array_a = [state_colors.get(state, 4) for state in token_states_a]
        ax_state_a.imshow([state_array_a], cmap=plt.cm.Set3, aspect='auto', interpolation='nearest', vmin=0, vmax=11)
        ax_state_a.set_title(f"Token States at Step {step_a} (Block {block_idx_a})")
        ax_state_a.set_yticks([])
        ax_state_a.set_xticks([]) 
        
        # State B
        state_array_b = [state_colors.get(state, 4) for state in token_states_b]
        ax_state_b.imshow([state_array_b], cmap=plt.cm.Set3, aspect='auto', interpolation='nearest', vmin=0, vmax=11)
        ax_state_b.set_title(f"Token States at Step {step_b} (Block {block_idx_b})")
        ax_state_b.set_xlabel("Token Index")
        ax_state_b.set_yticks([])
        
        # Legend        
        legend_elements = [
            Patch(facecolor=plt.cm.Set3(0), label='Prompt'),
            Patch(facecolor=plt.cm.Set3(1), label='Masked'),
            Patch(facecolor=plt.cm.Set3(2), label='Transferred'),
            Patch(facecolor=plt.cm.Set3(3), label='Generated')
        ]
        # Add legend
        ax_state_b.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        # ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        if show_fig:
            plt.show()
        if save_dir is not None:
            # Save plot to save_dir
            fig_path = os.path.join(save_dir, f"cosine_similarity_{step_pair_str.replace('→', '_to_')}.png")
            fig.savefig(fig_path)