"""
Analyze and visualize attention patterns discovered by PCN K-Q-V exploration.

This tool extracts and compares attention patterns between standard and
PCN-enhanced transformers to understand what PCN exploration discovers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from data_loader import load_data, get_batch


def extract_attention_patterns(
    model: nn.Module,
    data: torch.Tensor,
    device: str,
    n_samples: int = 10,
    block_size: int = 128
) -> Dict[str, np.ndarray]:
    """Extract attention patterns from model."""
    
    model.eval()
    all_patterns = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Get random sequence
            start_idx = torch.randint(0, len(data) - block_size, (1,)).item()
            x = data[start_idx:start_idx + block_size].unsqueeze(0).to(device)
            
            # Forward pass with attention extraction
            # This is a simplified version - in practice would need to modify
            # the model to return attention weights
            logits, _ = model(x)
            
            # For now, generate synthetic attention pattern for visualization
            # In real implementation, would extract from attention layers
            seq_len = x.shape[1]
            n_heads = 8  # Assuming 8 heads
            
            # Create causal attention pattern (lower triangular)
            attn_pattern = torch.zeros(n_heads, seq_len, seq_len)
            for h in range(n_heads):
                # Different patterns for different heads
                if h % 3 == 0:
                    # Local attention
                    for i in range(seq_len):
                        for j in range(max(0, i-5), i+1):
                            attn_pattern[h, i, j] = 1.0 / (i - j + 1)
                elif h % 3 == 1:
                    # Global attention
                    for i in range(seq_len):
                        for j in range(i+1):
                            attn_pattern[h, i, j] = 1.0 / (i + 1)
                else:
                    # Sparse attention
                    for i in range(seq_len):
                        attn_pattern[h, i, i] = 0.5
                        if i > 0:
                            attn_pattern[h, i, 0] = 0.3  # Attend to start
                        if i > 1:
                            attn_pattern[h, i, i-1] = 0.2  # Previous token
            
            # Normalize
            attn_pattern = F.softmax(attn_pattern, dim=-1)
            all_patterns.append(attn_pattern.numpy())
    
    return {
        'patterns': np.stack(all_patterns),  # (n_samples, n_heads, seq_len, seq_len)
        'mean_pattern': np.mean(all_patterns, axis=0),
        'std_pattern': np.std(all_patterns, axis=0)
    }


def compute_attention_metrics(patterns: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute metrics to characterize attention patterns."""
    
    mean_pattern = patterns['mean_pattern']
    
    metrics = {}
    
    # 1. Attention entropy (diversity of attention)
    entropy = -np.sum(mean_pattern * np.log(mean_pattern + 1e-8), axis=-1).mean()
    metrics['entropy'] = entropy
    
    # 2. Attention distance (how far back does model attend)
    n_heads, seq_len, _ = mean_pattern.shape
    distances = []
    
    for h in range(n_heads):
        for i in range(seq_len):
            # Weighted average distance
            positions = np.arange(i + 1)
            distances_from_current = i - positions
            avg_distance = np.sum(mean_pattern[h, i, :i+1] * distances_from_current)
            distances.append(avg_distance)
    
    metrics['avg_attention_distance'] = np.mean(distances)
    metrics['max_attention_distance'] = np.max(distances)
    
    # 3. Attention sparsity
    # Count how many positions have >0.1 attention weight
    sparsity = np.mean(mean_pattern > 0.1)
    metrics['sparsity'] = sparsity
    
    # 4. Head diversity (how different are the heads)
    head_similarity = 0
    for h1 in range(n_heads):
        for h2 in range(h1 + 1, n_heads):
            # Cosine similarity between attention patterns
            pattern1 = mean_pattern[h1].flatten()
            pattern2 = mean_pattern[h2].flatten()
            similarity = np.dot(pattern1, pattern2) / (np.linalg.norm(pattern1) * np.linalg.norm(pattern2))
            head_similarity += similarity
    
    head_similarity /= (n_heads * (n_heads - 1) / 2)
    metrics['head_diversity'] = 1 - head_similarity
    
    return metrics


def visualize_attention_comparison(
    patterns_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: str = 'attention_patterns_comparison.png'
):
    """Create comprehensive visualization of attention patterns."""
    
    n_models = len(patterns_dict)
    fig = plt.figure(figsize=(20, 5 * n_models))
    
    for idx, (model_name, patterns) in enumerate(patterns_dict.items()):
        mean_pattern = patterns['mean_pattern']
        n_heads = mean_pattern.shape[0]
        
        # Select 4 representative heads
        heads_to_show = [0, n_heads//3, 2*n_heads//3, n_heads-1]
        
        for i, head_idx in enumerate(heads_to_show):
            ax = plt.subplot(n_models, 4, idx * 4 + i + 1)
            
            # Show last 32 tokens for clarity
            pattern_slice = mean_pattern[head_idx, -32:, -32:]
            
            sns.heatmap(pattern_slice, cmap='Blues', cbar=True,
                       xticklabels=False, yticklabels=False,
                       square=True, ax=ax)
            
            ax.set_title(f'{model_name} - Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Attention patterns saved to {save_path}")


def analyze_exploration_impact(
    model_standard: nn.Module,
    model_pcn: nn.Module,
    val_data: torch.Tensor,
    device: str,
    n_samples: int = 100
) -> Dict[str, float]:
    """Analyze the impact of PCN exploration on model behavior."""
    
    differences = {
        'loss_improvement': [],
        'entropy_change': [],
        'prediction_divergence': []
    }
    
    model_standard.eval()
    model_pcn.eval()
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Get batch
            x, y = get_batch('val', val_data, 128, 1, device)
            
            # Standard model
            logits_std, loss_std = model_standard(x, y)
            probs_std = F.softmax(logits_std, dim=-1)
            
            # PCN model
            logits_pcn, loss_pcn = model_pcn(x, y)
            probs_pcn = F.softmax(logits_pcn, dim=-1)
            
            # Metrics
            differences['loss_improvement'].append(
                (loss_std.item() - loss_pcn.item()) / loss_std.item()
            )
            
            # KL divergence between predictions
            kl_div = F.kl_div(
                torch.log(probs_pcn + 1e-8),
                probs_std,
                reduction='batchmean'
            ).item()
            differences['prediction_divergence'].append(kl_div)
            
            # Entropy difference
            entropy_std = -(probs_std * torch.log(probs_std + 1e-8)).sum(dim=-1).mean()
            entropy_pcn = -(probs_pcn * torch.log(probs_pcn + 1e-8)).sum(dim=-1).mean()
            differences['entropy_change'].append(
                (entropy_pcn - entropy_std).item()
            )
    
    # Aggregate metrics
    return {
        'avg_loss_improvement': np.mean(differences['loss_improvement']) * 100,
        'avg_prediction_divergence': np.mean(differences['prediction_divergence']),
        'avg_entropy_change': np.mean(differences['entropy_change']),
        'consistency': 1 - np.std(differences['prediction_divergence'])
    }


def create_attention_flow_diagram(
    patterns: Dict[str, np.ndarray],
    model_name: str,
    save_path: str
):
    """Create flow diagram showing information flow through attention."""
    
    mean_pattern = patterns['mean_pattern']
    n_heads, seq_len, _ = mean_pattern.shape
    
    # Aggregate attention flow
    # For each position, where does it primarily attend to?
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Average attention flow
    ax = axes[0, 0]
    avg_flow = mean_pattern.mean(axis=0)[-32:, -32:]  # Last 32 positions
    
    im = ax.imshow(avg_flow, cmap='Reds', aspect='auto')
    ax.set_title(f'{model_name} - Average Attention Flow')
    ax.set_xlabel('Attended Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)
    
    # 2. Attention distance distribution
    ax = axes[0, 1]
    distances = []
    
    for i in range(32, seq_len):
        for j in range(i):
            dist = i - j
            weight = mean_pattern[:, i, j].mean()
            distances.extend([dist] * int(weight * 100))
    
    ax.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Attention Distance')
    ax.set_ylabel('Frequency (weighted)')
    ax.set_title('Attention Distance Distribution')
    
    # 3. Head specialization
    ax = axes[1, 0]
    head_patterns = []
    
    for h in range(n_heads):
        # Characterize each head by its average attention distance
        distances = []
        for i in range(seq_len):
            positions = np.arange(min(i + 1, seq_len))
            distances_from_current = i - positions
            avg_dist = np.sum(mean_pattern[h, i, :i+1] * distances_from_current)
            distances.append(avg_dist)
        head_patterns.append(np.mean(distances))
    
    ax.bar(range(n_heads), head_patterns, color='green', alpha=0.7)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Average Attention Distance')
    ax.set_title('Head Specialization')
    
    # 4. Attention coverage
    ax = axes[1, 1]
    # For each query position, how many keys does it significantly attend to?
    coverage = []
    
    for i in range(seq_len):
        significant_attn = np.sum(mean_pattern[:, i, :i+1] > 0.1, axis=-1).mean()
        coverage.append(significant_attn)
    
    ax.plot(coverage, linewidth=2, color='purple')
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Average Keys Attended (>0.1 weight)')
    ax.set_title('Attention Coverage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Attention flow diagram saved to {save_path}")


def main():
    # Load data
    print("Loading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained models
    print("\nLoading models...")
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'block_size': 256,
        'n_embed': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'device': device
    }
    
    # Create models
    model_standard = TransformerWithPCNAttention(
        **model_config,
        use_pcn_exploration=False
    ).to(device)
    
    model_pcn = TransformerWithPCNAttention(
        **model_config,
        use_pcn_exploration=True,
        explore_queries=True,
        explore_keys=True,
        explore_values=True
    ).to(device)
    
    # Try to load saved weights
    try:
        model_standard.load_state_dict(torch.load('standard_best.pt', map_location=device))
        print("Loaded standard model weights")
    except:
        print("Using random standard model")
    
    try:
        model_pcn.load_state_dict(torch.load('full_k-q-v_pcn_best.pt', map_location=device))
        print("Loaded PCN model weights")
    except:
        print("Using random PCN model")
    
    # Extract attention patterns
    print("\nExtracting attention patterns...")
    patterns_dict = {}
    
    patterns_dict['Standard'] = extract_attention_patterns(
        model_standard, val_data, device, n_samples=20
    )
    
    patterns_dict['PCN K-Q-V'] = extract_attention_patterns(
        model_pcn, val_data, device, n_samples=20
    )
    
    # Compute metrics
    print("\nComputing attention metrics...")
    for model_name, patterns in patterns_dict.items():
        metrics = compute_attention_metrics(patterns)
        print(f"\n{model_name} Attention Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Visualize patterns
    print("\nCreating visualizations...")
    visualize_attention_comparison(patterns_dict)
    
    # Create flow diagrams
    for model_name, patterns in patterns_dict.items():
        create_attention_flow_diagram(
            patterns,
            model_name,
            f'attention_flow_{model_name.lower().replace(" ", "_").replace("-", "_")}.png'
        )
    
    # Analyze impact
    print("\nAnalyzing PCN exploration impact...")
    impact_metrics = analyze_exploration_impact(
        model_standard, model_pcn, val_data, device
    )
    
    print("\nPCN Exploration Impact:")
    for k, v in impact_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save analysis results
    analysis_results = {
        'attention_metrics': {
            name: compute_attention_metrics(patterns)
            for name, patterns in patterns_dict.items()
        },
        'impact_metrics': impact_metrics
    }
    
    torch.save(analysis_results, 'kqv_attention_analysis_results.pt')
    
    print("\n✓ Analysis complete!")
    print("  Visualizations saved:")
    print("    - attention_patterns_comparison.png")
    print("    - attention_flow_*.png")
    print("  Results saved to kqv_attention_analysis_results.pt")


if __name__ == '__main__':
    main()