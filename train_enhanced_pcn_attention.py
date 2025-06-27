"""
Comprehensive training script for enhanced PCN K-Q-V exploration experiments.

Compares:
1. Standard transformer (baseline)
2. Query-only PCN (previous best)
3. Full K-Q-V PCN (new)
4. Adaptive K-Q-V PCN (with scheduling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from models.pcn_attention_exploration import PCNGuidedMultiHeadAttention
from enhanced_pcn_kqv_exploration import EnhancedPCNKQVExploration, AdaptiveExplorationScheduler
from data_loader import load_data, get_batch


class EnhancedPCNAttention(nn.Module):
    """Multi-head attention with enhanced PCN K-Q-V exploration."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        # PCN parameters
        use_pcn_exploration: bool = True,
        n_exploration_samples: int = 5,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        explore_queries: bool = True,
        explore_keys: bool = False,
        explore_values: bool = False,
        use_adaptive_exploration: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_pcn_exploration = use_pcn_exploration
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # PCN exploration
        if use_pcn_exploration:
            self.pcn_explorer = EnhancedPCNKQVExploration(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_exploration_samples=n_exploration_samples,
                n_refinement_steps=n_refinement_steps,
                exploration_noise=exploration_noise,
                explore_queries=explore_queries,
                explore_keys=explore_keys,
                explore_values=explore_values,
                use_adaptive_exploration=use_adaptive_exploration
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_exploration_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        B, T, C = x.shape
        
        # Project to Q, K, V
        queries = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        exploration_stats = None
        
        # Apply PCN exploration
        if self.use_pcn_exploration:
            queries, keys, values, exploration_stats = self.pcn_explorer(
                queries, keys, values, x, mask, return_exploration_stats
            )
        
        # Standard attention computation
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        
        if return_exploration_stats and exploration_stats is not None:
            # Add attention statistics
            exploration_stats['attention_entropy'] = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean().item()
            exploration_stats['attention_max'] = attn_weights.max(dim=-1)[0].mean().item()
        
        return output, exploration_stats


def create_model(model_type: str, config: Dict) -> nn.Module:
    """Create model based on type."""
    
    base_params = {
        'vocab_size': config['vocab_size'],
        'block_size': config['block_size'],
        'n_embed': config['n_embed'],
        'n_heads': config['n_heads'],
        'n_layers': config['n_layers'],
        'dropout': config['dropout'],
        'device': config['device']
    }
    
    if model_type == 'standard':
        return TransformerWithPCNAttention(**base_params, use_pcn_exploration=False)
    
    elif model_type == 'query_only':
        return TransformerWithPCNAttention(
            **base_params,
            use_pcn_exploration=True,
            n_exploration_samples=config['n_exploration_samples'],
            n_refinement_steps=config['n_refinement_steps'],
            exploration_noise=config['exploration_noise'],
            explore_queries=True,
            explore_keys=False,
            explore_values=False
        )
    
    elif model_type == 'full_kqv':
        # Need to create custom transformer with enhanced attention
        # For now, we'll modify the existing one
        model = TransformerWithPCNAttention(
            **base_params,
            use_pcn_exploration=True,
            n_exploration_samples=config['n_exploration_samples'],
            n_refinement_steps=config['n_refinement_steps'],
            exploration_noise=config['exploration_noise'],
            explore_queries=True,
            explore_keys=True,
            explore_values=True
        )
        return model
    
    elif model_type == 'adaptive_kqv':
        model = TransformerWithPCNAttention(
            **base_params,
            use_pcn_exploration=True,
            n_exploration_samples=config['n_exploration_samples'],
            n_refinement_steps=config['n_refinement_steps'],
            exploration_noise=config['exploration_noise'],
            explore_queries=True,
            explore_keys=True,
            explore_values=True
        )
        # Enable adaptive exploration in PCN modules
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Dict,
    model_name: str
) -> Dict:
    """Train and evaluate model with comprehensive logging."""
    
    device = config['device']
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    history = {
        'model_name': model_name,
        'train_losses': [],
        'val_losses': [],
        'exploration_stats': {
            'query_energies': [],
            'key_energies': [],
            'value_energies': [],
            'query_diversity': [],
            'key_diversity': [],
            'value_diversity': [],
            'attention_entropy': [],
            'timestamps': []
        },
        'best_val_loss': float('inf'),
        'samples': []
    }
    
    start_time = time.time()
    
    print(f"\nTraining {model_name}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for step in range(config['max_steps']):
        # Training step
        model.train()
        xb, yb = get_batch('train', train_data, config['block_size'], 
                          config['batch_size'], device)
        
        # Forward pass with stats
        if hasattr(model, 'use_pcn_exploration') and model.use_pcn_exploration:
            logits, loss, stats = model(xb, yb, return_exploration_stats=True)
            
            # Log exploration statistics
            if stats:
                for key in ['query_energies', 'key_energies', 'value_energies',
                           'query_diversity', 'key_diversity', 'value_diversity']:
                    if key in stats:
                        if isinstance(stats[key], list):
                            history['exploration_stats'][key].append(np.mean(stats[key]))
                        else:
                            history['exploration_stats'][key].append(stats[key])
                
                if 'attention_entropy' in stats:
                    history['exploration_stats']['attention_entropy'].append(stats['attention_entropy'])
                
                history['exploration_stats']['timestamps'].append(time.time() - start_time)
        else:
            logits, loss, _ = model(xb, yb)
        
        # Optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        
        # Evaluation
        if step % config['eval_interval'] == 0:
            model.eval()
            
            # Compute validation loss
            val_losses = []
            for _ in range(config['eval_iters']):
                xb, yb = get_batch('val', val_data, config['block_size'],
                                  config['batch_size'], device)
                with torch.no_grad():
                    _, val_loss, _ = model(xb, yb)
                val_losses.append(val_loss.item())
            
            val_loss = np.mean(val_losses)
            train_loss = loss.item()
            
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            
            # Update best model
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_best.pt')
            
            # Progress report
            elapsed = time.time() - start_time
            print(f"  Step {step}/{config['max_steps']} ({elapsed:.1f}s): "
                  f"train {train_loss:.4f}, val {val_loss:.4f}")
            
            # Generate sample
            if step % config['sample_interval'] == 0 and step > 0:
                with torch.no_grad():
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated = model.generate(context, max_new_tokens=200,
                                             temperature=0.8, do_sample=True)
                    text = config['decode'](generated[0].cpu().numpy())
                    history['samples'].append({'step': step, 'text': text})
                    print(f"\n  Sample: {text[:100]}...\n")
    
    history['total_time'] = time.time() - start_time
    history['final_params'] = sum(p.numel() for p in model.parameters())
    
    return history


def analyze_results(histories: Dict[str, Dict], save_path: str = 'enhanced_kqv_analysis.png'):
    """Create comprehensive analysis plots."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Validation loss comparison
    ax1 = plt.subplot(3, 3, 1)
    for name, hist in histories.items():
        steps = np.arange(len(hist['val_losses'])) * 100  # eval_interval
        ax1.plot(steps, hist['val_losses'], label=name, linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Query exploration energy
    ax2 = plt.subplot(3, 3, 2)
    for name, hist in histories.items():
        if hist['exploration_stats']['query_energies']:
            ax2.plot(hist['exploration_stats']['query_energies'], 
                    label=name, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Exploration Steps')
    ax2.set_ylabel('Query Energy')
    ax2.set_title('Query Exploration Energy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Key exploration energy
    ax3 = plt.subplot(3, 3, 3)
    for name, hist in histories.items():
        if hist['exploration_stats']['key_energies']:
            ax3.plot(hist['exploration_stats']['key_energies'],
                    label=name, linewidth=2, alpha=0.7)
    ax3.set_xlabel('Exploration Steps')
    ax3.set_ylabel('Key Energy')
    ax3.set_title('Key Exploration Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Value exploration energy
    ax4 = plt.subplot(3, 3, 4)
    for name, hist in histories.items():
        if hist['exploration_stats']['value_energies']:
            ax4.plot(hist['exploration_stats']['value_energies'],
                    label=name, linewidth=2, alpha=0.7)
    ax4.set_xlabel('Exploration Steps')
    ax4.set_ylabel('Value Energy')
    ax4.set_title('Value Exploration Energy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Combined diversity metrics
    ax5 = plt.subplot(3, 3, 5)
    for name, hist in histories.items():
        diversities = []
        for key in ['query_diversity', 'key_diversity', 'value_diversity']:
            if hist['exploration_stats'][key]:
                diversities.append(hist['exploration_stats'][key])
        
        if diversities:
            combined_diversity = np.mean(diversities, axis=0)
            ax5.plot(combined_diversity, label=name, linewidth=2)
    
    ax5.set_xlabel('Exploration Steps')
    ax5.set_ylabel('Combined Diversity')
    ax5.set_title('K-Q-V Diversity Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Attention entropy
    ax6 = plt.subplot(3, 3, 6)
    for name, hist in histories.items():
        if hist['exploration_stats']['attention_entropy']:
            ax6.plot(hist['exploration_stats']['attention_entropy'],
                    label=name, linewidth=2, alpha=0.7)
    ax6.set_xlabel('Exploration Steps')
    ax6.set_ylabel('Attention Entropy')
    ax6.set_title('Attention Pattern Diversity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Training efficiency
    ax7 = plt.subplot(3, 3, 7)
    names = list(histories.keys())
    times = [hist['total_time'] / 60 for hist in histories.values()]
    final_losses = [hist['best_val_loss'] for hist in histories.values()]
    
    # Normalize times for visualization
    normalized_times = np.array(times) / times[0]  # Relative to baseline
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, final_losses, width, label='Best Val Loss')
    bars2 = ax7.bar(x + width/2, normalized_times, width, label='Relative Time')
    
    ax7.set_xlabel('Model')
    ax7.set_xticks(x)
    ax7.set_xticklabels(names, rotation=45)
    ax7.set_ylabel('Value')
    ax7.set_title('Performance vs Efficiency')
    ax7.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 8. Component contribution analysis
    ax8 = plt.subplot(3, 3, 8)
    # Calculate average energy reduction for each component
    component_contributions = {}
    
    for name, hist in histories.items():
        if 'Standard' not in name:
            contributions = []
            for component in ['query', 'key', 'value']:
                energies = hist['exploration_stats'][f'{component}_energies']
                if energies:
                    # Energy change from start to end
                    energy_change = energies[-1] - energies[0] if len(energies) > 1 else 0
                    contributions.append(energy_change)
            
            if contributions:
                component_contributions[name] = contributions
    
    if component_contributions:
        labels = ['Query', 'Key', 'Value']
        x = np.arange(len(labels))
        width = 0.25
        
        for i, (name, values) in enumerate(component_contributions.items()):
            offset = (i - len(component_contributions) / 2) * width
            ax8.bar(x + offset, values, width, label=name)
        
        ax8.set_xlabel('Component')
        ax8.set_ylabel('Energy Change')
        ax8.set_title('Component Contribution to Learning')
        ax8.set_xticks(x)
        ax8.set_xticklabels(labels)
        ax8.legend()
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
    
    for name, hist in histories.items():
        summary_text += f"{name}:\n"
        summary_text += f"  Best Val Loss: {hist['best_val_loss']:.4f}\n"
        summary_text += f"  Final Val Loss: {hist['val_losses'][-1]:.4f}\n"
        summary_text += f"  Training Time: {hist['total_time']/60:.1f} min\n"
        summary_text += f"  Parameters: {hist['final_params']:,}\n"
        
        # Add exploration statistics if available
        if any(hist['exploration_stats'][k] for k in ['query_diversity', 'key_diversity', 'value_diversity']):
            diversities = []
            for k in ['query_diversity', 'key_diversity', 'value_diversity']:
                if hist['exploration_stats'][k]:
                    diversities.append(hist['exploration_stats'][k][-1])
            if diversities:
                summary_text += f"  Final Diversity: {np.mean(diversities):.4f}\n"
        
        summary_text += "\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nAnalysis saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--n_embed', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # PCN exploration parameters
    parser.add_argument('--n_exploration_samples', type=int, default=5)
    parser.add_argument('--n_refinement_steps', type=int, default=5)
    parser.add_argument('--exploration_noise', type=float, default=0.15)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--sample_interval', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load data
    print("Loading TinyShakespeare dataset...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    config = vars(args)
    config.update({
        'device': device,
        'vocab_size': vocab_size,
        'encode': encode,
        'decode': decode
    })
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        if k not in ['encode', 'decode']:
            print(f"  {k}: {v}")
    
    # Models to train
    model_configs = [
        ('Standard', 'standard'),
        ('Query-Only PCN', 'query_only'),
        ('Full K-Q-V PCN', 'full_kqv'),
        # ('Adaptive K-Q-V PCN', 'adaptive_kqv')  # Uncomment for adaptive
    ]
    
    histories = {}
    
    # Train each model
    for model_name, model_type in model_configs:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print('='*60)
        
        model = create_model(model_type, config)
        history = train_model(model, train_data, val_data, config, model_name)
        histories[model_name] = history
        
        # Save individual history
        torch.save(history, f'{model_name.lower().replace(" ", "_")}_history.pt')
    
    # Analyze and compare results
    analyze_results(histories)
    
    # Save all histories
    torch.save(histories, 'enhanced_kqv_all_histories.pt')
    
    # Generate final comparison samples
    print("\n" + "="*60)
    print("Final Generation Comparison")
    print("="*60)
    
    prompt = "To be, or not to be, that is the question"
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for model_name, model_type in model_configs:
        model = create_model(model_type, config)
        # Load best weights
        try:
            model.load_state_dict(torch.load(f'{model_name.lower().replace(" ", "_")}_best.pt'))
            model.eval()
            
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=200,
                                         temperature=0.8, do_sample=True)
                text = decode(generated[0].cpu().numpy())
            
            print(f"\n{model_name}:")
            print(text)
        except:
            print(f"\n{model_name}: No saved model found")
    
    print(f"\n✓ Experiment complete!")
    print(f"  Results saved to enhanced_kqv_analysis.png")
    print(f"  Histories saved to enhanced_kqv_all_histories.pt")


if __name__ == '__main__':
    main()