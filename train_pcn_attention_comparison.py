"""
Shorter but complete training comparison of Standard vs PCN Attention Transformers
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from data_loader import load_data, get_batch


def train_model(model, train_data, val_data, config, model_name):
    """Train and evaluate model."""
    device = config['device']
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'pcn_energies': [],
        'pcn_diversity': []
    }
    
    start_time = time.time()
    
    print(f"\nTraining {model_name}...")
    
    for iter in range(config['max_iters']):
        # Training step
        model.train()
        xb, yb = get_batch('train', train_data, config['block_size'], 
                          config['batch_size'], device)
        
        # Forward pass
        if hasattr(model, 'use_pcn_exploration') and model.use_pcn_exploration:
            logits, loss, stats = model(xb, yb, return_exploration_stats=True)
            if stats and stats.get('query_energies'):
                history['pcn_energies'].append(np.mean(stats['query_energies']))
                history['pcn_diversity'].append(np.mean(stats['query_diversity']))
        else:
            logits, loss, _ = model(xb, yb, return_exploration_stats=False)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Evaluation
        if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
            model.eval()
            
            # Calculate validation loss
            val_losses = []
            for _ in range(config['eval_iters']):
                xb, yb = get_batch('val', val_data, config['block_size'],
                                  config['batch_size'], device)
                with torch.no_grad():
                    _, val_loss, _ = model(xb, yb, return_exploration_stats=False)
                val_losses.append(val_loss.item())
            
            val_loss = np.mean(val_losses)
            train_loss = loss.item()
            
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
            
            elapsed = time.time() - start_time
            print(f"  Iter {iter:4d}/{config['max_iters']} ({elapsed:5.1f}s): "
                  f"train {train_loss:.4f}, val {val_loss:.4f}")
    
    history['training_time'] = time.time() - start_time
    
    # Generate final sample
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=200, 
                                 temperature=0.8, do_sample=True)
        history['final_sample'] = config['decode'](generated[0].cpu().numpy())
    
    return history


def compare_and_visualize(histories, config):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Validation loss
    ax = axes[0, 0]
    for name, hist in histories.items():
        iters = np.arange(len(hist['val_losses'])) * config['eval_interval']
        ax.plot(iters, hist['val_losses'], label=name, linewidth=2, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Training loss
    ax = axes[0, 1]
    for name, hist in histories.items():
        iters = np.arange(len(hist['train_losses'])) * config['eval_interval']
        ax.plot(iters, hist['train_losses'], label=name, linewidth=2, marker='o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PCN Query Energy
    ax = axes[0, 2]
    for name, hist in histories.items():
        if hist['pcn_energies']:
            ax.plot(hist['pcn_energies'], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Query Energy')
    ax.set_title('PCN Query Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PCN Query Diversity
    ax = axes[1, 0]
    for name, hist in histories.items():
        if hist['pcn_diversity']:
            ax.plot(hist['pcn_diversity'], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Query Diversity')
    ax.set_title('PCN Query Diversity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Final metrics
    ax = axes[1, 1]
    models = list(histories.keys())
    metrics = {
        'Final Val Loss': [hist['val_losses'][-1] for hist in histories.values()],
        'Best Val Loss': [hist['best_val_loss'] for hist in histories.values()],
        'Time (min)': [hist['training_time']/60 for hist in histories.values()]
    }
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i*width - width, values, width, label=metric)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.legend()
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Training Summary\n" + "="*40 + "\n\n"
    for name, hist in histories.items():
        summary_text += f"{name}:\n"
        summary_text += f"  Final val loss: {hist['val_losses'][-1]:.4f}\n"
        summary_text += f"  Best val loss: {hist['best_val_loss']:.4f}\n"
        summary_text += f"  Perplexity: {np.exp(hist['val_losses'][-1]):.2f}\n"
        summary_text += f"  Time: {hist['training_time']/60:.1f} min\n"
        if hist['pcn_energies']:
            summary_text += f"  Final energy: {hist['pcn_energies'][-1]:.4f}\n"
            summary_text += f"  Final diversity: {hist['pcn_diversity'][-1]:.4f}\n"
        summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('pcn_attention_comparison.png', dpi=150)
    plt.close()


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64,
        'block_size': 256,
        'max_iters': 3000,  # ~10-15 minutes
        'eval_interval': 100,
        'eval_iters': 50,
        'learning_rate': 3e-4,
    }
    
    print("PCN Attention Comparison Training")
    print("="*50)
    print(f"Config: {config}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    config['encode'] = encode
    config['decode'] = decode
    
    # Create models
    print("\nCreating models...")
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'block_size': config['block_size'],
        'n_embed': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'device': config['device']
    }
    
    # Standard transformer
    standard_model = TransformerWithPCNAttention(
        **model_config,
        use_pcn_exploration=False
    )
    
    # PCN attention transformer
    pcn_model = TransformerWithPCNAttention(
        **model_config,
        use_pcn_exploration=True,
        n_exploration_samples=3,
        n_refinement_steps=3,
        exploration_noise=0.1,
        explore_queries=True
    )
    
    # Print model sizes
    def count_params(m):
        return sum(p.numel() for p in m.parameters())
    
    print(f"\nModel parameters:")
    print(f"  Standard: {count_params(standard_model):,}")
    print(f"  PCN Attention: {count_params(pcn_model):,}")
    print(f"  Overhead: {(count_params(pcn_model)/count_params(standard_model) - 1)*100:.1f}%")
    
    # Train models
    histories = {}
    
    print("\n" + "="*50)
    histories['Standard'] = train_model(
        standard_model, train_data, val_data, config, "Standard Transformer"
    )
    
    print("\n" + "="*50)
    histories['PCN-Attention'] = train_model(
        pcn_model, train_data, val_data, config, "PCN Attention Transformer"
    )
    
    # Generate comparison samples
    print("\n" + "="*50)
    print("Final Generation Samples:")
    print("="*50)
    
    for name, hist in histories.items():
        print(f"\n{name}:")
        print(hist['final_sample'][:300] + "...")
    
    # Create visualizations
    compare_and_visualize(histories, config)
    
    # Save results
    torch.save({
        'histories': histories,
        'config': config,
        'model_config': model_config
    }, 'pcn_attention_comparison_results.pt')
    
    print("\n✓ Training complete!")
    print("  Results saved to: pcn_attention_comparison.png")
    print("  Data saved to: pcn_attention_comparison_results.pt")


if __name__ == '__main__':
    main()