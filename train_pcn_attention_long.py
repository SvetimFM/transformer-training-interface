"""
Extended training comparison: Standard Transformer vs PCN-Guided Attention Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from data_loader import load_data, get_batch


def train_and_evaluate(model, train_data, val_data, config, model_name):
    """Train model and track metrics."""
    device = config['device']
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'timestamps': [],
        'exploration_stats': {'energies': [], 'diversity': []}
    }
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for iter in range(config['max_iters']):
        # Training
        model.train()
        xb, yb = get_batch('train', train_data, config['block_size'], 
                          config['batch_size'], device)
        
        if hasattr(model, 'use_pcn_exploration') and model.use_pcn_exploration:
            logits, loss, stats = model(xb, yb, return_exploration_stats=True)
            if stats and stats.get('query_energies'):
                history['exploration_stats']['energies'].append(np.mean(stats['query_energies']))
                history['exploration_stats']['diversity'].append(np.mean(stats['query_diversity']))
        else:
            logits, loss, _ = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        
        # Evaluation
        if iter % config['eval_interval'] == 0:
            model.eval()
            val_losses = []
            
            for _ in range(config['eval_iters']):
                xb, yb = get_batch('val', val_data, config['block_size'],
                                  config['batch_size'], device)
                with torch.no_grad():
                    _, loss, _ = model(xb, yb, return_exploration_stats=False)
                val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            train_loss = loss.item()
            
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['timestamps'].append(time.time() - start_time)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_name}_best.pt')
            
            # Progress update
            elapsed = time.time() - start_time
            print(f"[{model_name}] Iter {iter}/{config['max_iters']} "
                  f"({elapsed/60:.1f}m): train {train_loss:.4f}, "
                  f"val {val_loss:.4f}, best {best_val_loss:.4f}")
            
            # Generate sample every 1000 iterations
            if iter % 1000 == 0 and iter > 0:
                with torch.no_grad():
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    sample = model.generate(context, max_new_tokens=200, 
                                          temperature=0.8, do_sample=True)
                    text = config['decode'](sample[0].cpu().numpy())
                print(f"\n[{model_name}] Sample:\n{text}\n")
    
    history['total_time'] = time.time() - start_time
    history['best_val_loss'] = best_val_loss
    return history


def plot_results(histories, save_prefix):
    """Create comprehensive result plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Validation loss over time
    ax = axes[0, 0]
    for name, hist in histories.items():
        times = np.array(hist['timestamps']) / 60  # Convert to minutes
        ax.plot(times, hist['val_losses'], label=name, linewidth=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Training loss
    ax = axes[0, 1]
    for name, hist in histories.items():
        ax.plot(hist['train_losses'], label=name, linewidth=2)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Learning curves
    ax = axes[0, 2]
    for name, hist in histories.items():
        ax.plot(hist['val_losses'], label=f'{name} (val)', linewidth=2)
        ax.plot(hist['train_losses'], label=f'{name} (train)', 
                linewidth=2, linestyle='--', alpha=0.7)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. PCN Energy evolution (if available)
    ax = axes[1, 0]
    for name, hist in histories.items():
        if hist['exploration_stats']['energies']:
            ax.plot(hist['exploration_stats']['energies'], label=name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Query Energy')
    ax.set_title('PCN Query Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. PCN Diversity evolution
    ax = axes[1, 1]
    for name, hist in histories.items():
        if hist['exploration_stats']['diversity']:
            ax.plot(hist['exploration_stats']['diversity'], label=name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Query Diversity')
    ax.set_title('PCN Query Diversity Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Final metrics comparison
    ax = axes[1, 2]
    metrics_data = []
    labels = []
    
    for name, hist in histories.items():
        final_val = hist['val_losses'][-1]
        best_val = hist['best_val_loss']
        time_mins = hist['total_time'] / 60
        
        metrics_data.append([final_val, best_val, time_mins/10])  # Scale time for visibility
        labels.append(name)
    
    metrics_data = np.array(metrics_data).T
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics_data[0], width, label='Final Val Loss')
    bars2 = ax.bar(x, metrics_data[1], width, label='Best Val Loss')
    bars3 = ax.bar(x + width, metrics_data[2], width, label='Time (×10 min)')
    
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_results.png', dpi=150)
    plt.close()


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 64,
        'block_size': 256,
        'max_iters': 10000,  # ~20-30 minutes
        'eval_interval': 100,
        'eval_iters': 200,
        'learning_rate': 3e-4,
        'grad_clip': 1.0,
    }
    
    print(f"Starting extended training run at {datetime.now()}")
    print(f"Configuration: {config}")
    
    # Load data
    print("\nLoading TinyShakespeare dataset...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    config['encode'] = encode
    config['decode'] = decode
    
    print(f"Dataset: {len(train_data):,} train, {len(val_data):,} val tokens")
    print(f"Vocab size: {vocab_size}")
    
    # Create models
    print("\nCreating models...")
    
    # Standard Transformer
    standard_model = TransformerWithPCNAttention(
        vocab_size=vocab_size,
        block_size=config['block_size'],
        n_embed=384,
        n_heads=6,
        n_layers=6,
        dropout=0.2,
        use_pcn_exploration=False,
        device=config['device']
    )
    
    # PCN-Guided Attention Transformer
    pcn_model = TransformerWithPCNAttention(
        vocab_size=vocab_size,
        block_size=config['block_size'],
        n_embed=384,
        n_heads=6,
        n_layers=6,
        dropout=0.2,
        use_pcn_exploration=True,
        pcn_layers=[0, 2, 4],  # Apply to alternating layers
        n_exploration_samples=3,
        n_refinement_steps=3,
        exploration_noise=0.1,
        explore_queries=True,
        device=config['device']
    )
    
    # Model sizes
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    standard_params = count_params(standard_model)
    pcn_params = count_params(pcn_model)
    
    print(f"\nModel parameters:")
    print(f"  Standard Transformer: {standard_params:,}")
    print(f"  PCN Attention Transformer: {pcn_params:,}")
    print(f"  Overhead: {pcn_params - standard_params:,} ({(pcn_params/standard_params - 1)*100:.1f}%)")
    
    # Train models
    histories = {}
    
    print("\n" + "="*60)
    print("TRAINING STANDARD TRANSFORMER")
    print("="*60)
    histories['Standard'] = train_and_evaluate(
        standard_model, train_data, val_data, config, "standard"
    )
    
    print("\n" + "="*60)
    print("TRAINING PCN ATTENTION TRANSFORMER")
    print("="*60)
    histories['PCN-Attention'] = train_and_evaluate(
        pcn_model, train_data, val_data, config, "pcn_attention"
    )
    
    # Generate final samples
    print("\n" + "="*60)
    print("FINAL GENERATION COMPARISON")
    print("="*60)
    
    prompt = "ROMEO:\nSpeak, what is in thy heart"
    context = torch.tensor(encode(prompt), dtype=torch.long, 
                          device=config['device']).unsqueeze(0)
    
    for name, model in [("Standard", standard_model), ("PCN-Attention", pcn_model)]:
        model.eval()
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=200,
                                     temperature=0.8, do_sample=True)
            text = decode(generated[0].cpu().numpy())
        print(f"\n{name}:\n{text}")
    
    # Plot results
    plot_results(histories, 'pcn_attention_long')
    
    # Save histories
    torch.save(histories, 'pcn_attention_long_histories.pt')
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for name, hist in histories.items():
        print(f"\n{name}:")
        print(f"  Final validation loss: {hist['val_losses'][-1]:.4f}")
        print(f"  Best validation loss: {hist['best_val_loss']:.4f}")
        print(f"  Training time: {hist['total_time']/60:.1f} minutes")
        print(f"  Final perplexity: {np.exp(hist['val_losses'][-1]):.2f}")
        
        if hist['exploration_stats']['energies']:
            print(f"  Final query energy: {hist['exploration_stats']['energies'][-1]:.4f}")
            print(f"  Final query diversity: {hist['exploration_stats']['diversity'][-1]:.4f}")
    
    print(f"\nTraining completed at {datetime.now()}")
    print("\nResults saved:")
    print("  - Plot: pcn_attention_long_results.png")
    print("  - Histories: pcn_attention_long_histories.pt")
    print("  - Best models: standard_best.pt, pcn_attention_best.pt")


if __name__ == '__main__':
    main()