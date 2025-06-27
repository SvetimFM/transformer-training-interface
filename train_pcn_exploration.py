"""
Training script to compare standard transformer vs transformer with PCN exploration.

This script trains both models on TinyShakespeare and compares:
1. Validation loss / perplexity
2. Generation quality
3. Computational overhead
4. Exploration statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
import numpy as np
import time
import os
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append('src')

from models.transformer_with_pcn_exploration import TransformerWithPCNExploration
from models.bigram import BigramLM
from data_loader import load_data, get_batch


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Dict,
    model_name: str = "model"
) -> Dict:
    """
    Train a model and return training history.
    """
    device = config['device']
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'exploration_stats': [],
        'generation_samples': [],
        'training_time': 0
    }
    
    # Training loop
    start_time = time.time()
    
    for iter in range(config['max_iters']):
        # Training step
        model.train()
        xb, yb = get_batch('train', train_data, config['block_size'], 
                          config['batch_size'], device)
        
        # Forward pass
        if hasattr(model, 'use_pcn_exploration') and model.use_pcn_exploration:
            logits, loss, stats = model(xb, yb, return_exploration_stats=True)
            history['exploration_stats'].append(stats)
        else:
            logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Evaluation
        if iter % config['eval_interval'] == 0:
            model.eval()
            
            # Validation loss
            val_losses = []
            for _ in range(config['eval_iters']):
                xb, yb = get_batch('val', val_data, config['block_size'],
                                  config['batch_size'], device)
                with torch.no_grad():
                    logits, loss = model(xb, yb)[:2]  # Only take first two returns
                val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            train_loss = loss.item()
            
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            
            print(f"[{model_name}] Iter {iter}/{config['max_iters']}: "
                  f"train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                  f"perplexity {np.exp(val_loss):.2f}")
            
            # Generate sample
            if iter % config['sample_interval'] == 0:
                sample = generate_sample(model, config['encode'], config['decode'],
                                       device, max_length=200)
                history['generation_samples'].append({
                    'iter': iter,
                    'text': sample
                })
                print(f"\n[{model_name}] Sample:\n{sample}\n")
    
    history['training_time'] = time.time() - start_time
    return history


def generate_sample(model, encode, decode, device, prompt="O ", max_length=100):
    """Generate a text sample from the model."""
    model.eval()
    
    # Encode prompt
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        if hasattr(model, 'generate'):
            generated = model.generate(context, max_new_tokens=max_length,
                                     temperature=0.8, do_sample=True)
        else:
            # Fallback for models without generate method
            generated = context
            for _ in range(max_length):
                logits, _ = model(generated)
                logits = logits[:, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
    
    return decode(generated[0].cpu().numpy())


def compare_models(histories: Dict[str, Dict], config: Dict):
    """
    Create comparison plots and analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Validation loss
    ax = axes[0, 0]
    for name, history in histories.items():
        iters = np.arange(0, config['max_iters'], config['eval_interval'])
        ax.plot(iters, history['val_losses'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    ax = axes[0, 1]
    for name, history in histories.items():
        iters = np.arange(0, config['max_iters'], config['eval_interval'])
        perplexity = np.exp(history['val_losses'])
        ax.plot(iters, perplexity, label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training time
    ax = axes[1, 0]
    names = list(histories.keys())
    times = [h['training_time'] / 60 for h in histories.values()]  # Convert to minutes
    bars = ax.bar(names, times, color=['blue', 'orange', 'green'][:len(names)])
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}m', ha='center', va='bottom')
    
    # Plot 4: Final metrics comparison
    ax = axes[1, 1]
    metrics = {
        'Final Val Loss': [h['val_losses'][-1] for h in histories.values()],
        'Final Perplexity': [np.exp(h['val_losses'][-1]) for h in histories.values()],
    }
    
    # Create grouped bar chart
    x = np.arange(len(names))
    width = 0.35
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('pcn_exploration_comparison.png', dpi=150)
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"  Final validation loss: {history['val_losses'][-1]:.4f}")
        print(f"  Final perplexity: {np.exp(history['val_losses'][-1]):.2f}")
        print(f"  Training time: {history['training_time']/60:.1f} minutes")
        
        # Exploration statistics if available
        if history['exploration_stats'] and len(history['exploration_stats']) > 0:
            # Average over last 10% of training
            recent_stats = history['exploration_stats'][-len(history['exploration_stats'])//10:]
            if recent_stats and recent_stats[0]:
                avg_energy = np.mean([s.get('energies', [0])[0] if s.get('energies') else 0 
                                     for s in recent_stats])
                avg_diversity = np.mean([s.get('diversity', [0])[0] if s.get('diversity') else 0 
                                        for s in recent_stats])
                print(f"  Average exploration energy: {avg_energy:.4f}")
                print(f"  Average exploration diversity: {avg_diversity:.4f}")


def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # PCN exploration parameters
    parser.add_argument('--n_exploration_samples', type=int, default=5)
    parser.add_argument('--n_exploration_steps', type=int, default=10)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--exploration_points', type=str, default='final',
                       choices=['final', 'all', 'middle'])
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=20)
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
    
    # Config
    config = {
        'device': device,
        'batch_size': args.batch_size,
        'block_size': args.block_size,
        'max_iters': args.max_iters,
        'eval_interval': args.eval_interval,
        'eval_iters': args.eval_iters,
        'sample_interval': args.sample_interval,
        'learning_rate': args.learning_rate,
        'grad_clip': args.grad_clip,
        'encode': encode,
        'decode': decode
    }
    
    # Create models
    print("\nCreating models...")
    
    # 1. Standard Transformer
    standard_transformer = TransformerWithPCNExploration(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_pcn_exploration=False,
        device=device
    )
    
    # 2. Transformer with PCN Exploration
    pcn_transformer = TransformerWithPCNExploration(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_pcn_exploration=True,
        exploration_points=args.exploration_points,
        n_exploration_samples=args.n_exploration_samples,
        n_exploration_steps=args.n_exploration_steps,
        exploration_noise=args.exploration_noise,
        device=device
    )
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard_transformer.parameters())
    pcn_params = sum(p.numel() for p in pcn_transformer.parameters())
    
    print(f"\nModel sizes:")
    print(f"  Standard Transformer: {standard_params:,} parameters")
    print(f"  PCN Exploration Transformer: {pcn_params:,} parameters")
    print(f"  Overhead: {(pcn_params - standard_params):,} parameters "
          f"({(pcn_params/standard_params - 1)*100:.1f}%)")
    
    # Train models
    histories = {}
    
    print("\n" + "="*60)
    print("TRAINING STANDARD TRANSFORMER")
    print("="*60)
    histories['Standard'] = train_model(
        standard_transformer, train_data, val_data, config, "Standard"
    )
    
    print("\n" + "="*60)
    print("TRAINING PCN EXPLORATION TRANSFORMER")
    print("="*60)
    histories['PCN-Exploration'] = train_model(
        pcn_transformer, train_data, val_data, config, "PCN-Exploration"
    )
    
    # Compare results
    compare_models(histories, config)
    
    # Save final models
    torch.save(standard_transformer.state_dict(), 'standard_transformer.pt')
    torch.save(pcn_transformer.state_dict(), 'pcn_exploration_transformer.pt')
    
    # Save histories
    torch.save(histories, 'training_histories.pt')
    
    print("\n✓ Training complete! Results saved.")
    print("  - Comparison plot: pcn_exploration_comparison.png")
    print("  - Model weights: standard_transformer.pt, pcn_exploration_transformer.pt")
    print("  - Training histories: training_histories.pt")


if __name__ == '__main__':
    main()