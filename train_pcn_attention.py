"""
Training script to compare standard transformer vs transformer with PCN-guided attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from models.bigram import BigramLM  # For baseline
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
        'exploration_stats': {
            'query_energies': [],
            'query_diversity': [],
            'attention_entropy': []
        },
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
        if hasattr(model, 'use_pcn_exploration'):
            logits, loss, stats = model(xb, yb, return_exploration_stats=True)
            if stats and stats.get('query_energies'):
                history['exploration_stats']['query_energies'].append(
                    np.mean(stats['query_energies'])
                )
                history['exploration_stats']['query_diversity'].append(
                    np.mean(stats['query_diversity']) if stats['query_diversity'] else 0
                )
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
                    if hasattr(model, 'use_pcn_exploration'):
                        logits, loss, _ = model(xb, yb, return_exploration_stats=False)
                    else:
                        logits, loss = model(xb, yb)
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


def analyze_attention_exploration(histories: Dict[str, Dict]):
    """
    Analyze PCN attention exploration statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Query energies over training
    ax = axes[0, 0]
    for name, history in histories.items():
        if 'PCN' in name and history['exploration_stats']['query_energies']:
            iters = np.arange(len(history['exploration_stats']['query_energies']))
            ax.plot(iters, history['exploration_stats']['query_energies'], 
                   label=name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Query Energy')
    ax.set_title('Query Exploration Energy During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Query diversity
    ax = axes[0, 1]
    for name, history in histories.items():
        if 'PCN' in name and history['exploration_stats']['query_diversity']:
            iters = np.arange(len(history['exploration_stats']['query_diversity']))
            ax.plot(iters, history['exploration_stats']['query_diversity'],
                   label=name, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Query Diversity')
    ax.set_title('Query Diversity During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation loss comparison
    ax = axes[1, 0]
    for name, history in histories.items():
        iters = np.arange(0, len(history['val_losses']) * 50, 50)  # Assuming eval_interval=50
        ax.plot(iters, history['val_losses'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics
    ax = axes[1, 1]
    names = list(histories.keys())
    final_losses = [h['val_losses'][-1] for h in histories.values()]
    final_perplexities = [np.exp(loss) for loss in final_losses]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_losses, width, label='Val Loss')
    bars2 = ax.bar(x + width/2, final_perplexities, width, label='Perplexity')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    ax.set_title('Final Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('pcn_attention_analysis.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--n_embed', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # PCN exploration parameters
    parser.add_argument('--n_exploration_samples', type=int, default=3)
    parser.add_argument('--n_refinement_steps', type=int, default=5)
    parser.add_argument('--exploration_noise', type=float, default=0.1)
    parser.add_argument('--explore_queries', action='store_true', default=True)
    parser.add_argument('--explore_keys', action='store_true', default=False)
    parser.add_argument('--explore_values', action='store_true', default=False)
    parser.add_argument('--pcn_layers', type=int, nargs='*', default=None,
                       help='Which layers to apply PCN to (default: all)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
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
    standard_transformer = TransformerWithPCNAttention(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_pcn_exploration=False,
        device=device
    )
    
    # 2. Transformer with PCN-guided attention (all layers)
    pcn_transformer_all = TransformerWithPCNAttention(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_pcn_exploration=True,
        pcn_layers=None,  # Apply to all layers
        n_exploration_samples=args.n_exploration_samples,
        n_refinement_steps=args.n_refinement_steps,
        exploration_noise=args.exploration_noise,
        explore_queries=args.explore_queries,
        explore_keys=args.explore_keys,
        explore_values=args.explore_values,
        device=device
    )
    
    # 3. Transformer with PCN-guided attention (first layer only)
    pcn_transformer_first = TransformerWithPCNAttention(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_pcn_exploration=True,
        pcn_layers=[0],  # Apply to first layer only
        n_exploration_samples=args.n_exploration_samples,
        n_refinement_steps=args.n_refinement_steps,
        exploration_noise=args.exploration_noise,
        explore_queries=args.explore_queries,
        explore_keys=args.explore_keys,
        explore_values=args.explore_values,
        device=device
    )
    
    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\nModel sizes:")
    print(f"  Standard Transformer: {count_params(standard_transformer):,} parameters")
    print(f"  PCN Attention (all layers): {count_params(pcn_transformer_all):,} parameters")
    print(f"  PCN Attention (first layer): {count_params(pcn_transformer_first):,} parameters")
    
    # Train models
    histories = {}
    
    print("\n" + "="*60)
    print("TRAINING STANDARD TRANSFORMER")
    print("="*60)
    histories['Standard'] = train_model(
        standard_transformer, train_data, val_data, config, "Standard"
    )
    
    print("\n" + "="*60)
    print("TRAINING PCN ATTENTION TRANSFORMER (ALL LAYERS)")
    print("="*60)
    histories['PCN-Attention-All'] = train_model(
        pcn_transformer_all, train_data, val_data, config, "PCN-All"
    )
    
    print("\n" + "="*60)
    print("TRAINING PCN ATTENTION TRANSFORMER (FIRST LAYER)")
    print("="*60)
    histories['PCN-Attention-First'] = train_model(
        pcn_transformer_first, train_data, val_data, config, "PCN-First"
    )
    
    # Analyze results
    analyze_attention_exploration(histories)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for name, history in histories.items():
        print(f"\n{name}:")
        print(f"  Final validation loss: {history['val_losses'][-1]:.4f}")
        print(f"  Final perplexity: {np.exp(history['val_losses'][-1]):.2f}")
        print(f"  Training time: {history['training_time']/60:.1f} minutes")
        
        if 'PCN' in name and history['exploration_stats']['query_energies']:
            print(f"  Mean query energy: {np.mean(history['exploration_stats']['query_energies']):.4f}")
            print(f"  Mean query diversity: {np.mean(history['exploration_stats']['query_diversity']):.4f}")
    
    # Save models and results
    torch.save(standard_transformer.state_dict(), 'standard_transformer_attention.pt')
    torch.save(pcn_transformer_all.state_dict(), 'pcn_transformer_attention_all.pt')
    torch.save(pcn_transformer_first.state_dict(), 'pcn_transformer_attention_first.pt')
    torch.save(histories, 'pcn_attention_histories.pt')
    
    print("\n✓ Training complete! Results saved.")
    print("  - Analysis plot: pcn_attention_analysis.png")
    print("  - Model weights: *_attention.pt")
    print("  - Training histories: pcn_attention_histories.pt")


if __name__ == '__main__':
    main()