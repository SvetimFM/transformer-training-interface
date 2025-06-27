"""
Energy Guidance Experiment V2

Tests the improved PCN approach that:
1. Uses separate latent space (not token embeddings)
2. Guides attention through energy-based biases
3. Refines token distributions at output layer
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('src')

from models.energy_guided_transformer import (
    EnergyGuidedTransformer,
    LightweightEnergyTransformer
)
from models.embedding_matching_transformer import StandardTransformer
from data_loader import load_data, get_batch


def train_and_evaluate(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: dict,
    model_name: str
) -> dict:
    """Train model and collect statistics."""
    
    device = config['device']
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    train_losses = []
    val_losses = []
    energy_stats = []
    
    start_time = time.time()
    
    for step in range(config['steps']):
        # Training
        model.train()
        x, y = get_batch('train', train_data, config['block_size'], 
                        config['batch_size'], device)
        
        if hasattr(model, 'use_pcn_guidance'):
            logits, loss, stats = model(x, y, return_stats=True)
            if 'final_refinement' in stats:
                energy_stats.append(stats['final_refinement'])
        else:
            logits, loss, stats = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Evaluation
        if step % config['eval_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_loss_batch = []
                for _ in range(5):
                    x_val, y_val = get_batch('val', val_data, config['block_size'],
                                           config['batch_size'], device)
                    _, vloss, _ = model(x_val, y_val)
                    val_loss_batch.append(vloss.item())
                
                val_loss = np.mean(val_loss_batch)
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                
                print(f"{model_name} Step {step}: train {loss.item():.3f}, val {val_loss:.3f}")
                
                if energy_stats and step > 0:
                    recent_energy = energy_stats[-1]
                    print(f"  Energy: {recent_energy.get('final_energy', 0):.3f}, "
                          f"Reduction: {recent_energy.get('energy_reduction', 0):.3f}")
    
    training_time = time.time() - start_time
    
    # Generate sample
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        if hasattr(model, 'generate'):
            generated = model.generate(context, max_new_tokens=100, 
                                     temperature=0.8, do_sample=True)
        else:
            # Fallback for standard transformer
            generated = generate_standard(model, context, 100, 0.8, device)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_val': val_losses[-1],
        'best_val': min(val_losses),
        'time': training_time,
        'generated': generated,
        'energy_stats': energy_stats[-10:] if energy_stats else []
    }


def generate_standard(model, context, max_tokens, temperature, device):
    """Generation for standard transformer."""
    idx = context
    for _ in range(max_tokens):
        idx_cond = idx if idx.size(1) <= 128 else idx[:, -128:]
        
        with torch.no_grad():
            logits, _, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def visualize_results(results: dict, save_path: str = 'energy_guidance_v2_results.png'):
    """Create visualization of experiment results."""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    steps = np.arange(len(results['standard']['val_losses'])) * 10
    
    for name in ['standard', 'lightweight', 'full']:
        if name in results:
            ax1.plot(steps, results[name]['val_losses'], label=name.capitalize(), linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Final performance
    ax2 = plt.subplot(2, 3, 2)
    models = list(results.keys())
    final_losses = [results[m]['final_val'] for m in models]
    
    bars = ax2.bar(models, final_losses, color=['blue', 'green', 'red'][:len(models)], alpha=0.7)
    ax2.set_ylabel('Final Validation Loss')
    ax2.set_title('Final Performance')
    
    for bar, val in zip(bars, final_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Training efficiency
    ax3 = plt.subplot(2, 3, 3)
    times = [results[m]['time'] for m in models]
    efficiency = [results['standard']['final_val'] / results[m]['final_val'] * 
                 results['standard']['time'] / results[m]['time'] for m in models]
    
    bars = ax3.bar(models, efficiency, color=['blue', 'green', 'red'][:len(models)], alpha=0.7)
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Performance per Unit Time')
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Energy trajectory (if available)
    ax4 = plt.subplot(2, 3, 4)
    
    for name in ['lightweight', 'full']:
        if name in results and results[name]['energy_stats']:
            energies = [s['final_energy'] for s in results[name]['energy_stats']]
            ax4.plot(energies, 'o-', label=name.capitalize())
    
    ax4.set_xlabel('Recent Steps')
    ax4.set_ylabel('Final Energy')
    ax4.set_title('Energy Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy reduction
    ax5 = plt.subplot(2, 3, 5)
    
    for name in ['lightweight', 'full']:
        if name in results and results[name]['energy_stats']:
            reductions = [s.get('energy_reduction', 0) for s in results[name]['energy_stats']]
            if reductions:
                ax5.bar(name, np.mean(reductions), alpha=0.7)
    
    ax5.set_ylabel('Average Energy Reduction')
    ax5.set_title('PCN Refinement Effectiveness')
    
    # 6. Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""Energy-Guided Transformer V2 Results
=====================================

Performance Improvement:
• Lightweight: {((results['standard']['final_val'] - results.get('lightweight', {}).get('final_val', float('inf'))) / results['standard']['final_val'] * 100):.1f}%
• Full Model: {((results['standard']['final_val'] - results.get('full', {}).get('final_val', float('inf'))) / results['standard']['final_val'] * 100):.1f}%

Efficiency:
• Lightweight overhead: {(results.get('lightweight', {}).get('time', 0) / results['standard']['time']):.1f}x
• Full model overhead: {(results.get('full', {}).get('time', 0) / results['standard']['time']):.1f}x

Key Innovation:
✓ Separate latent space (no embedding table dependency)
✓ Energy-based token selection
✓ Attention guidance through PCN biases
✓ Iterative refinement at output layer
"""
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nResults saved to {save_path}")


def main():
    # Configuration
    config = {
        'batch_size': 16,
        'block_size': 64,
        'lr': 3e-4,
        'steps': 200,
        'eval_interval': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Energy-Guided Transformer Experiment V2")
    print("=" * 60)
    print(f"Device: {config['device']}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Model configurations
    base_config = {
        'vocab_size': vocab_size,
        'block_size': config['block_size'],
        'n_embed': 128,
        'n_heads': 4,
        'n_layers': 3,
        'dropout': 0.1
    }
    
    results = {}
    
    # 1. Standard Transformer (baseline)
    print("\n" + "="*60)
    print("Training Standard Transformer...")
    print("="*60)
    
    standard = StandardTransformer(**base_config, device=config['device'])
    print(f"Parameters: {sum(p.numel() for p in standard.parameters()):,}")
    
    results['standard'] = train_and_evaluate(
        standard, train_data, val_data, config, "Standard"
    )
    
    # 2. Lightweight Energy-Guided (PCN at output only)
    print("\n" + "="*60)
    print("Training Lightweight Energy-Guided Transformer...")
    print("="*60)
    
    lightweight = LightweightEnergyTransformer(
        **base_config,
        pcn_latent_dim=64,
        n_refinement_steps=3,
        device=config['device']
    )
    print(f"Parameters: {sum(p.numel() for p in lightweight.parameters()):,}")
    
    results['lightweight'] = train_and_evaluate(
        lightweight, train_data, val_data, config, "Lightweight"
    )
    
    # 3. Full Energy-Guided (PCN throughout)
    print("\n" + "="*60)
    print("Training Full Energy-Guided Transformer...")
    print("="*60)
    
    full_model = EnergyGuidedTransformer(
        **base_config,
        use_pcn_guidance=True,
        pcn_latent_dim=128,
        n_refinement_steps=3,
        guidance_strength=0.1,
        use_contrastive=True,
        guided_layers=[1, 2],  # Guide middle layers
        device=config['device']
    )
    print(f"Parameters: {sum(p.numel() for p in full_model.parameters()):,}")
    
    results['full'] = train_and_evaluate(
        full_model, train_data, val_data, config, "Full"
    )
    
    # Generate samples
    print("\n" + "="*60)
    print("Generated Samples")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name.capitalize()}:")
        text = decode(result['generated'][0].cpu().numpy())
        print(text[:200])
    
    # Visualize results
    visualize_results(results)
    
    # Save results
    torch.save(results, 'energy_guidance_v2_results.pt')
    
    print("\n✓ Experiment complete!")
    print("  Files created:")
    print("    - energy_guidance_v2_results.png")
    print("    - energy_guidance_v2_results.pt")


if __name__ == '__main__':
    main()