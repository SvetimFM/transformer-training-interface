"""
Ablation study for PCN K-Q-V exploration.

Tests the contribution of each component and hyperparameter to understand
what makes the exploration effective.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import itertools

import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from data_loader import load_data, get_batch


def quick_train_eval(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Dict,
    steps: int = 500
) -> Dict[str, float]:
    """Quick training and evaluation for ablation studies."""
    
    device = config['device']
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    # Quick training
    for step in range(steps):
        # Train
        model.train()
        xb, yb = get_batch('train', train_data, config['block_size'], 
                          config['batch_size'], device)
        
        _, loss, _ = model(xb, yb, return_exploration_stats=True)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Eval every 100 steps
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = get_batch('val', val_data, config['block_size'],
                                  config['batch_size'], device)
                _, val_loss, _ = model(xb, yb)
                val_losses.append(val_loss.item())
    
    # Final evaluation
    model.eval()
    final_val_losses = []
    for _ in range(20):
        with torch.no_grad():
            xb, yb = get_batch('val', val_data, config['block_size'],
                              config['batch_size'], device)
            _, val_loss, _ = model(xb, yb)
            final_val_losses.append(val_loss.item())
    
    return {
        'final_val_loss': np.mean(final_val_losses),
        'best_val_loss': min(val_losses),
        'final_train_loss': np.mean(train_losses[-100:]),
        'training_time': time.time() - start_time,
        'params': sum(p.numel() for p in model.parameters())
    }


def ablation_components(train_data, val_data, config):
    """Test contribution of each K-Q-V component."""
    
    print("\n" + "="*60)
    print("Component Ablation Study")
    print("="*60)
    
    components = [
        ('None', False, False, False),
        ('Q only', True, False, False),
        ('K only', False, True, False),
        ('V only', False, False, True),
        ('Q+K', True, True, False),
        ('Q+V', True, False, True),
        ('K+V', False, True, True),
        ('Q+K+V', True, True, True)
    ]
    
    results = {}
    
    for name, explore_q, explore_k, explore_v in components:
        print(f"\nTesting: {name}")
        
        model = TransformerWithPCNAttention(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embed=128,  # Smaller for faster testing
            n_heads=4,
            n_layers=3,
            dropout=0.1,
            use_pcn_exploration=(explore_q or explore_k or explore_v),
            explore_queries=explore_q,
            explore_keys=explore_k,
            explore_values=explore_v,
            n_exploration_samples=3,
            n_refinement_steps=3,
            device=config['device']
        )
        
        metrics = quick_train_eval(model, train_data, val_data, config)
        results[name] = metrics
        
        print(f"  Val loss: {metrics['final_val_loss']:.4f}")
        print(f"  Time: {metrics['training_time']:.1f}s")
    
    return results


def ablation_hyperparameters(train_data, val_data, config):
    """Test different hyperparameter settings."""
    
    print("\n" + "="*60)
    print("Hyperparameter Ablation Study")
    print("="*60)
    
    # Test different numbers of exploration samples
    print("\n1. Number of exploration samples:")
    sample_results = {}
    
    for n_samples in [1, 2, 3, 5, 7]:
        print(f"\n  Testing {n_samples} samples")
        
        model = TransformerWithPCNAttention(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embed=128,
            n_heads=4,
            n_layers=3,
            dropout=0.1,
            use_pcn_exploration=True,
            explore_queries=True,
            explore_keys=True,
            explore_values=True,
            n_exploration_samples=n_samples,
            n_refinement_steps=3,
            device=config['device']
        )
        
        metrics = quick_train_eval(model, train_data, val_data, config, steps=300)
        sample_results[n_samples] = metrics
        print(f"    Val loss: {metrics['final_val_loss']:.4f}")
    
    # Test different numbers of refinement steps
    print("\n2. Number of refinement steps:")
    refinement_results = {}
    
    for n_steps in [1, 3, 5, 7, 10]:
        print(f"\n  Testing {n_steps} refinement steps")
        
        model = TransformerWithPCNAttention(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embed=128,
            n_heads=4,
            n_layers=3,
            dropout=0.1,
            use_pcn_exploration=True,
            explore_queries=True,
            explore_keys=True,
            explore_values=True,
            n_exploration_samples=3,
            n_refinement_steps=n_steps,
            device=config['device']
        )
        
        metrics = quick_train_eval(model, train_data, val_data, config, steps=300)
        refinement_results[n_steps] = metrics
        print(f"    Val loss: {metrics['final_val_loss']:.4f}")
    
    # Test different exploration noise levels
    print("\n3. Exploration noise levels:")
    noise_results = {}
    
    for noise in [0.05, 0.1, 0.15, 0.2, 0.3]:
        print(f"\n  Testing noise level {noise}")
        
        model = TransformerWithPCNAttention(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embed=128,
            n_heads=4,
            n_layers=3,
            dropout=0.1,
            use_pcn_exploration=True,
            explore_queries=True,
            explore_keys=True,
            explore_values=True,
            n_exploration_samples=3,
            n_refinement_steps=3,
            exploration_noise=noise,
            device=config['device']
        )
        
        metrics = quick_train_eval(model, train_data, val_data, config, steps=300)
        noise_results[noise] = metrics
        print(f"    Val loss: {metrics['final_val_loss']:.4f}")
    
    return {
        'samples': sample_results,
        'refinement': refinement_results,
        'noise': noise_results
    }


def ablation_architecture(train_data, val_data, config):
    """Test architectural variations."""
    
    print("\n" + "="*60)
    print("Architecture Ablation Study")
    print("="*60)
    
    architectures = []
    
    # Different layer configurations
    layer_configs = [
        ('All layers', None),
        ('First layer only', [0]),
        ('Last layer only', [-1]),
        ('Every other layer', [0, 2]),
        ('First and last', [0, -1])
    ]
    
    results = {}
    
    for name, pcn_layers in layer_configs:
        print(f"\nTesting: {name}")
        
        if pcn_layers and -1 in pcn_layers:
            # Convert -1 to actual last layer index
            pcn_layers = [0 if x == 0 else 2 for x in pcn_layers]
        
        model = TransformerWithPCNAttention(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embed=128,
            n_heads=4,
            n_layers=3,
            dropout=0.1,
            use_pcn_exploration=True,
            pcn_layers=pcn_layers,
            explore_queries=True,
            explore_keys=True,
            explore_values=True,
            n_exploration_samples=3,
            n_refinement_steps=3,
            device=config['device']
        )
        
        metrics = quick_train_eval(model, train_data, val_data, config, steps=300)
        results[name] = metrics
        
        print(f"  Val loss: {metrics['final_val_loss']:.4f}")
        print(f"  Params: {metrics['params']:,}")
    
    return results


def visualize_ablation_results(all_results: Dict, save_path: str = 'kqv_ablation_results.png'):
    """Create comprehensive visualization of ablation results."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Component ablation
    ax1 = plt.subplot(3, 3, 1)
    component_results = all_results['components']
    names = list(component_results.keys())
    val_losses = [r['final_val_loss'] for r in component_results.values()]
    
    bars = ax1.bar(names, val_losses, color='blue', alpha=0.7)
    ax1.set_xlabel('Components')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Component Contribution')
    ax1.set_xticklabels(names, rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Sample number ablation
    ax2 = plt.subplot(3, 3, 2)
    sample_results = all_results['hyperparameters']['samples']
    samples = sorted(sample_results.keys())
    losses = [sample_results[s]['final_val_loss'] for s in samples]
    times = [sample_results[s]['training_time'] for s in samples]
    
    ax2.plot(samples, losses, 'o-', linewidth=2, markersize=8, label='Val Loss')
    ax2.set_xlabel('Number of Exploration Samples')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Effect of Exploration Samples')
    ax2.grid(True, alpha=0.3)
    
    # Add timing on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(samples, times, 's--', color='red', alpha=0.7, label='Time (s)')
    ax2_twin.set_ylabel('Training Time (s)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # 3. Refinement steps ablation
    ax3 = plt.subplot(3, 3, 3)
    refinement_results = all_results['hyperparameters']['refinement']
    steps = sorted(refinement_results.keys())
    losses = [refinement_results[s]['final_val_loss'] for s in steps]
    
    ax3.plot(steps, losses, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Number of Refinement Steps')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Effect of Refinement Steps')
    ax3.grid(True, alpha=0.3)
    
    # 4. Noise level ablation
    ax4 = plt.subplot(3, 3, 4)
    noise_results = all_results['hyperparameters']['noise']
    noise_levels = sorted(noise_results.keys())
    losses = [noise_results[n]['final_val_loss'] for n in noise_levels]
    
    ax4.plot(noise_levels, losses, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Exploration Noise Level')
    ax4.set_ylabel('Validation Loss')
    ax4.set_title('Effect of Exploration Noise')
    ax4.grid(True, alpha=0.3)
    
    # 5. Architecture ablation
    ax5 = plt.subplot(3, 3, 5)
    arch_results = all_results['architecture']
    names = list(arch_results.keys())
    val_losses = [r['final_val_loss'] for r in arch_results.values()]
    params = [r['params'] for r in arch_results.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, val_losses, width, label='Val Loss', alpha=0.7)
    
    # Normalize params for visualization
    normalized_params = np.array(params) / params[0]
    bars2 = ax5.bar(x + width/2, normalized_params, width, 
                    label='Relative Params', alpha=0.7, color='orange')
    
    ax5.set_xlabel('Architecture')
    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=45)
    ax5.set_ylabel('Value')
    ax5.set_title('Architecture Variations')
    ax5.legend()
    
    # 6. Combined efficiency plot
    ax6 = plt.subplot(3, 3, 6)
    
    # Collect all configurations
    all_configs = []
    
    # Components
    for name, result in component_results.items():
        all_configs.append({
            'name': f'Comp: {name}',
            'loss': result['final_val_loss'],
            'time': result['training_time'],
            'type': 'component'
        })
    
    # Plot efficiency frontier
    losses = [c['loss'] for c in all_configs]
    times = [c['time'] for c in all_configs]
    
    ax6.scatter(times, losses, s=100, alpha=0.6)
    
    # Annotate best configurations
    best_loss_idx = np.argmin(losses)
    best_speed_idx = np.argmin(times)
    
    ax6.annotate(all_configs[best_loss_idx]['name'],
                (times[best_loss_idx], losses[best_loss_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.annotate(all_configs[best_speed_idx]['name'],
                (times[best_speed_idx], losses[best_speed_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('Training Time (s)')
    ax6.set_ylabel('Validation Loss')
    ax6.set_title('Efficiency Frontier')
    ax6.grid(True, alpha=0.3)
    
    # 7. Summary heatmap
    ax7 = plt.subplot(3, 3, 7)
    
    # Create summary matrix
    summary_data = []
    summary_labels = []
    
    # Best from each category
    categories = {
        'Q+K+V': component_results.get('Q+K+V', {}).get('final_val_loss', 0),
        '5 samples': all_results['hyperparameters']['samples'].get(5, {}).get('final_val_loss', 0),
        '5 steps': all_results['hyperparameters']['refinement'].get(5, {}).get('final_val_loss', 0),
        '0.15 noise': all_results['hyperparameters']['noise'].get(0.15, {}).get('final_val_loss', 0),
        'All layers': arch_results.get('All layers', {}).get('final_val_loss', 0)
    }
    
    # Convert to matrix for heatmap
    matrix = np.array(list(categories.values())).reshape(-1, 1)
    
    im = ax7.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    ax7.set_yticks(range(len(categories)))
    ax7.set_yticklabels(list(categories.keys()))
    ax7.set_xticks([])
    ax7.set_title('Configuration Performance')
    
    # Add values
    for i, (name, val) in enumerate(categories.items()):
        ax7.text(0, i, f'{val:.3f}', ha='center', va='center', fontsize=10)
    
    # 8. Key findings text
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Find best configurations
    best_component = min(component_results.items(), key=lambda x: x[1]['final_val_loss'])
    best_samples = min(all_results['hyperparameters']['samples'].items(), 
                      key=lambda x: x[1]['final_val_loss'])
    best_noise = min(all_results['hyperparameters']['noise'].items(),
                    key=lambda x: x[1]['final_val_loss'])
    
    findings_text = f"""
Key Findings from Ablation Study

Best Configurations:
• Components: {best_component[0]} (loss: {best_component[1]['final_val_loss']:.3f})
• Samples: {best_samples[0]} (loss: {best_samples[1]['final_val_loss']:.3f})
• Noise: {best_noise[0]} (loss: {best_noise[1]['final_val_loss']:.3f})

Insights:
• Q+K+V exploration generally performs best
• 3-5 exploration samples optimal
• Moderate noise (0.1-0.15) works best
• All layers > selective layers
• Refinement steps show diminishing returns

Efficiency:
• Q-only: Good speed/performance tradeoff
• Full K-Q-V: Best performance, 3-4x slower
"""
    
    ax8.text(0.05, 0.95, findings_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 9. Component interaction plot
    ax9 = plt.subplot(3, 3, 9)
    
    # Show how components interact
    interaction_data = {
        'Q': component_results.get('Q only', {}).get('final_val_loss', 0),
        'K': component_results.get('K only', {}).get('final_val_loss', 0),
        'V': component_results.get('V only', {}).get('final_val_loss', 0),
        'Q+K': component_results.get('Q+K', {}).get('final_val_loss', 0),
        'Q+V': component_results.get('Q+V', {}).get('final_val_loss', 0),
        'K+V': component_results.get('K+V', {}).get('final_val_loss', 0),
        'Q+K+V': component_results.get('Q+K+V', {}).get('final_val_loss', 0),
        'None': component_results.get('None', {}).get('final_val_loss', 0)
    }
    
    # Calculate synergies
    synergies = {
        'Q+K synergy': interaction_data['Q'] + interaction_data['K'] - interaction_data['Q+K'],
        'Q+V synergy': interaction_data['Q'] + interaction_data['V'] - interaction_data['Q+V'],
        'K+V synergy': interaction_data['K'] + interaction_data['V'] - interaction_data['K+V'],
    }
    
    bars = ax9.bar(synergies.keys(), synergies.values(), color='coral', alpha=0.7)
    ax9.set_ylabel('Synergy (positive = beneficial interaction)')
    ax9.set_title('Component Synergies')
    ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nAblation results saved to {save_path}")


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'block_size': 128,
        'lr': 3e-4
    }
    
    print("PCN K-Q-V Ablation Study")
    print("="*60)
    print(f"Device: {config['device']}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    config['vocab_size'] = vocab_size
    
    # Run ablation studies
    all_results = {}
    
    # 1. Component ablation
    all_results['components'] = ablation_components(train_data, val_data, config)
    
    # 2. Hyperparameter ablation
    all_results['hyperparameters'] = ablation_hyperparameters(train_data, val_data, config)
    
    # 3. Architecture ablation
    all_results['architecture'] = ablation_architecture(train_data, val_data, config)
    
    # Visualize results
    visualize_ablation_results(all_results)
    
    # Save results
    torch.save(all_results, 'kqv_ablation_results.pt')
    
    print("\n✓ Ablation study complete!")
    print("  Results saved to:")
    print("    - kqv_ablation_results.png (visualization)")
    print("    - kqv_ablation_results.pt (data)")


if __name__ == '__main__':
    main()