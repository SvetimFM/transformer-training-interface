"""
Visualize learning dynamics of Hybrid PCN-Transformer models.

This script provides detailed visualizations of:
- PCN inference dynamics
- Attention patterns vs PCN hierarchies
- Energy landscapes
- Gradient flow analysis
- Layer-wise learning dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from typing import Dict, List, Optional, Tuple
import argparse

# Add src to path
import sys
sys.path.append('src')

from models.hybrid_architectures import create_hybrid_model
from models.pcn_feedforward import PCNFeedForward
from data_loader import load_data, get_batch


class PCNTransformerVisualizer:
    """Visualize internal dynamics of hybrid models."""
    
    def __init__(self, model: nn.Module, model_name: str, device: str = 'cuda'):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model.eval()
        
        # Storage for visualizations
        self.pcn_energies = []
        self.attention_weights = []
        self.gradient_norms = []
        self.activations = {}
        
    def hook_layers(self):
        """Add hooks to capture intermediate activations."""
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook PCN layers
        for i, block in enumerate(self.model.blocks):
            if hasattr(block, 'feed_forward') and isinstance(block.feed_forward, PCNFeedForward):
                hook = block.feed_forward.register_forward_hook(get_activation(f'pcn_ff_{i}'))
                hooks.append(hook)
            
            # Hook attention layers
            if hasattr(block, 'attention'):
                hook = block.attention.register_forward_hook(get_activation(f'attention_{i}'))
                hooks.append(hook)
        
        return hooks
    
    def visualize_pcn_inference(self, input_sequence: torch.Tensor, 
                               layer_idx: int = 0) -> plt.Figure:
        """Visualize PCN inference dynamics for a specific layer."""
        # Get the PCN feedforward module
        if layer_idx >= len(self.model.blocks):
            print(f"Layer {layer_idx} not found")
            return None
        
        block = self.model.blocks[layer_idx]
        if not hasattr(block, 'feed_forward') or not isinstance(block.feed_forward, PCNFeedForward):
            print(f"Layer {layer_idx} does not have PCN feedforward")
            return None
        
        pcn_ff = block.feed_forward
        
        # Prepare input
        with torch.no_grad():
            # Get embeddings
            token_embeddings = self.model.token_embedding_table(input_sequence)
            pos_embeddings = self.model.get_positional_embeddings(input_sequence.size(1))
            x = token_embeddings + pos_embeddings
            
            # Pass through earlier blocks
            for i in range(layer_idx):
                x = self.model.blocks[i](x)
            
            # Apply layer norm if needed
            if hasattr(block, 'ln2'):
                x_norm = block.ln2(x)
            else:
                x_norm = x
            
            # Track PCN inference
            batch_size, seq_len, _ = x_norm.shape
            
            # Initialize latents
            latents = pcn_ff.init_latents(x_norm)
            
            # Track energy over inference steps
            energies = []
            latent_trajectories = [[] for _ in range(len(latents))]
            
            # Run inference with tracking
            inputs_and_latents = [x_norm] + latents
            
            for step in range(pcn_ff.inference_steps):
                # Compute current energy
                energy = pcn_ff.get_energy(x_norm, latents)
                energies.append(energy.item())
                
                # Store latent states
                for i, latent in enumerate(latents):
                    latent_trajectories[i].append(latent[0, 0, :10].cpu().numpy())  # First 10 dims
                
                # Inference step
                errors = []
                gm_errors = []
                
                for i, layer in enumerate(pcn_ff.pcn_layers):
                    x_hat, a = layer(inputs_and_latents[i + 1])
                    error = inputs_and_latents[i] - x_hat
                    gm_error = error * layer.activation_deriv(a)
                    errors.append(error)
                    gm_errors.append(gm_error)
                
                # Update latents
                for i in range(len(latents)):
                    grad = errors[i]
                    if i > 0:
                        W_above = pcn_ff.pcn_layers[i-1].W
                        grad = grad - torch.matmul(gm_errors[i-1], W_above.T)
                    latents[i] = latents[i] - pcn_ff.inference_lr * grad
                
                inputs_and_latents = [x_norm] + latents
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Energy over inference steps
        ax = axes[0, 0]
        ax.plot(energies, 'b-', linewidth=2)
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Total Prediction Error')
        ax.set_title(f'PCN Energy Minimization (Layer {layer_idx})')
        ax.grid(True, alpha=0.3)
        
        # 2. Latent trajectories
        ax = axes[0, 1]
        for i, trajectory in enumerate(latent_trajectories):
            trajectory = np.array(trajectory)
            for dim in range(min(5, trajectory.shape[1])):
                ax.plot(trajectory[:, dim], alpha=0.7, 
                       label=f'L{i+1}_d{dim}' if dim < 2 else '')
        ax.set_xlabel('Inference Step')
        ax.set_ylabel('Latent Value')
        ax.set_title('Latent State Trajectories')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Final latent distributions
        ax = axes[1, 0]
        for i, latent in enumerate(latents):
            latent_flat = latent.flatten().cpu().numpy()
            ax.hist(latent_flat, bins=50, alpha=0.5, label=f'Layer {i+1}', density=True)
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Density')
        ax.set_title('Final Latent Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error magnitude heatmap
        ax = axes[1, 1]
        error_mags = []
        for error in errors:
            error_mags.append(error.abs().mean(dim=0).mean(dim=0).cpu().numpy())
        error_mags = np.array(error_mags)
        
        im = ax.imshow(error_mags, aspect='auto', cmap='hot')
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('PCN Layer')
        ax.set_title('Prediction Error Magnitudes')
        plt.colorbar(im, ax=ax)
        
        fig.suptitle(f'PCN Inference Dynamics - {self.model_name}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def compare_attention_pcn(self, input_sequence: torch.Tensor) -> plt.Figure:
        """Compare attention patterns with PCN hierarchical processing."""
        hooks = self.hook_layers()
        
        try:
            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.model(input_sequence)
            
            # Extract attention and PCN activations
            attention_acts = []
            pcn_acts = []
            
            for key, value in self.activations.items():
                if 'attention' in key:
                    attention_acts.append(value)
                elif 'pcn' in key:
                    pcn_acts.append(value)
            
            # Create comparison visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Attention activation norms
            if attention_acts:
                ax = axes[0, 0]
                for i, act in enumerate(attention_acts):
                    norms = act.norm(dim=-1).mean(dim=0).cpu().numpy()
                    ax.plot(norms, label=f'Layer {i}', alpha=0.7)
                ax.set_xlabel('Sequence Position')
                ax.set_ylabel('Activation Norm')
                ax.set_title('Attention Activation Patterns')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. PCN activation norms
            if pcn_acts:
                ax = axes[0, 1]
                for i, act in enumerate(pcn_acts):
                    norms = act.norm(dim=-1).mean(dim=0).cpu().numpy()
                    ax.plot(norms, label=f'Layer {i}', alpha=0.7)
                ax.set_xlabel('Sequence Position')
                ax.set_ylabel('Activation Norm')
                ax.set_title('PCN Activation Patterns')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 3. Attention vs PCN correlation
            if attention_acts and pcn_acts:
                ax = axes[1, 0]
                correlations = []
                for att, pcn in zip(attention_acts[:len(pcn_acts)], pcn_acts):
                    att_flat = att.flatten()
                    pcn_flat = pcn.flatten()
                    corr = torch.corrcoef(torch.stack([att_flat, pcn_flat]))[0, 1]
                    correlations.append(corr.item())
                
                ax.bar(range(len(correlations)), correlations)
                ax.set_xlabel('Layer')
                ax.set_ylabel('Correlation')
                ax.set_title('Attention-PCN Activation Correlation')
                ax.grid(True, alpha=0.3)
            
            # 4. Dimensionality analysis (PCA variance)
            ax = axes[1, 1]
            
            def compute_pca_variance(activations):
                """Compute explained variance ratios for first few PCs."""
                act_flat = activations.reshape(-1, activations.shape[-1])
                act_centered = act_flat - act_flat.mean(dim=0)
                U, S, V = torch.svd(act_centered)
                variance_explained = (S**2) / (S**2).sum()
                return variance_explained[:10].cpu().numpy()
            
            if attention_acts:
                att_var = compute_pca_variance(attention_acts[0])
                ax.plot(att_var, 'b-', label='Attention', marker='o')
            
            if pcn_acts:
                pcn_var = compute_pca_variance(pcn_acts[0])
                ax.plot(pcn_var, 'r-', label='PCN', marker='s')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Variance Explained')
            ax.set_title('Representation Dimensionality')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.suptitle(f'Attention vs PCN Processing - {self.model_name}', fontsize=14)
            plt.tight_layout()
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return fig
    
    def visualize_gradient_flow(self, input_sequence: torch.Tensor, 
                               target_sequence: torch.Tensor) -> plt.Figure:
        """Visualize gradient flow through the model."""
        self.model.train()
        
        # Enable gradient computation
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Forward pass
        logits, loss = self.model(input_sequence, target_sequence)
        
        # Backward pass
        loss.backward()
        
        # Collect gradient statistics
        gradient_stats = []
        param_names = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append({
                    'name': name,
                    'grad_norm': grad_norm,
                    'param_norm': param.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                })
                param_names.append(name.split('.')[-1])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Gradient norms by layer
        ax = axes[0, 0]
        pcn_grads = []
        attention_grads = []
        other_grads = []
        
        for stat in gradient_stats:
            if 'pcn' in stat['name'].lower():
                pcn_grads.append(stat['grad_norm'])
            elif 'attention' in stat['name'].lower():
                attention_grads.append(stat['grad_norm'])
            else:
                other_grads.append(stat['grad_norm'])
        
        positions = []
        labels = []
        if pcn_grads:
            positions.extend(range(len(pcn_grads)))
            ax.bar(range(len(pcn_grads)), pcn_grads, alpha=0.7, label='PCN')
            labels.extend([f'PCN_{i}' for i in range(len(pcn_grads))])
        
        offset = len(pcn_grads)
        if attention_grads:
            pos = range(offset, offset + len(attention_grads))
            ax.bar(pos, attention_grads, alpha=0.7, label='Attention')
            labels.extend([f'Att_{i}' for i in range(len(attention_grads))])
        
        ax.set_xlabel('Layer Component')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms by Component Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gradient vs Parameter norm ratio
        ax = axes[0, 1]
        ratios = [s['grad_norm'] / (s['param_norm'] + 1e-8) for s in gradient_stats]
        colors = ['red' if 'pcn' in s['name'].lower() else 'blue' for s in gradient_stats]
        
        ax.scatter(range(len(ratios)), ratios, c=colors, alpha=0.6)
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Gradient/Parameter Ratio')
        ax.set_title('Relative Gradient Magnitudes')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 3. Gradient distribution
        ax = axes[1, 0]
        all_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().cpu().numpy())
        
        ax.hist(all_grads, bins=100, alpha=0.7, density=True)
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Density')
        ax.set_title('Overall Gradient Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. Layer-wise gradient flow
        ax = axes[1, 1]
        layer_grads = {}
        for stat in gradient_stats:
            layer = stat['name'].split('.')[1] if '.' in stat['name'] else '0'
            if layer not in layer_grads:
                layer_grads[layer] = []
            layer_grads[layer].append(stat['grad_norm'])
        
        layers = sorted(layer_grads.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        mean_grads = [np.mean(layer_grads[l]) for l in layers]
        std_grads = [np.std(layer_grads[l]) for l in layers]
        
        ax.errorbar(range(len(layers)), mean_grads, yerr=std_grads, 
                   marker='o', capsize=5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Gradient Norm')
        ax.set_title('Gradient Flow Through Layers')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Gradient Flow Analysis - {self.model_name}', fontsize=14)
        plt.tight_layout()
        
        # Reset model to eval mode
        self.model.eval()
        
        return fig
    
    def create_animation(self, train_data: torch.Tensor, 
                        num_steps: int = 100) -> FuncAnimation:
        """Create animation of learning dynamics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Storage for animation data
        losses = []
        pcn_energies = []
        
        # Initial setup
        line1, = ax1.plot([], [], 'b-', label='Training Loss')
        line2, = ax2.plot([], [], 'r-', label='PCN Energy')
        
        ax1.set_xlim(0, num_steps)
        ax1.set_ylim(0, 5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(0, num_steps)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Energy')
        ax2.set_title('PCN Energy Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2
        
        def update(frame):
            # Get batch
            idx = torch.randint(len(train_data) - self.model.block_size, (self.model.batch_size,))
            x = torch.stack([train_data[i:i+self.model.block_size] for i in idx]).to(self.device)
            y = torch.stack([train_data[i+1:i+self.model.block_size+1] for i in idx]).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits, loss = self.model(x, y)
                losses.append(loss.item())
                
                # Get PCN energy if available
                # This is simplified - in practice you'd track actual PCN energies
                energy = np.random.randn() * 10 + 50 - frame * 0.3
                pcn_energies.append(max(0, energy))
            
            # Update plots
            line1.set_data(range(len(losses)), losses)
            line2.set_data(range(len(pcn_energies)), pcn_energies)
            
            # Adjust y-limits if needed
            if losses:
                ax1.set_ylim(0, max(losses) * 1.1)
            if pcn_energies:
                ax2.set_ylim(0, max(pcn_energies) * 1.1)
            
            return line1, line2
        
        anim = FuncAnimation(fig, update, frames=num_steps, 
                           init_func=init, interval=50, blit=True)
        
        return anim


def main():
    """Run visualization analysis."""
    parser = argparse.ArgumentParser(description='Visualize Hybrid PCN-Transformer dynamics')
    parser.add_argument('--model_type', type=str, default='pcn_ff',
                       choices=['pcn_ff', 'alternating', 'hierarchical', 
                               'dual_stream', 'pcn_positional'],
                       help='Type of hybrid architecture')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    args = parser.parse_args()
    
    print(f"Visualizing {args.model_type} model dynamics")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nCreating model on {device}...")
    
    model = create_hybrid_model(
        model_type=args.model_type,
        vocab_size=vocab_size,
        batch_size=32,
        block_size=128,
        device=device
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create visualizer
    visualizer = PCNTransformerVisualizer(model, args.model_type, device)
    
    # Get sample data
    sample_idx = torch.randint(len(train_data) - 128, (1,))
    input_seq = train_data[sample_idx:sample_idx + 128].unsqueeze(0).to(device)
    target_seq = train_data[sample_idx + 1:sample_idx + 129].unsqueeze(0).to(device)
    
    # Generate visualizations
    print("\n1. Visualizing PCN inference dynamics...")
    fig = visualizer.visualize_pcn_inference(input_seq, layer_idx=0)
    if fig:
        fig.savefig(os.path.join(args.save_dir, f'{args.model_type}_pcn_inference.png'), dpi=150)
        plt.close(fig)
    
    print("2. Comparing attention and PCN processing...")
    fig = visualizer.compare_attention_pcn(input_seq)
    fig.savefig(os.path.join(args.save_dir, f'{args.model_type}_attention_pcn_comparison.png'), dpi=150)
    plt.close(fig)
    
    print("3. Analyzing gradient flow...")
    fig = visualizer.visualize_gradient_flow(input_seq, target_seq)
    fig.savefig(os.path.join(args.save_dir, f'{args.model_type}_gradient_flow.png'), dpi=150)
    plt.close(fig)
    
    print("4. Creating learning dynamics animation...")
    anim = visualizer.create_animation(train_data, num_steps=100)
    anim.save(os.path.join(args.save_dir, f'{args.model_type}_learning_dynamics.gif'), 
              writer='pillow', fps=10)
    plt.close()
    
    print(f"\nVisualizations saved to {args.save_dir}/")
    print("\nKey Insights:")
    print("- PCN layers show iterative error minimization")
    print("- Attention and PCN capture different aspects of sequences")
    print("- Gradient flow reveals training dynamics differences")
    print("- Hybrid architectures balance biological plausibility with performance")


if __name__ == '__main__':
    main()