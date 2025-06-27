"""
Compare different Hybrid PCN-Transformer architectures.

This script benchmarks all hybrid architectures against standard transformer
on TinyShakespeare, measuring:
- Training speed
- Convergence rate
- Final performance
- Memory usage
- Biological plausibility metrics
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# Add src to path
import sys
sys.path.append('src')

from models.hybrid_architectures import create_hybrid_model
from models.bigram import BigramLM
from data_loader import load_data, get_batch


class ArchitectureBenchmark:
    """Benchmark different architectures on the same task."""
    
    def __init__(self, vocab_size: int, batch_size: int = 32, 
                 block_size: int = 128, device: str = 'cuda'):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.results = {}
    
    def create_models(self, n_embed: int = 128, n_heads: int = 8, 
                     n_layers: int = 4) -> Dict[str, nn.Module]:
        """Create all models to compare."""
        models = {}
        
        # Standard transformer (baseline)
        print("Creating standard transformer...")
        config = type('Config', (), {
            'model': type('ModelConfig', (), {
                'n_embed': n_embed,
                'n_heads': n_heads,
                'n_layers': n_layers,
                'dropout': 0.2,
                'use_layer_norm': True,
                'use_residual': True,
                'norm_position': 'pre'
            })(),
            'training': type('TrainingConfig', (), {
                'device': self.device
            })()
        })()
        
        models['standard'] = BigramLM(
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            config=config
        ).to(self.device)
        
        # Hybrid architectures
        architectures = ['pcn_ff', 'alternating', 'hierarchical', 
                        'dual_stream', 'pcn_positional']
        
        for arch in architectures:
            print(f"Creating {arch} model...")
            models[arch] = create_hybrid_model(
                model_type=arch,
                vocab_size=self.vocab_size,
                batch_size=self.batch_size,
                block_size=self.block_size,
                n_embed=n_embed,
                n_heads=n_heads,
                n_layers=n_layers,
                device=self.device
            ).to(self.device)
        
        return models
    
    def benchmark_model(self, model: nn.Module, model_name: str, 
                       train_data: torch.Tensor, val_data: torch.Tensor,
                       num_iters: int = 1000) -> Dict:
        """Benchmark a single model."""
        print(f"\nBenchmarking {model_name}...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        
        # Training metrics
        train_losses = []
        val_losses = []
        iter_times = []
        memory_usage = []
        
        # Measure initial validation loss
        val_loss = self.evaluate(model, val_data, num_samples=50)
        val_losses.append(val_loss)
        
        # Training loop
        start_time = time.time()
        
        for iter in range(num_iters):
            iter_start = time.time()
            
            # Get batch
            xb, yb = get_batch('train', train_data, self.block_size, 
                              self.batch_size, self.device)
            
            # Forward pass
            logits, loss = model(xb, yb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record metrics
            train_losses.append(loss.item())
            iter_times.append(time.time() - iter_start)
            
            # Memory usage (GPU)
            if self.device == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            
            # Periodic evaluation
            if (iter + 1) % 100 == 0:
                val_loss = self.evaluate(model, val_data, num_samples=50)
                val_losses.append(val_loss)
                print(f"  Iter {iter+1}: train_loss={loss.item():.4f}, "
                      f"val_loss={val_loss:.4f}")
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_val_loss = self.evaluate(model, val_data, num_samples=200)
        
        # Compute convergence metrics
        convergence_iter = self.find_convergence(train_losses)
        
        # Biological plausibility score (based on architecture)
        bio_score = self.compute_biological_plausibility(model, model_name)
        
        results = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': final_val_loss,
            'total_time': total_time,
            'avg_iter_time': np.mean(iter_times),
            'convergence_iter': convergence_iter,
            'memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'biological_plausibility': bio_score
        }
        
        return results
    
    def evaluate(self, model: nn.Module, data: torch.Tensor, 
                num_samples: int = 50) -> float:
        """Evaluate model on validation data."""
        model.eval()
        losses = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                xb, yb = get_batch('val', data, self.block_size, 
                                  self.batch_size, self.device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
        
        model.train()
        return np.mean(losses)
    
    def find_convergence(self, losses: List[float], window: int = 50, 
                        threshold: float = 0.01) -> int:
        """Find iteration where model converges."""
        if len(losses) < window * 2:
            return len(losses)
        
        for i in range(window, len(losses) - window):
            before = np.mean(losses[i-window:i])
            after = np.mean(losses[i:i+window])
            if abs(before - after) / before < threshold:
                return i
        
        return len(losses)
    
    def compute_biological_plausibility(self, model: nn.Module, 
                                      model_name: str) -> float:
        """
        Compute biological plausibility score based on:
        - Local vs global computations
        - Presence of PCN layers
        - Hierarchical structure
        """
        score = 0.0
        
        # Base scores for different architectures
        if model_name == 'standard':
            score = 0.2  # Only local within layers
        elif model_name == 'pcn_ff':
            score = 0.6  # PCN in feedforward
        elif model_name == 'alternating':
            score = 0.7  # Balanced local/global
        elif model_name == 'hierarchical':
            score = 0.8  # Most biological
        elif model_name == 'dual_stream':
            score = 0.5  # Mixed approach
        elif model_name == 'pcn_positional':
            score = 0.4  # PCN only in positions
        
        return score
    
    def run_comparison(self, train_data: torch.Tensor, val_data: torch.Tensor,
                      num_iters: int = 1000) -> pd.DataFrame:
        """Run full comparison of all architectures."""
        # Create models
        models = self.create_models()
        
        # Benchmark each model
        for name, model in models.items():
            self.results[name] = self.benchmark_model(
                model, name, train_data, val_data, num_iters
            )
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Architecture': name,
                'Parameters': result['total_params'],
                'Final Val Loss': result['final_val_loss'],
                'Convergence Iter': result['convergence_iter'],
                'Avg Iter Time (ms)': result['avg_iter_time'] * 1000,
                'Memory (MB)': result['memory_usage'],
                'Bio Score': result['biological_plausibility']
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_results(self, save_dir: str = 'results'):
        """Plot comparison results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Training curves
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for name, result in self.results.items():
            losses = result['train_losses']
            plt.plot(losses[::10], label=name, alpha=0.8)  # Plot every 10th point
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for name, result in self.results.items():
            losses = result['val_losses']
            iters = np.linspace(0, len(result['train_losses']), len(losses))
            plt.plot(iters, losses, label=name, marker='o', alpha=0.8)
        plt.xlabel('Iteration')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()
        
        # 2. Performance vs Efficiency
        plt.figure(figsize=(10, 6))
        
        for name, result in self.results.items():
            plt.scatter(result['avg_iter_time'] * 1000, 
                       result['final_val_loss'],
                       s=result['total_params'] / 1000,  # Size by params
                       label=name, alpha=0.7)
        
        plt.xlabel('Average Iteration Time (ms)')
        plt.ylabel('Final Validation Loss')
        plt.title('Performance vs Training Speed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotations
        for name, result in self.results.items():
            plt.annotate(f"{result['total_params']//1000}K",
                        (result['avg_iter_time'] * 1000, result['final_val_loss']),
                        fontsize=8, alpha=0.6)
        
        plt.savefig(os.path.join(save_dir, 'performance_efficiency.png'), dpi=150)
        plt.close()
        
        # 3. Biological Plausibility vs Performance
        plt.figure(figsize=(8, 6))
        
        bio_scores = [r['biological_plausibility'] for r in self.results.values()]
        val_losses = [r['final_val_loss'] for r in self.results.values()]
        names = list(self.results.keys())
        
        plt.scatter(bio_scores, val_losses, s=100, alpha=0.7)
        for i, name in enumerate(names):
            plt.annotate(name, (bio_scores[i], val_losses[i]), 
                        fontsize=10, ha='center')
        
        plt.xlabel('Biological Plausibility Score')
        plt.ylabel('Final Validation Loss')
        plt.title('Biological Plausibility vs Performance')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'biological_performance.png'), dpi=150)
        plt.close()


def main():
    """Run architecture comparison."""
    print("Hybrid PCN-Transformer Architecture Comparison")
    print("=" * 60)
    
    # Load data
    print("\nLoading TinyShakespeare dataset...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark = ArchitectureBenchmark(
        vocab_size=vocab_size,
        batch_size=32,
        block_size=128,
        device=device
    )
    
    # Run comparison
    print("\nRunning architecture comparison...")
    df = benchmark.run_comparison(train_data, val_data, num_iters=1000)
    
    # Display results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'results/comparison_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save dataframe
    df.to_csv(os.path.join(save_dir, 'comparison.csv'), index=False)
    
    # Save detailed results
    with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, result in benchmark.results.items():
            json_results[name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
                if k not in ['train_losses', 'val_losses']  # Skip large arrays
            }
        json.dump(json_results, f, indent=2)
    
    # Plot results
    print(f"\nGenerating plots...")
    benchmark.plot_results(save_dir)
    
    print(f"\nResults saved to: {save_dir}")
    
    # Print insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Best performing
    best_perf = df.loc[df['Final Val Loss'].idxmin()]
    print(f"Best Performance: {best_perf['Architecture']} "
          f"(loss: {best_perf['Final Val Loss']:.4f})")
    
    # Fastest
    fastest = df.loc[df['Avg Iter Time (ms)'].idxmin()]
    print(f"Fastest Training: {fastest['Architecture']} "
          f"({fastest['Avg Iter Time (ms)']:.2f} ms/iter)")
    
    # Most biological
    most_bio = df.loc[df['Bio Score'].idxmax()]
    print(f"Most Biological: {most_bio['Architecture']} "
          f"(score: {most_bio['Bio Score']:.2f})")
    
    # Best trade-off (performance * bio_score / time)
    df['Trade-off Score'] = (1 / df['Final Val Loss']) * df['Bio Score'] / df['Avg Iter Time (ms)']
    best_trade = df.loc[df['Trade-off Score'].idxmax()]
    print(f"Best Trade-off: {best_trade['Architecture']}")


if __name__ == '__main__':
    main()