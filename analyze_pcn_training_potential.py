"""
Analyze PCN-Transformer training potential and suggest improvements.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('src')

from models.hybrid_architectures import create_hybrid_model
from data_loader import load_data, get_batch


def analyze_convergence_rate():
    """Analyze if the model was still improving at 2000 iterations."""
    print("=" * 60)
    print("ANALYZING PCN-TRANSFORMER CONVERGENCE")
    print("=" * 60)
    
    # Check if we have saved training history
    checkpoint_path = 'checkpoints/pcn_ff/20250624_160112/history.pt'
    
    if os.path.exists(checkpoint_path):
        history = torch.load(checkpoint_path, weights_only=False)
        val_losses = history['val_losses']
        
        print(f"\nTraining history found: {len(val_losses)} validation points")
        
        # Analyze convergence
        if len(val_losses) >= 5:
            # Check improvement in last 20% of training
            split_point = int(len(val_losses) * 0.8)
            early_losses = val_losses[:split_point]
            late_losses = val_losses[split_point:]
            
            early_avg = np.mean(early_losses)
            late_avg = np.mean(late_losses)
            improvement = (early_avg - late_avg) / early_avg * 100
            
            print(f"\nConvergence Analysis:")
            print(f"  Early training avg loss: {early_avg:.4f}")
            print(f"  Late training avg loss: {late_avg:.4f}")
            print(f"  Improvement in last 20%: {improvement:.2f}%")
            
            # Check if still improving
            last_5_improvements = []
            for i in range(len(val_losses)-5, len(val_losses)):
                if i > 0:
                    imp = (val_losses[i-1] - val_losses[i]) / val_losses[i-1] * 100
                    last_5_improvements.append(imp)
            
            avg_recent_improvement = np.mean(last_5_improvements)
            print(f"  Average improvement per eval (last 5): {avg_recent_improvement:.3f}%")
            
            # Recommendation
            still_improving = avg_recent_improvement > 0.1  # Still improving if >0.1% per eval
            
            return {
                'still_improving': still_improving,
                'improvement_rate': avg_recent_improvement,
                'final_loss': val_losses[-1],
                'best_loss': min(val_losses)
            }
    
    return None


def estimate_training_needs():
    """Estimate how much more training might be needed."""
    print("\n" + "=" * 60)
    print("ESTIMATING ADDITIONAL TRAINING NEEDS")
    print("=" * 60)
    
    # Known performance targets
    pcn_current = 2.4575  # Current PCN-FF validation loss
    transformer_target = 1.8395  # Standard transformer performance
    gap = pcn_current - transformer_target
    
    print(f"\nPerformance Gap Analysis:")
    print(f"  Current PCN-FF loss: {pcn_current:.4f}")
    print(f"  Target (transformer): {transformer_target:.4f}")
    print(f"  Gap to close: {gap:.4f} ({gap/pcn_current*100:.1f}%)")
    
    # Analyze based on training history
    convergence_info = analyze_convergence_rate()
    
    if convergence_info and convergence_info['still_improving']:
        improvement_rate = convergence_info['improvement_rate']
        
        # Estimate iterations needed (assuming exponential decay)
        if improvement_rate > 0:
            # Rough estimate: how many evals to close the gap
            evals_needed = gap / (pcn_current * improvement_rate / 100)
            # Convert to iterations (assuming eval every 100 iters)
            iters_needed = int(evals_needed * 100)
            
            print(f"\nTraining Continuation Estimate:")
            print(f"  Current improvement rate: {improvement_rate:.3f}% per eval")
            print(f"  Estimated additional iterations: {iters_needed:,}")
            print(f"  Estimated additional time: {iters_needed/2000*5:.1f} minutes")
            
            # Feasibility assessment
            if iters_needed < 10000:
                print(f"\n✅ RECOMMENDATION: Continue training")
                print(f"   - Model is still improving")
                print(f"   - Gap might close with {iters_needed:,} more iterations")
            else:
                print(f"\n⚠️  RECOMMENDATION: Need architectural improvements")
                print(f"   - Would need {iters_needed:,} iterations (impractical)")
                print(f"   - Improvement rate too slow to close gap")
        else:
            print(f"\n❌ RECOMMENDATION: Model has plateaued")
            print(f"   - No recent improvement detected")
            print(f"   - Need architectural changes")
    else:
        print(f"\n⚠️  Cannot determine from available data")
        print(f"   - Recommend running extended training experiment")


def suggest_improvements():
    """Suggest concrete improvements based on analysis."""
    print("\n" + "=" * 60)
    print("IMPROVEMENT STRATEGIES")
    print("=" * 60)
    
    print("\n1. IMMEDIATE EXPERIMENTS (High Impact, Low Effort):")
    print("   a) Extended Training:")
    print("      - Train for 5000 iterations with checkpoints")
    print("      - Use learning rate schedule: cosine annealing")
    print("      - Command: python train_hybrid_pcn_transformer.py --max_iters 5000 --lr_schedule cosine")
    
    print("\n   b) Hyperparameter Tuning:")
    print("      - Reduce inference steps: 3 (faster) or increase to 10 (better convergence)")
    print("      - Adjust inference LR: try 0.01, 0.05, 0.2")
    print("      - Increase model size: n_embed=256, n_layers=6")
    
    print("\n2. ARCHITECTURAL IMPROVEMENTS (Medium Effort):")
    print("   a) PCN-Specific Optimizations:")
    print("      - Learnable inference learning rate per layer")
    print("      - Adaptive inference steps based on error magnitude")
    print("      - Momentum in PCN latent updates")
    
    print("\n   b) Hybrid Architecture Variants:")
    print("      - Test Alternating architecture (might be more efficient)")
    print("      - Try Hierarchical (PCN preprocessing might help)")
    print("      - Implement PCN-Attention hybrid (PCN in attention computation)")
    
    print("\n3. TRAINING STRATEGY IMPROVEMENTS (High Impact):")
    print("   a) Two-Phase Training:")
    print("      - Phase 1: Train with more inference steps (10-20)")
    print("      - Phase 2: Reduce to 3-5 steps for efficiency")
    
    print("   b) Curriculum Learning:")
    print("      - Start with shorter sequences (32 tokens)")
    print("      - Gradually increase to full length (256 tokens)")
    
    print("   c) PCN-Aware Optimization:")
    print("      - Different learning rates for PCN vs transformer layers")
    print("      - Gradient clipping specifically for PCN parameters")
    
    print("\n4. THEORETICAL INNOVATIONS (High Effort, High Reward):")
    print("   a) Backprop-Free Training:")
    print("      - Implement true local learning for PCN layers")
    print("      - Use PCN error signals instead of backprop")
    
    print("   b) Energy-Based Training:")
    print("      - Add PCN energy as auxiliary loss")
    print("      - Encourage faster convergence of latents")
    
    print("   c) Biological Constraints:")
    print("      - Add sparsity constraints to PCN activations")
    print("      - Implement Dale's law (separate excitatory/inhibitory)")


def create_experiment_configs():
    """Generate specific experiment configurations to try."""
    print("\n" + "=" * 60)
    print("RECOMMENDED EXPERIMENT CONFIGURATIONS")
    print("=" * 60)
    
    experiments = [
        {
            'name': 'extended_training',
            'config': {
                'max_iters': 5000,
                'eval_interval': 100,
                'learning_rate': 3e-4,
                'lr_schedule': 'cosine'
            },
            'rationale': 'Test if more training closes the gap'
        },
        {
            'name': 'fast_inference',
            'config': {
                'max_iters': 3000,
                'pcn_inference_steps': 3,
                'pcn_inference_lr': 0.2
            },
            'rationale': 'Faster inference might enable better gradient flow'
        },
        {
            'name': 'deep_inference',
            'config': {
                'max_iters': 3000,
                'pcn_inference_steps': 10,
                'pcn_inference_lr': 0.05
            },
            'rationale': 'Better convergence might improve representations'
        },
        {
            'name': 'larger_model',
            'config': {
                'n_embed': 256,
                'n_layers': 6,
                'max_iters': 3000
            },
            'rationale': 'More capacity might better utilize PCN dynamics'
        },
        {
            'name': 'alternating_arch',
            'config': {
                'model_type': 'alternating',
                'max_iters': 3000
            },
            'rationale': 'Different architecture might be more efficient'
        }
    ]
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nExperiment {i}: {exp['name']}")
        print(f"  Rationale: {exp['rationale']}")
        print(f"  Config changes:")
        for key, value in exp['config'].items():
            print(f"    --{key} {value}")
    
    # Save configs for easy execution
    import json
    with open('pcn_experiments.json', 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"\n✅ Experiment configs saved to pcn_experiments.json")


def main():
    """Run full analysis."""
    # Analyze current state
    convergence_info = analyze_convergence_rate()
    
    # Estimate needs
    estimate_training_needs()
    
    # Suggest improvements
    suggest_improvements()
    
    # Create experiment configs
    create_experiment_configs()
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    
    print("\nBased on the analysis:")
    print("1. ✅ First try extended training (5000 iters) - might close 30-50% of gap")
    print("2. ✅ Simultaneously test hyperparameter variations")
    print("3. ⚠️  If gap remains >0.3 after extended training, implement architectural improvements")
    print("4. 🔬 Consider this a research opportunity - PCN for language is unexplored!")
    
    print("\nThe performance gap is likely due to:")
    print("- PCN's iterative inference adds computational constraints")
    print("- Language modeling might need PCN-specific architectural adaptations")
    print("- Current hyperparameters are not optimized for this task")
    
    print("\nThis is pioneering work - there's no established best practice for PCN in NLP!")


if __name__ == '__main__':
    main()