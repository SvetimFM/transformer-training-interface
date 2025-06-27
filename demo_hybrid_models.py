"""
Quick demonstration of Hybrid PCN-Transformer models.

Shows how to:
1. Create different hybrid architectures
2. Train them briefly
3. Generate text
4. Compare their characteristics
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# Add src to path
import sys
sys.path.append('src')

from models.hybrid_architectures import (
    PCNFeedForwardTransformer,
    AlternatingPCNTransformer,
    HierarchicalPCNTransformer,
    DualStreamPCNTransformer,
    PCNPositionalTransformer,
    create_hybrid_model
)
from data_loader import load_data, get_batch


def demonstrate_architecture(model_type: str, vocab_size: int, device: str = 'cuda'):
    """Demonstrate a specific hybrid architecture."""
    print(f"\n{'='*60}")
    print(f"Demonstrating: {model_type.upper()}")
    print(f"{'='*60}")
    
    # Create model
    model = create_hybrid_model(
        model_type=model_type,
        vocab_size=vocab_size,
        batch_size=16,
        block_size=64,
        n_embed=128,
        n_heads=8,
        n_layers=4,
        device=device
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    pcn_params = sum(p.numel() for n, p in model.named_parameters() if 'pcn' in n.lower())
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  PCN parameters: {pcn_params:,} ({pcn_params/total_params*100:.1f}%)")
    
    # Architecture-specific insights
    insights = {
        'pcn_ff': "Replaces feedforward with PCN - biological local learning",
        'alternating': "Alternates attention/PCN - balanced global/local",
        'hierarchical': "PCN→Transformer - biological feature extraction",
        'dual_stream': "Parallel paths - adaptive processing",
        'pcn_positional': "PCN positions - learned position encoding"
    }
    
    print(f"\nKey Feature: {insights.get(model_type, 'Unknown')}")
    
    return model


def quick_train_and_generate(model, train_data, decode, device, num_iters=100):
    """Quick training and generation demo."""
    print("\nQuick Training Demo:")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    
    start_time = time.time()
    
    for i in range(num_iters):
        # Get batch
        xb, yb = get_batch('train', train_data, 64, 16, device)
        
        # Forward
        logits, loss = model(xb, yb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1}: Loss = {loss.item():.4f}")
    
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.2f}s ({train_time/num_iters*1000:.1f}ms/iter)")
    
    # Generate sample
    print("\nGenerating text:")
    model.eval()
    
    prompts = ["To be", "The king", "O, "]
    
    for prompt in prompts:
        # Encode prompt
        prompt_ids = train_data[:len(prompt)]  # Simplified encoding
        context = prompt_ids.unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=30)
        
        # Decode (simplified)
        text = decode(generated[0].cpu().numpy())
        print(f"  '{prompt}...' → {text[:50]}...")
    
    model.train()
    return losses


def compare_architectures():
    """Compare all hybrid architectures."""
    print("\nHybrid PCN-Transformer Architecture Comparison")
    print("=" * 60)
    
    # Load data
    print("\nLoading TinyShakespeare...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test each architecture
    architectures = ['pcn_ff', 'alternating', 'hierarchical', 'dual_stream', 'pcn_positional']
    results = {}
    
    for arch in architectures:
        model = demonstrate_architecture(arch, vocab_size, device)
        losses = quick_train_and_generate(model, train_data, decode, device, num_iters=50)
        
        results[arch] = {
            'final_loss': losses[-1],
            'params': sum(p.numel() for p in model.parameters())
        }
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<20} {'Parameters':>12} {'Final Loss':>12}")
    print("-" * 44)
    
    for arch, res in results.items():
        print(f"{arch:<20} {res['params']:>12,} {res['final_loss']:>12.4f}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("• For drop-in replacement: Use PCN-FF")
    print("• For efficiency: Use Alternating")
    print("• For biological realism: Use Hierarchical")
    print("• For best performance: Use Dual-Stream")
    print("• For position-sensitive tasks: Use PCN-Positional")


def main():
    """Run the demonstration."""
    print("Hybrid PCN-Transformer Models Demo")
    print("==================================")
    
    # Show individual architectures
    print("\n1. Individual Architecture Demos")
    
    # Simple vocab for demo
    vocab_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and show each architecture
    architectures = {
        'pcn_ff': "Standard Transformer + PCN FeedForward",
        'alternating': "Alternating Attention/PCN Layers",
        'hierarchical': "Hierarchical PCN + Transformer",
        'dual_stream': "Dual Stream Processing",
        'pcn_positional': "PCN Positional Encoding"
    }
    
    for arch, description in architectures.items():
        print(f"\n{arch.upper()}: {description}")
        model = create_hybrid_model(
            model_type=arch,
            vocab_size=vocab_size,
            batch_size=8,
            block_size=32,
            n_embed=64,
            n_heads=4,
            n_layers=2
        )
        print(f"  Created successfully! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Full comparison
    print("\n2. Full Architecture Comparison on TinyShakespeare")
    compare_architectures()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Train a model: python train_hybrid_pcn_transformer.py --model_type pcn_ff")
    print("2. Compare all: python compare_architectures.py")
    print("3. Visualize: python visualize_pcn_transformer.py --model_type hierarchical")


if __name__ == '__main__':
    main()