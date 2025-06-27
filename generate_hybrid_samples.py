"""
Generate samples from trained hybrid PCN-Transformer models.
"""

import torch
import argparse
import os
import sys
sys.path.append('src')

from models.hybrid_architectures import create_hybrid_model
from data_loader import load_data


def generate_samples(checkpoint_path: str, num_samples: int = 5, 
                    max_length: int = 200, temperature: float = 0.8,
                    top_k: int = 40):
    """Generate text samples from a trained model."""
    
    print("Loading model and data...")
    
    # Load data for vocabulary
    _, _, vocab_size, encode, decode = load_data()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # Extract model config from checkpoint path
    model_type = checkpoint_path.split('/')[1]  # Assumes checkpoints/model_type/...
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_hybrid_model(
        model_type=model_type,
        vocab_size=vocab_size,
        batch_size=1,
        block_size=128,  # Must match training
        n_embed=128,
        n_heads=8,
        n_layers=4,
        device=device
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel: {model_type}")
    print(f"Checkpoint iteration: {checkpoint['iter']}")
    print(f"Validation loss: {checkpoint['loss']:.4f}")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("\n" + "="*60)
    
    # Test prompts
    prompts = [
        "To be or not to be",
        "ROMEO:",
        "O, what a",
        "The king",
        "Enter HAMLET",
        "First Citizen:",
        "",  # Empty prompt for unconditional generation
    ]
    
    for i, prompt in enumerate(prompts[:num_samples]):
        print(f"\nSample {i+1}:")
        print(f"Prompt: '{prompt}'")
        print("-" * 40)
        
        # Encode prompt
        if prompt:
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        # Generate with temperature and top-k sampling
        with torch.no_grad():
            for _ in range(max_length - context.size(1)):
                # Get predictions
                logits, _ = model(context)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                context = torch.cat([context, next_token], dim=1)
                
                # Stop if we hit the block size
                if context.size(1) >= 128:
                    break
        
        # Decode and print
        generated_text = decode(context[0].cpu().numpy())
        print(generated_text)
    
    print("\n" + "="*60)
    
    # Show PCN-specific insights
    if 'pcn' in model_type:
        print("\nPCN Insights:")
        print("- This model uses Predictive Coding Networks in its architecture")
        print("- Learning happened through local error minimization")
        print("- The model balances biological plausibility with performance")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained models')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/pcn_ff/20250624_160112/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=7,
                       help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k filtering')
    
    args = parser.parse_args()
    
    generate_samples(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )


if __name__ == '__main__':
    main()