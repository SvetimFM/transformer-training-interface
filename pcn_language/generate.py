"""
Text generation script for PCN language model.
"""

import torch
import argparse
from pathlib import Path

from pcn_lm import PCNLanguageModel
from dataset import TextDataset
from tokenizer import CharacterTokenizer


def generate_text(
    model: PCNLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str = "",
    max_length: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    T_infer: int = 20,
    eta_infer: float = 0.1,
    device: str = 'cuda'
):
    """
    Generate text from a PCN language model.
    
    Args:
        model: Trained PCN language model
        tokenizer: Character tokenizer
        prompt: Starting text
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        T_infer: Inference steps per token
        eta_infer: Inference learning rate
        device: Device to use
        
    Returns:
        Generated text string
    """
    model.eval()
    model.to(device)
    
    # Encode prompt
    if prompt:
        prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    else:
        # Start with a random character
        prompt_tokens = torch.randint(0, tokenizer.vocab_size, (1, 1)).to(device)
    
    print(f"Generating from prompt: '{prompt}'")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            T_infer=T_infer,
            eta_infer=eta_infer
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens[0])
    
    return generated_text


def main(args):
    """Main generation function."""
    
    # Load model checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Create dataset to get vocabulary
    dataset = TextDataset(
        text_source=args.dataset,
        sequence_length=128,  # Doesn't matter for generation
        train=True
    )
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(dataset.chars)
    vocab_size = tokenizer.vocab_size
    
    # Determine model architecture from checkpoint
    # This is a bit hacky but works for our saved models
    state_dict = checkpoint['model_state_dict']
    embed_dim = state_dict['embedding.weight'].shape[1]
    
    # Infer layer dimensions from saved weights
    layer_dims = [embed_dim]
    layer_idx = 0
    while f'layers.{layer_idx}.W' in state_dict:
        layer_dims.append(state_dict[f'layers.{layer_idx}.W'].shape[0])
        layer_idx += 1
    
    print(f"\nModel architecture:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Layer dims: {layer_dims}")
    
    # Create model
    model = PCNLanguageModel(
        vocab_size=vocab_size,
        dims=layer_dims,
        sequence_length=args.max_length
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Generate text
    print(f"\nGenerating text...")
    
    if args.interactive:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            prompt = input("\nEnter prompt (or press Enter for random): ")
            if prompt.lower() == 'quit':
                break
            
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                T_infer=args.T_infer,
                eta_infer=args.eta_infer,
                device=args.device
            )
            
            print("\nGenerated text:")
            print(generated)
            print("-" * 50)
    else:
        # Single generation
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            T_infer=args.T_infer,
            eta_infer=args.eta_infer,
            device=args.device
        )
        
        print("\nGenerated text:")
        print(generated)
        
        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(generated)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from PCN language model")
    
    # Model arguments
    parser.add_argument('checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        help='Dataset used for training (for vocabulary)')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, default="",
                        help='Starting prompt for generation')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling')
    
    # PCN inference arguments
    parser.add_argument('--T_infer', type=int, default=20,
                        help='Inference steps per token')
    parser.add_argument('--eta_infer', type=float, default=0.1,
                        help='Inference learning rate')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save generated text')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive generation mode')
    
    args = parser.parse_args()
    main(args)