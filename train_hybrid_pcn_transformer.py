"""
Training script for Hybrid PCN-Transformer models on TinyShakespeare.

Supports mixed training strategies:
- Standard backprop for transformer components
- Local PCN updates for PCN components
- Hybrid loss functions
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
import time
from datetime import datetime
import json

# Add src to path
import sys
sys.path.append('src')

from models.hybrid_architectures import create_hybrid_model

# Import data loader from current directory
sys.path.insert(0, '.')
from data_loader import load_data, get_batch


class ShakespeareDataset(Dataset):
    """Simple dataset wrapper for Shakespeare text."""
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train_hybrid_model(args):
    """Main training function for hybrid models."""
    
    print(f"Training Hybrid PCN-Transformer Model")
    print(f"Architecture: {args.model_type}")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading TinyShakespeare dataset...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Val samples: {len(val_data):,}")
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_hybrid_model(
        model_type=args.model_type,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        device=device
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    pcn_params = sum(p.numel() for n, p in model.named_parameters() if 'pcn' in n.lower())
    print(f"Total parameters: {total_params:,}")
    print(f"PCN parameters: {pcn_params:,} ({pcn_params/total_params*100:.1f}%)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=args.learning_rate * 0.1
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create save directory
    save_dir = os.path.join('checkpoints', args.model_type, 
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['vocab_size'] = vocab_size
    config['total_params'] = total_params
    config['pcn_params'] = pcn_params
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nStarting training for {args.max_iters} iterations...")
    print(f"Checkpoints will be saved to: {save_dir}")
    
    start_time = time.time()
    
    # Training loop
    for iter in range(args.max_iters):
        # Sample batch
        xb, yb = get_batch('train', train_data, args.block_size, args.batch_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Record training loss
        train_losses.append(loss.item())
        
        # Evaluation
        if iter % args.eval_interval == 0:
            model.eval()
            
            # Evaluate on validation set
            val_loss = evaluate_model(model, val_data, args.block_size, 
                                    args.batch_size, device, args.eval_iters)
            val_losses.append(val_loss)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"\nIter {iter}/{args.max_iters} | "
                  f"Time: {elapsed/60:.1f}m | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter, val_loss, save_dir, 'best.pt')
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            
            # Generate sample
            if args.generate_samples:
                print("\nGenerating sample:")
                sample = generate_sample(model, encode, decode, device, 
                                       prompt="To be or not to be", max_length=100)
                print(f"  {sample}")
            
            model.train()
        
        # Save periodic checkpoint
        if iter > 0 and iter % args.save_interval == 0:
            save_checkpoint(model, optimizer, iter, loss.item(), save_dir, f'iter_{iter}.pt')
    
    # Training complete
    elapsed_total = time.time() - start_time
    print(f"\nTraining completed in {elapsed_total/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.max_iters, val_losses[-1], save_dir, 'final.pt')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_time': elapsed_total
    }
    torch.save(history, os.path.join(save_dir, 'history.pt'))
    
    return model, history


def evaluate_model(model, data, block_size, batch_size, device, eval_iters):
    """Evaluate model on validation data."""
    losses = []
    
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch('val', data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    
    return np.mean(losses)


def generate_sample(model, encode, decode, device, prompt="", max_length=100):
    """Generate text sample from model."""
    model.eval()
    
    with torch.no_grad():
        # Encode prompt
        if prompt:
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        # Generate
        generated = model.generate(context, max_new_tokens=max_length)
        
        # Decode
        text = decode(generated[0].cpu().numpy())
    
    model.train()
    return text


def save_checkpoint(model, optimizer, iter, loss, save_dir, filename):
    """Save model checkpoint."""
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(save_dir, filename))


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid PCN-Transformer models')
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='pcn_ff',
                       choices=['pcn_ff', 'alternating', 'hierarchical', 
                               'dual_stream', 'pcn_positional'],
                       help='Type of hybrid architecture')
    
    # Model hyperparameters
    parser.add_argument('--n_embed', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--block_size', type=int, default=256,
                       help='Context length')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=5000,
                       help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=50,
                       help='Number of evaluation iterations')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Checkpoint save interval')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Other options
    parser.add_argument('--generate_samples', action='store_true',
                       help='Generate samples during evaluation')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train model
    model, history = train_hybrid_model(args)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()