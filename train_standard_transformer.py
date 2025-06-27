"""
Train a standard transformer for comparison with PCN hybrids.
"""

import torch
import numpy as np
import time
import os
from datetime import datetime

import sys
sys.path.append('src')
sys.path.insert(0, '.')

from models.bigram import BigramLM
from data_loader import load_data, get_batch


def train_standard_transformer():
    """Train standard transformer with same settings as PCN hybrid."""
    
    print("Training Standard Transformer (Baseline)")
    print("=" * 60)
    
    # Settings matching PCN training
    n_embed = 128
    n_heads = 8
    n_layers = 4
    batch_size = 32
    block_size = 128
    learning_rate = 3e-4
    max_iters = 2000
    eval_interval = 100
    dropout = 0.2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("\nLoading TinyShakespeare dataset...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Create model config
    config = type('Config', (), {
        'model': type('ModelConfig', (), {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre'
        })(),
        'training': type('TrainingConfig', (), {
            'device': device
        })()
    })()
    
    # Create model
    print("\nCreating standard transformer...")
    model = BigramLM(
        vocab_size=vocab_size,
        batch_size=batch_size,
        block_size=block_size,
        config=config
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=learning_rate * 0.1
    )
    
    # Training
    print(f"\nStarting training for {max_iters} iterations...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create save directory
    save_dir = os.path.join('checkpoints', 'standard', 
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    for iter in range(max_iters):
        # Sample batch
        xb, yb = get_batch('train', train_data, block_size, batch_size, device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        # Evaluation
        if iter % eval_interval == 0:
            model.eval()
            
            # Validation loss
            val_loss_sum = 0
            for _ in range(50):
                xb_val, yb_val = get_batch('val', val_data, block_size, batch_size, device)
                _, val_loss = model(xb_val, yb_val)
                val_loss_sum += val_loss.item()
            
            val_loss = val_loss_sum / 50
            val_losses.append(val_loss)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Iter {iter}/{max_iters} | "
                  f"Time: {elapsed/60:.1f}m | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best.pt'))
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            
            # Generate sample
            if iter % 200 == 0:
                model.eval()
                context = torch.tensor(encode("To be or not to be"), 
                                     dtype=torch.long, device=device).unsqueeze(0)
                generated = model.generate(context, max_new_tokens=100)
                print(f"\nSample: {decode(generated[0].cpu().numpy())}")
            
            model.train()
    
    # Training complete
    elapsed_total = time.time() - start_time
    print(f"\nTraining completed in {elapsed_total/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    checkpoint = {
        'iter': max_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_losses[-1],
    }
    torch.save(checkpoint, os.path.join(save_dir, 'final.pt'))
    
    return model, best_val_loss


if __name__ == '__main__':
    model, best_loss = train_standard_transformer()