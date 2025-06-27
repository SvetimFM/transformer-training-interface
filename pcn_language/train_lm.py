"""
Training script for PCN language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import argparse
from tqdm import tqdm
import os
import math
from datetime import datetime
from typing import Optional

from pcn_lm import PCNLanguageModel
from dataset import TextDataset
from tokenizer import CharacterTokenizer


class PCNLanguageTrainer:
    """Trainer for PCN language models."""
    
    def __init__(
        self,
        model: PCNLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eta_infer: float = 0.1,
        eta_learn: float = 0.01,
        T_infer: int = 20,
        T_learn: int = 50,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta_infer = eta_infer
        self.eta_learn = eta_learn
        self.T_infer = T_infer
        self.T_learn = T_learn
        self.device = torch.device(device)
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size, seq_len = inputs.shape
            
            # Reset hidden states for new batch
            self.model.reset_hidden_states()
            
            # Process sequence
            logits, energies = self.model.process_sequence(
                inputs, targets, 
                T_infer=self.T_infer, 
                eta_infer=self.eta_infer
            )
            
            # Get weights for learning phase
            weights = [layer.W for layer in self.model.layers] + [self.model.readout.weight]
            
            # Learning phase - update weights based on final states
            for t in range(self.T_learn):
                # Recompute with current weights
                with torch.no_grad():
                    # Process full sequence again to get errors
                    embedded = self.model.embed_tokens(inputs)
                    
                    batch_errors = []
                    batch_gm_errors = []
                    batch_inputs_latents = []
                    
                    for pos in range(seq_len):
                        x_input = embedded[:, pos, :]
                        inputs_latents = [x_input]
                        
                        # Use stored hidden states
                        for l, hidden in enumerate(self.model.hidden_states):
                            inputs_latents.append(hidden[:, pos, :])
                        
                        # Compute errors for this position
                        errors, gm_errors = self.model.compute_errors(inputs_latents)
                        
                        # Add supervised error
                        y_hat = self.model.readout(inputs_latents[-1])
                        target_pos = F.one_hot(targets[:, pos], num_classes=self.model.vocab_size).float()
                        eps_sup = y_hat - target_pos
                        
                        batch_errors.append(errors)
                        batch_gm_errors.append(gm_errors)
                        batch_inputs_latents.append(inputs_latents)
                    
                    # Update weights based on accumulated gradients
                    for l in range(self.model.L):
                        grad_sum = torch.zeros_like(weights[l])
                        for pos in range(seq_len):
                            grad = -(batch_gm_errors[pos][l].T @ batch_inputs_latents[pos][l+1]) / batch_size
                            grad_sum += grad / seq_len
                        weights[l] -= self.eta_learn * grad_sum
                    
                    # Update readout weights
                    grad_out = torch.zeros_like(weights[-1])
                    for pos in range(seq_len):
                        y_hat = self.model.readout(batch_inputs_latents[pos][-1])
                        target_pos = F.one_hot(targets[:, pos], num_classes=self.model.vocab_size).float()
                        eps_sup = y_hat - target_pos
                        grad_out += eps_sup.T @ batch_inputs_latents[pos][-1] / batch_size
                    weights[-1] -= self.eta_learn * grad_out / seq_len
            
            # Compute loss for logging
            with torch.no_grad():
                logits, _ = self.model.process_sequence(inputs, targets, T_infer=1, eta_infer=0)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                total_loss += loss.item() * batch_size * seq_len
                total_tokens += batch_size * seq_len
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'ppl': math.exp(loss.item())})
        
        avg_loss = total_loss / total_tokens
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for inputs, targets in tqdm(self.val_loader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size, seq_len = inputs.shape
            
            # Reset hidden states
            self.model.reset_hidden_states()
            
            # Process sequence
            logits, _ = self.model.process_sequence(
                inputs, targets, 
                T_infer=self.T_infer, 
                eta_infer=self.eta_infer
            )
            
            # Compute loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
        
        avg_loss = total_loss / total_tokens
        return avg_loss
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            train_ppl = math.exp(train_loss)
            print(f"Train Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")
            
            # Evaluate
            val_loss = self.evaluate()
            val_ppl = math.exp(val_loss)
            print(f"Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"Saved best model (val_ppl: {val_ppl:.2f})")


def main(args):
    """Main training function."""
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = TextDataset(
        text_source=args.dataset,
        sequence_length=args.sequence_length,
        train=True
    )
    
    val_dataset = TextDataset(
        text_source=args.dataset,
        sequence_length=args.sequence_length,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    vocab_size = train_dataset.vocab_size
    layer_dims = [args.embed_dim, args.hidden_dim, args.hidden_dim // 2, args.hidden_dim // 4]
    
    print(f"\nCreating PCN language model:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Layer dims: {layer_dims}")
    print(f"  Sequence length: {args.sequence_length}")
    
    model = PCNLanguageModel(
        vocab_size=vocab_size,
        dims=layer_dims,
        sequence_length=args.sequence_length
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = PCNLanguageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        eta_infer=args.eta_infer,
        eta_learn=args.eta_learn,
        T_infer=args.T_infer,
        T_learn=args.T_learn,
        device=args.device
    )
    
    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"pcn_lm_{args.dataset}_{timestamp}.pth"
    
    print(f"\nStarting training...")
    trainer.train(args.num_epochs, save_path)
    
    print(f"\nTraining completed! Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCN language model")
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        help='Dataset: "shakespeare" or path to text file')
    parser.add_argument('--sequence_length', type=int, default=128,
                        help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--eta_infer', type=float, default=0.1,
                        help='Inference learning rate')
    parser.add_argument('--eta_learn', type=float, default=0.01,
                        help='Weight learning rate')
    parser.add_argument('--T_infer', type=int, default=20,
                        help='Inference steps per position')
    parser.add_argument('--T_learn', type=int, default=50,
                        help='Learning steps per batch')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    main(args)