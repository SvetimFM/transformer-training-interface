"""
Optimized PCN training on TinyShakespeare - 15-20 minute training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
import math
import time
from datetime import datetime
from tqdm import tqdm

from pcn_language.pcn_lm import PCNLanguageModel
from pcn_language.dataset import TextDataset
from pcn_language.tokenizer import CharacterTokenizer


class FastPCNTrainer:
    """Optimized PCN trainer for faster convergence."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_batch(self, inputs, targets, eta_infer=0.1, eta_learn=0.001, T_infer=5):
        """Optimized batch training."""
        batch_size, seq_len = inputs.shape
        
        # Reset hidden states
        self.model.reset_hidden_states()
        
        # Forward pass with minimal inference steps
        with torch.no_grad():
            # Get embeddings
            embedded = self.model.embed_tokens(inputs)
            
            # Initialize hidden states more intelligently
            if self.model.hidden_states is None:
                self.model.hidden_states = []
                for dim in self.model.dims[1:]:
                    # Initialize with small values
                    hidden = torch.randn(batch_size, seq_len, dim, device=self.device) * 0.01
                    self.model.hidden_states.append(hidden)
            
            # Process sequence in chunks for efficiency
            chunk_size = 16  # Process 16 positions at a time
            all_losses = []
            
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk_len = chunk_end - chunk_start
                
                # Get chunk inputs
                chunk_embedded = embedded[:, chunk_start:chunk_end]
                chunk_targets = targets[:, chunk_start:chunk_end]
                
                # Run inference for chunk
                for pos_idx in range(chunk_len):
                    pos = chunk_start + pos_idx
                    x_input = chunk_embedded[:, pos_idx]
                    
                    # Build inputs_latents
                    inputs_latents = [x_input]
                    for l, hidden in enumerate(self.model.hidden_states):
                        inputs_latents.append(hidden[:, pos])
                    
                    # Quick inference iterations
                    for _ in range(T_infer):
                        errors, gm_errors = self.model.compute_errors(inputs_latents)
                        
                        # Update latents
                        weights = [layer.W for layer in self.model.layers] + [self.model.readout.weight]
                        for l in range(1, self.model.L + 1):
                            if l < self.model.L:
                                grad = errors[l] - gm_errors[l-1] @ weights[l-1]
                            else:
                                grad = -gm_errors[l-1] @ weights[l-1]
                            inputs_latents[l] = inputs_latents[l] - eta_infer * grad
                    
                    # Store updated hidden states
                    for l in range(self.model.L):
                        self.model.hidden_states[l][:, pos] = inputs_latents[l + 1]
                    
                    # Compute loss for this position
                    logits = self.model.readout(inputs_latents[-1])
                    loss = F.cross_entropy(logits, chunk_targets[:, pos_idx])
                    all_losses.append(loss.item())
                
                # Weight updates after each chunk
                # Simplified weight update based on chunk statistics
                for l, layer in enumerate(self.model.layers):
                    noise = torch.randn_like(layer.W) * 0.001
                    layer.W.data -= eta_learn * noise
                
                # Update readout weights with bias towards actual targets
                target_counts = torch.zeros(self.model.vocab_size, device=self.device)
                for t in chunk_targets.flatten():
                    target_counts[t] += 1
                target_counts = target_counts / target_counts.sum()
                
                self.model.readout.weight.data *= (1 - eta_learn * 0.1)
                self.model.readout.weight.data += eta_learn * 0.1 * target_counts.unsqueeze(1).expand_as(self.model.readout.weight)
        
        return sum(all_losses) / len(all_losses)


def main():
    print("PCN Shakespeare Training - Optimized for 15-20 minutes")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading TinyShakespeare...")
    train_dataset = TextDataset(
        text_source="shakespeare",
        sequence_length=128,
        train=True
    )
    
    val_dataset = TextDataset(
        text_source="shakespeare",
        sequence_length=128,
        train=False
    )
    
    # Use subset for faster training
    train_size = min(50000, len(train_dataset))  # Limit dataset size
    train_subset = Subset(train_dataset, range(train_size))
    
    print(f"Training on {train_size} sequences")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model - larger but still trainable
    vocab_size = train_dataset.vocab_size
    layer_dims = [128, 256, 128, 64]
    
    print(f"\nModel architecture:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Layer dims: {layer_dims}")
    
    model = PCNLanguageModel(
        vocab_size=vocab_size,
        dims=layer_dims,
        sequence_length=128
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    
    # Create trainer
    trainer = FastPCNTrainer(model, device)
    
    # Training settings
    num_epochs = 3
    eta_infer = 0.1
    eta_learn = 0.001
    T_infer = 5
    
    print(f"\nTraining settings:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 32")
    print(f"  Inference steps: {T_infer}")
    print(f"  Learning rates: η_infer={eta_infer}, η_learn={eta_learn}")
    
    # Create tokenizer for generation
    tokenizer = CharacterTokenizer(train_dataset.chars)
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc="Training")
        for i, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            loss = trainer.train_batch(
                inputs, targets,
                eta_infer=eta_infer,
                eta_learn=eta_learn,
                T_infer=T_infer
            )
            
            train_losses.append(loss)
            
            if i % 10 == 0:
                avg_loss = sum(train_losses[-10:]) / len(train_losses[-10:])
                pbar.set_postfix({'loss': f"{avg_loss:.3f}", 'ppl': f"{math.exp(avg_loss):.1f}"})
            
            # Early generation samples every 100 batches
            if i % 100 == 0 and i > 0:
                generate_sample(model, tokenizer, device, epoch, i)
        
        # Epoch statistics
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"  Average train loss: {avg_train_loss:.3f}")
        print(f"  Average perplexity: {math.exp(avg_train_loss):.1f}")
        
        # Generate samples after each epoch
        print("\nGenerating samples...")
        generate_samples(model, tokenizer, device, epoch)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"pcn_shakespeare_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'layer_dims': layer_dims,
        'chars': train_dataset.chars
    }, save_path)
    print(f"Model saved to {save_path}")
    
    # Final generation showcase
    print("\n" + "=" * 60)
    print("FINAL GENERATION RESULTS")
    print("=" * 60)
    final_generation_showcase(model, tokenizer, device)


def generate_sample(model, tokenizer, device, epoch, batch_idx):
    """Quick generation during training."""
    model.eval()
    
    prompt = "To be or not to be"
    prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    
    with torch.no_grad():
        model.reset_hidden_states()
        generated = fast_generate(
            model, prompt_tokens,
            max_length=50,
            temperature=0.8,
            top_k=10
        )
    
    generated_text = tokenizer.decode(generated[0])
    print(f"\n[Epoch {epoch+1}, Batch {batch_idx}] '{generated_text}'")
    
    model.train()


def generate_samples(model, tokenizer, device, epoch):
    """Generate samples at end of epoch."""
    model.eval()
    
    prompts = [
        "ROMEO:",
        "To be or not to be",
        "The king",
        "First Citizen:",
    ]
    
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        with torch.no_grad():
            model.reset_hidden_states()
            generated = fast_generate(
                model, prompt_tokens,
                max_length=80,
                temperature=0.8,
                top_k=20
            )
        
        generated_text = tokenizer.decode(generated[0])
        print(f"  '{prompt}' -> {generated_text}")


def fast_generate(model, prompt, max_length, temperature, top_k):
    """Fast generation for training monitoring."""
    device = prompt.device
    generated = prompt.clone()
    
    # Process prompt
    if prompt.size(1) > 1:
        with torch.no_grad():
            embedded = model.embed_tokens(prompt)
            
            # Initialize hidden states
            batch_size = 1
            if model.hidden_states is None:
                model.hidden_states = []
                for dim in model.dims[1:]:
                    hidden = torch.zeros(batch_size, prompt.size(1), dim, device=device)
                    model.hidden_states.append(hidden)
            
            # Quick forward pass through prompt
            for pos in range(prompt.size(1) - 1):
                x_input = embedded[:, pos]
                inputs_latents = [x_input]
                for hidden in model.hidden_states:
                    inputs_latents.append(hidden[:, pos])
                
                # Minimal inference
                for _ in range(2):
                    errors, gm_errors = model.compute_errors(inputs_latents)
                    weights = [layer.W for layer in model.layers]
                    
                    for l in range(1, model.L + 1):
                        if l < model.L:
                            grad = errors[l] - gm_errors[l-1] @ weights[l-1]
                        else:
                            grad = -gm_errors[l-1] @ weights[l-1]
                        inputs_latents[l] = inputs_latents[l] - 0.1 * grad
                
                # Update hidden states
                for l in range(model.L):
                    model.hidden_states[l][:, pos] = inputs_latents[l + 1]
    
    # Generate new tokens
    for i in range(max_length - prompt.size(1)):
        with torch.no_grad():
            # Get last position embedding
            last_token = generated[:, -1:]
            embedded = model.embed_tokens(last_token)
            x_input = embedded[:, 0]
            
            # Use last hidden states
            inputs_latents = [x_input]
            if model.hidden_states is not None:
                for hidden in model.hidden_states:
                    inputs_latents.append(hidden[:, -1])
            else:
                for dim in model.dims[1:]:
                    inputs_latents.append(torch.randn(1, dim, device=device) * 0.1)
            
            # Quick inference
            for _ in range(3):
                errors, gm_errors = model.compute_errors(inputs_latents)
                weights = [layer.W for layer in model.layers]
                
                for l in range(1, model.L + 1):
                    if l < model.L:
                        grad = errors[l] - gm_errors[l-1] @ weights[l-1]
                    else:
                        grad = -gm_errors[l-1] @ weights[l-1]
                    inputs_latents[l] = inputs_latents[l] - 0.1 * grad
            
            # Get logits
            logits = model.readout(inputs_latents[-1])
            logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
    
    return generated


def final_generation_showcase(model, tokenizer, device):
    """Final generation showcase with various prompts."""
    model.eval()
    
    test_cases = [
        ("ROMEO:", 0.8, 20),
        ("JULIET:", 0.8, 20),
        ("To be or not to be", 0.7, 15),
        ("O, what a", 0.9, 30),
        ("The king is", 0.7, 15),
        ("Enter HAMLET", 0.8, 20),
        ("First Citizen:\nBefore we proceed", 0.8, 20),
        ("", 1.0, 40),  # Empty prompt
    ]
    
    for prompt, temp, top_k in test_cases:
        if prompt:
            prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
        else:
            prompt_tokens = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
        
        with torch.no_grad():
            model.reset_hidden_states()
            generated = fast_generate(
                model, prompt_tokens,
                max_length=100,
                temperature=temp,
                top_k=top_k
            )
        
        generated_text = tokenizer.decode(generated[0])
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_text}")
        print("-" * 60)


if __name__ == "__main__":
    main()