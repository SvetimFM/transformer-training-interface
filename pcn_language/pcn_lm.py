"""
PCN Language Model - Predictive Coding Network for text generation.

This adapts the PCN architecture for sequential language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

import sys
sys.path.append('..')
from pcn_model.layers import PCNLayer
from pcn_model.network import PredictiveCodingNetwork


class PCNLanguageModel(PredictiveCodingNetwork):
    """
    PCN adapted for language modeling with sequential processing.
    
    Key modifications:
    - Embedding layer for token inputs
    - Sequential processing with hidden state
    - Autoregressive generation capabilities
    """
    
    def __init__(
        self,
        vocab_size: int,
        dims: List[int],
        sequence_length: int = 128,
        embed_dim: Optional[int] = None
    ):
        """
        Initialize PCN language model.
        
        Args:
            vocab_size: Size of token vocabulary
            dims: Layer dimensions [embed_dim, h1, h2, ..., hL]
            sequence_length: Maximum sequence length
            embed_dim: Embedding dimension (if None, uses dims[0])
        """
        # Set embedding dimension
        if embed_dim is None:
            embed_dim = dims[0]
        
        # Initialize base PCN with embedding dimension as input
        super().__init__(dims=dims, output_dim=vocab_size)
        
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # Add embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        # Add positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(sequence_length, embed_dim))
        
        # Hidden states for sequential processing
        self.hidden_states = None
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens with positional encoding.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (batch_size, seq_len, embed_dim)
        """
        # Get token embeddings
        embedded = self.embedding(tokens)
        
        # Add positional encoding
        seq_len = tokens.size(1)
        embedded = embedded + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        return embedded
    
    def init_hidden_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize hidden states for sequential processing."""
        # Initialize latents for each position
        hidden = []
        for dim in self.dims[1:]:  # Skip input dimension
            hidden.append(torch.zeros(batch_size, self.sequence_length, dim, device=device))
        return hidden
    
    def process_sequence(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        T_infer: int = 20,
        eta_infer: float = 0.1
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Process a sequence through the PCN.
        
        Args:
            tokens: Input tokens of shape (batch_size, seq_len)
            targets: Target tokens for training (batch_size, seq_len)
            T_infer: Number of inference steps per position
            eta_infer: Inference learning rate
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            energies: Energy values during processing
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Embed tokens
        embedded = self.embed_tokens(tokens)  # (B, L, D)
        
        # Initialize hidden states if needed
        if self.hidden_states is None:
            self.hidden_states = self.init_hidden_states(batch_size, device)
        
        # Process each position sequentially
        all_logits = []
        all_energies = []
        
        for pos in range(seq_len):
            # Get input for current position
            x_input = embedded[:, pos, :]  # (B, D)
            
            # Initialize latents for this position
            inputs_latents = [x_input]
            for l, hidden in enumerate(self.hidden_states):
                # Use hidden state from this position
                inputs_latents.append(hidden[:, pos, :].clone())
            
            # Get weight references
            weights = [layer.W for layer in self.layers] + [self.readout.weight]
            
            # Run inference for this position
            position_energies = []
            for t in range(T_infer):
                # Compute errors
                errors, gain_modulated_errors = self.compute_errors(inputs_latents)
                
                # Compute output and supervised error if targets provided
                y_hat = self.readout(inputs_latents[-1])
                
                if targets is not None:
                    target_pos = F.one_hot(targets[:, pos], num_classes=self.vocab_size).float()
                    eps_sup = y_hat - target_pos
                    eps_L = eps_sup @ weights[-1]
                else:
                    eps_L = torch.zeros_like(inputs_latents[-1])
                
                errors_extended = errors + [eps_L]
                
                # Track energy
                total_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors) / batch_size
                if targets is not None:
                    total_energy += 0.5 * eps_sup.pow(2).sum().item() / batch_size
                position_energies.append(total_energy)
                
                # Update latents
                for l in range(1, self.L + 1):
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                    inputs_latents[l] = inputs_latents[l] - eta_infer * grad_Xl
            
            # Store final hidden states
            for l, hidden in enumerate(self.hidden_states):
                hidden[:, pos, :] = inputs_latents[l + 1].detach()
            
            # Get final output for this position
            final_logits = self.readout(inputs_latents[-1])
            all_logits.append(final_logits)
            all_energies.extend(position_energies)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # (B, L, V)
        
        return logits, all_energies
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        T_infer: int = 20,
        eta_infer: float = 0.1
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Starting tokens of shape (1, prompt_length)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (None for full distribution)
            T_infer: Inference steps per token
            eta_infer: Inference learning rate
            
        Returns:
            Generated token sequence
        """
        self.eval()
        device = prompt.device
        
        # Initialize with prompt
        generated = prompt.clone()
        self.hidden_states = None
        
        # Process prompt to initialize hidden states
        if prompt.size(1) > 0:
            self.process_sequence(prompt, T_infer=T_infer, eta_infer=eta_infer)
        
        # Generate tokens one by one
        for _ in range(max_length - prompt.size(1)):
            # Get last token
            last_token = generated[:, -1:]
            
            # Process through PCN
            logits, _ = self.process_sequence(last_token, T_infer=T_infer, eta_infer=eta_infer)
            logits = logits[:, -1, :] / temperature  # (1, V)
            
            # Apply top-k if specified
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def reset_hidden_states(self):
        """Reset hidden states for new sequence."""
        self.hidden_states = None