"""
PCN-based FeedForward layer - Scientifically accurate implementation.

This implements a feedforward network using Predictive Coding principles,
where each layer predicts the activity of the layer below, and learning
happens through local error minimization.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List, Tuple

import sys
sys.path.append('../..')
from pcn_model.layers import PCNLayer


class PCNFeedForward(nn.Module):
    """
    PCN-based feedforward network for transformer blocks.
    
    Implements a hierarchical predictive coding network that replaces
    the standard two-layer FFN in transformers. The key difference is
    that this uses local learning rules and iterative inference.
    
    Architecture:
    - Input (n_embed) -> Latent (expanded) -> Output (n_embed)
    - Each layer predicts the one below
    - Inference minimizes prediction errors
    
    Args:
        n_embed: Input/output dimension (transformer hidden size)
        dropout: Dropout probability
        expansion_factor: How much to expand the hidden dimension
        inference_steps: Number of inference iterations per forward pass
        inference_lr: Learning rate for latent updates during inference
    """
    
    def __init__(
        self, 
        n_embed: int, 
        dropout: float = 0.2,
        expansion_factor: int = 4,
        inference_steps: int = 5,
        inference_lr: float = 0.1,
        **kwargs  # Ignore extra kwargs for compatibility
    ):
        super().__init__()
        
        self.n_embed = n_embed
        self.hidden_dim = n_embed * expansion_factor
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        
        # In PCN, we need generative weights that predict downward
        # Layer 1: Predicts input from hidden representation
        self.W_1 = nn.Parameter(torch.randn(n_embed, self.hidden_dim) * 0.02)
        
        # Layer 2: Maps from top-level representation to hidden
        # This is like the "readout" that generates the hidden layer
        self.W_2 = nn.Parameter(torch.randn(self.hidden_dim, n_embed) * 0.02)
        
        # Biases
        self.b_1 = nn.Parameter(torch.zeros(n_embed))
        self.b_2 = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Activation function and its derivative
        self.activation = F.relu
        self.activation_deriv = lambda x: (x > 0).float()
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using PCN inference.
        
        Unlike standard feedforward, this:
        1. Initializes latent representations
        2. Runs iterative inference to minimize prediction errors
        3. Uses the final latent state to generate output
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Apply layer norm to input
        x_norm = self.layer_norm(x)
        
        # Initialize latent representations
        # z_hidden: Hidden layer representation
        z_hidden = torch.randn(batch_size, seq_len, self.hidden_dim, device=device) * 0.1
        z_hidden.requires_grad = False  # We'll update manually
        
        # z_top: Top-level representation (what generates the output)
        z_top = torch.randn(batch_size, seq_len, self.n_embed, device=device) * 0.1
        z_top.requires_grad = False
        
        # PCN Inference: Minimize prediction errors through local updates
        for _ in range(self.inference_steps):
            # 1. Generate predictions
            # Top layer generates hidden layer
            hidden_pred = self.activation(F.linear(z_top, self.W_2, self.b_2))
            
            # Hidden layer generates input reconstruction
            # W_1 is (n_embed, hidden_dim), F.linear expects (out_features, in_features)
            input_pred = F.linear(z_hidden, self.W_1, self.b_1)
            
            # 2. Compute prediction errors
            e_input = x_norm - input_pred  # Error at input level
            e_hidden = z_hidden - hidden_pred  # Error at hidden level
            
            # 3. Compute gradients using local learning rules
            # Gradient for hidden layer
            grad_hidden = -e_hidden  # Error from above
            # For the error from below, we need to backprop through W_1
            # e_input has shape (batch, seq, n_embed), W_1 has shape (n_embed, hidden_dim)
            # We want gradient w.r.t. z_hidden, so we need W_1.t()
            grad_hidden = grad_hidden + F.linear(e_input, self.W_1.t())  # Error from below
            
            # Gradient for top layer
            # Backprop through activation function
            g_hidden = e_hidden * self.activation_deriv(F.linear(z_top, self.W_2, self.b_2))
            grad_top = -F.linear(g_hidden, self.W_2.t())
            
            # 4. Update latents
            z_hidden = z_hidden - self.inference_lr * grad_hidden
            z_top = z_top - self.inference_lr * grad_top
        
        # After inference, the top-level representation z_top contains
        # the processed information. In standard FFN terms, this is like
        # having done: input -> expand -> contract, but through inference
        
        # Apply dropout and return
        output = self.dropout(z_top)
        
        # Add residual connection (handled by transformer block)
        return output
    
    def compute_energy(self, x: torch.Tensor, z_hidden: torch.Tensor, 
                      z_top: torch.Tensor) -> torch.Tensor:
        """
        Compute the total prediction error (energy) of the current state.
        
        This can be used for analysis or as an auxiliary loss.
        """
        # Generate predictions
        hidden_pred = self.activation(F.linear(z_top, self.W_2, self.b_2))
        input_pred = F.linear(z_hidden, self.W_1.t(), self.b_1)
        
        # Compute errors
        e_input = x - input_pred
        e_hidden = z_hidden - hidden_pred
        
        # Total energy is sum of squared errors
        energy = 0.5 * (e_input.pow(2).sum() + e_hidden.pow(2).sum())
        
        return energy / (x.shape[0] * x.shape[1])  # Normalize by batch and sequence


class PCNFeedForwardWithMemory(PCNFeedForward):
    """
    Extended version that maintains hidden states across forward passes.
    
    This allows the PCN to build up representations over time,
    similar to how biological neural networks maintain state.
    """
    
    def __init__(self, *args, momentum: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.register_buffer('z_hidden_memory', None)
        self.register_buffer('z_top_memory', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize or retrieve latent representations
        if self.z_hidden_memory is None or self.z_hidden_memory.shape[0] != batch_size:
            z_hidden = torch.randn(batch_size, seq_len, self.hidden_dim, device=device) * 0.1
            z_top = torch.randn(batch_size, seq_len, self.n_embed, device=device) * 0.1
        else:
            # Use momentum to blend old and new
            z_hidden = self.momentum * self.z_hidden_memory + \
                      (1 - self.momentum) * torch.randn(batch_size, seq_len, self.hidden_dim, device=device) * 0.1
            z_top = self.momentum * self.z_top_memory + \
                   (1 - self.momentum) * torch.randn(batch_size, seq_len, self.n_embed, device=device) * 0.1
        
        # Run standard forward pass
        x_norm = self.layer_norm(x)
        
        # PCN Inference
        for _ in range(self.inference_steps):
            # Generate predictions
            hidden_pred = self.activation(F.linear(z_top, self.W_2, self.b_2))
            input_pred = F.linear(z_hidden, self.W_1, self.b_1)
            
            # Compute errors
            e_input = x_norm - input_pred
            e_hidden = z_hidden - hidden_pred
            
            # Compute gradients
            grad_hidden = -e_hidden + F.linear(e_input, self.W_1.t())
            g_hidden = e_hidden * self.activation_deriv(F.linear(z_top, self.W_2, self.b_2))
            grad_top = -F.linear(g_hidden, self.W_2.t())
            
            # Update latents
            z_hidden = z_hidden - self.inference_lr * grad_hidden
            z_top = z_top - self.inference_lr * grad_top
        
        # Store latents for next forward pass
        self.z_hidden_memory = z_hidden.detach()
        self.z_top_memory = z_top.detach()
        
        return self.dropout(z_top)