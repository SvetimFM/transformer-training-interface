"""
PCN-based FeedForward layer as a drop-in replacement for standard transformer FFN.

This module implements a Predictive Coding Network version of the feedforward
layer, maintaining the same interface as the standard FeedForward but using
hierarchical predictive processing internally.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List

import sys
sys.path.append('../..')
from pcn_model.layers import PCNLayer


class PCNFeedForward(nn.Module):
    """
    PCN-based feedforward network for transformer blocks.
    
    Instead of two linear layers with ReLU, this uses a 2-3 layer PCN
    that performs local inference to compute the output.
    
    Args:
        n_embed: Input/output dimension (transformer hidden size)
        dropout: Dropout probability
        n_pcn_layers: Number of PCN layers (default 2)
        inference_steps: Number of inference iterations per forward pass
        inference_lr: Learning rate for latent updates during inference
    """
    
    def __init__(
        self, 
        n_embed: int, 
        dropout: float = 0.2,
        n_pcn_layers: int = 2,
        inference_steps: int = 5,
        inference_lr: float = 0.1
    ):
        super().__init__()
        
        self.n_embed = n_embed
        self.dropout_prob = dropout
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        
        # Define PCN layer dimensions
        # For PCN, we need to think about the hierarchy differently
        # We have input -> hidden -> output, where each layer predicts the one below
        if n_pcn_layers == 2:
            # Two PCN layers: one expands, one contracts
            dims = [n_embed, 4 * n_embed]  # Just two levels in hierarchy
        elif n_pcn_layers == 3:
            # Three PCN layers with intermediate expansion
            dims = [n_embed, 2 * n_embed, 4 * n_embed]
        else:
            raise ValueError(f"n_pcn_layers must be 2 or 3, got {n_pcn_layers}")
        
        self.dims = dims
        self.n_layers = len(dims) - 1
        
        # Build PCN hierarchy
        # Each layer predicts the activity of the layer below from the layer above
        self.pcn_layers = nn.ModuleList([
            PCNLayer(
                in_dim=dims[i+1],  # From higher layer
                out_dim=dims[i],   # To lower layer
                activation_fn=torch.relu,
                activation_deriv=lambda a: (a > 0).float()
            )
            for i in range(self.n_layers)
        ])
        
        # Output projection from the highest latent back to n_embed
        self.output_proj = nn.Linear(dims[-1], n_embed, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.ln = nn.LayerNorm(n_embed)
        
        # Additional projection from input to match standard FFN behavior
        # This allows the network to learn a residual-like connection
        self.input_proj = nn.Linear(n_embed, dims[-1], bias=False)
        
    def init_latents(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Initialize latent variables for PCN inference."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize latents with small random values
        latents = []
        for dim in self.dims[1:]:
            latent = torch.randn(batch_size, seq_len, dim, device=device) * 0.01
            latent.requires_grad = False  # We'll update manually
            latents.append(latent)
        
        return latents
    
    def pcn_inference(self, x: torch.Tensor, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run PCN inference to find optimal latent states.
        
        This minimizes prediction errors through local updates only.
        """
        # x is the input (batch_size, seq_len, n_embed)
        inputs_and_latents = [x] + latents
        
        for _ in range(self.inference_steps):
            # Compute errors at each layer
            errors = []
            gm_errors = []
            
            for i, layer in enumerate(self.pcn_layers):
                # Get prediction from layer above
                x_hat, a = layer(inputs_and_latents[i + 1])
                
                # Compute prediction error
                error = inputs_and_latents[i] - x_hat
                
                # Compute gain-modulated error
                gm_error = error * layer.activation_deriv(a)
                
                errors.append(error)
                gm_errors.append(gm_error)
            
            # Update latents (except input which is fixed)
            for i in range(len(latents)):
                # For latent i (which corresponds to layer i+1 in the hierarchy):
                # 1. It receives top-down prediction error from trying to predict layer i
                # 2. It receives bottom-up error if it's not the top layer
                
                # Top-down gradient from prediction error
                grad = torch.zeros_like(latents[i])
                
                # Error from trying to predict the layer below
                if i < len(errors):
                    # Layer i+1 tries to predict layer i, so we get error[i]
                    # We need to backprop through the PCN layer
                    # gradient = d(error)/d(latent) = -W^T @ gm_error
                    W = self.pcn_layers[i].W  # Shape: (out_dim, in_dim)
                    grad = grad - torch.matmul(gm_errors[i], W.T)
                
                # Bottom-up gradient from layer above trying to predict this layer
                if i > 0:
                    # Layer i+2 tries to predict layer i+1 (this layer)
                    # So we get the prediction error directly
                    grad = grad + errors[i]
                
                # Update latent with gradient descent
                latents[i] = latents[i] - self.inference_lr * grad
        
        return latents
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using PCN inference.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        # Store input shape
        batch_size, seq_len, _ = x.shape
        
        # Apply layer norm to input for stability
        x_norm = self.ln(x)
        
        # Initialize latents
        with torch.no_grad():
            latents = self.init_latents(x_norm)
        
        # Run PCN inference
        with torch.no_grad():
            optimized_latents = self.pcn_inference(x_norm, latents)
        
        # The top latent is our learned representation
        top_latent = optimized_latents[-1]
        
        # Project back to original dimension
        output = self.output_proj(top_latent)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Note: Residual connection is handled by the transformer block
        return output
    
    def get_energy(self, x: torch.Tensor, latents: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the total prediction error (energy) of the current state.
        
        This can be used for analysis or as an auxiliary loss.
        """
        inputs_and_latents = [x] + latents
        total_error = 0.0
        
        for i, layer in enumerate(self.pcn_layers):
            x_hat, _ = layer(inputs_and_latents[i + 1])
            error = inputs_and_latents[i] - x_hat
            total_error = total_error + (error ** 2).sum()
        
        return total_error
    

class HybridPCNFeedForward(nn.Module):
    """
    Hybrid feedforward that can switch between PCN and standard FFN.
    
    Useful for ablation studies and gradual PCN integration.
    """
    
    def __init__(
        self,
        n_embed: int,
        dropout: float = 0.2,
        use_pcn: bool = True,
        pcn_ratio: float = 1.0,  # How much PCN vs standard FFN
        **pcn_kwargs
    ):
        super().__init__()
        
        self.use_pcn = use_pcn
        self.pcn_ratio = pcn_ratio
        
        if use_pcn:
            self.pcn_ff = PCNFeedForward(n_embed, dropout, **pcn_kwargs)
        
        # Always have standard FFN for comparison/mixing
        self.standard_ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_pcn:
            return self.standard_ff(x)
        
        if self.pcn_ratio >= 1.0:
            return self.pcn_ff(x)
        
        # Mix PCN and standard outputs
        pcn_out = self.pcn_ff(x)
        standard_out = self.standard_ff(x)
        
        return self.pcn_ratio * pcn_out + (1 - self.pcn_ratio) * standard_out