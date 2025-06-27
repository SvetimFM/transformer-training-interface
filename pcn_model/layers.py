"""
PCN Layer Implementation

Implements individual layers of a Predictive Coding Network with top-down predictions.
"""

import torch
import torch.nn as nn
from torch.amp import autocast


class PCNLayer(nn.Module):
    """
    A single layer in a Predictive Coding Network.
    
    This layer generates top-down predictions from layer l+1 to layer l using
    learned weights W^(l) and a nonlinear activation function.
    
    Args:
        in_dim (int): Dimension of the layer above (d_{l+1})
        out_dim (int): Dimension of the current layer (d_l)
        activation_fn (callable): Nonlinear activation function f^(l)
        activation_deriv (callable): Derivative of activation function f^(l)'
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_fn=torch.relu,
        activation_deriv=lambda a: (a > 0).float()
    ):
        super().__init__()
        
        # Initialize generative weights W^(l) with shape (out_dim, in_dim)
        # This maps from layer l+1 (dimension in_dim) to layer l (dimension out_dim)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x_above):
        """
        Generate predictions for the layer below.
        
        Args:
            x_above: Activity from layer l+1, shape (batch_size, in_dim)
        
        Returns:
            x_hat: Predictions for layer l, shape (batch_size, out_dim)
            a: Pre-activations, shape (batch_size, out_dim)
        """
        with autocast(device_type='cuda'):
            # Compute pre-activations: A^(l) = X^(l+1) @ W^(l)^T
            a = x_above @ self.W.T
            
            # Apply activation function: X_hat^(l) = f^(l)(A^(l))
            x_hat = self.activation_fn(a)
            
            return x_hat, a
    
    def __repr__(self):
        return f"PCNLayer(in_dim={self.in_dim}, out_dim={self.out_dim})"