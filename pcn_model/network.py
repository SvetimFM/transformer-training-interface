"""
Predictive Coding Network Implementation

Full network architecture that orchestrates multiple PCN layers for hierarchical predictive coding.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .layers import PCNLayer


class PredictiveCodingNetwork(nn.Module):
    """
    A complete Predictive Coding Network with hierarchical layers.
    
    The network consists of:
    - Multiple PCN layers forming a generative hierarchy
    - A linear readout layer for supervised tasks
    - Methods for initializing latents and computing errors
    
    Args:
        dims (List[int]): Layer dimensions [d_0, d_1, ..., d_L]
        output_dim (int): Dimension of the output/readout layer
    """
    
    def __init__(self, dims: List[int], output_dim: int):
        super().__init__()
        
        self.dims = dims
        self.L = len(dims) - 1  # Number of latent layers
        
        # Build the generative hierarchy of PCN layers
        # Each layer l predicts layer l-1 from layer l+1
        self.layers = nn.ModuleList([
            PCNLayer(
                in_dim=dims[l+1],   # Reads from layer l+1
                out_dim=dims[l]     # Predicts layer l
            )
            for l in range(self.L)  # l = 0, ..., L-1
        ])
        
        # Readout layer for supervised learning
        # Maps top latent X^(L) to predicted output Y_hat
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)
        
    def init_latents(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """
        Initialize latent variables with random values.
        
        Args:
            batch_size: Number of samples in the batch
            device: Device to create tensors on
            
        Returns:
            List of initialized latent tensors [X^(1), ..., X^(L)]
        """
        return [
            torch.randn(batch_size, d, device=device, requires_grad=False)
            for d in self.dims[1:]  # Skip input dimension
        ]
    
    def compute_errors(
        self, 
        inputs_latents: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute prediction errors and gain-modulated errors for all layers.
        
        Args:
            inputs_latents: List [X^(0), X^(1), ..., X^(L)] of tensors
                           with shapes [(B, d_0), ..., (B, d_L)]
        
        Returns:
            errors: List of prediction errors [E^(0), ..., E^(L-1)]
            gain_modulated_errors: List of gain-modulated errors [H^(0), ..., H^(L-1)]
        """
        errors = []
        gain_modulated_errors = []
        
        for l, layer in enumerate(self.layers):  # l = 0, ..., L-1
            # Get predictions from layer l+1 to layer l
            x_hat, a = layer(inputs_latents[l + 1])
            
            # Compute prediction error: E^(l) = X^(l) - X_hat^(l)
            err = inputs_latents[l] - x_hat
            
            # Compute gain-modulated error: H^(l) = E^(l) ⊙ f^(l)'(A^(l))
            gm_err = err * layer.activation_deriv(a)
            
            errors.append(err)
            gain_modulated_errors.append(gm_err)
            
        return errors, gain_modulated_errors
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (not used during PCN training).
        
        This is mainly for compatibility and potential future use.
        PCN training uses the inference and learning procedures instead.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        raise NotImplementedError(
            "PCN does not use standard forward pass. "
            "Use the training procedures with inference and learning phases."
        )
    
    def get_total_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"PredictiveCodingNetwork(\n"
            f"  dims={self.dims},\n"
            f"  output_dim={self.readout.out_features},\n"
            f"  num_layers={self.L},\n"
            f"  total_params={self.get_total_parameters():,}\n"
            f")"
        )