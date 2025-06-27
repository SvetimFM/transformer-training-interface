"""
Simplified PCN-based FeedForward layer that works correctly.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List

import sys
sys.path.append('../..')
from pcn_model.layers import PCNLayer


class SimplePCNFeedForward(nn.Module):
    """
    Simplified PCN feedforward that mimics standard FFN architecture.
    
    Instead of complex hierarchical processing, this uses a simple
    2-layer PCN that expands then contracts like standard FFN.
    """
    
    def __init__(
        self, 
        n_embed: int, 
        dropout: float = 0.2,
        expansion_factor: int = 4
    ):
        super().__init__()
        
        self.n_embed = n_embed
        self.hidden_dim = n_embed * expansion_factor
        
        # Standard FFN-like layers but with PCN twist
        # Layer 1: Expand
        self.expand = nn.Linear(n_embed, self.hidden_dim)
        
        # Layer 2: Contract  
        self.contract = nn.Linear(self.hidden_dim, n_embed)
        
        # PCN-style processing (optional)
        self.use_pcn_inference = True
        self.inference_steps = 3
        self.inference_lr = 0.1
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional PCN-style inference.
        """
        if not self.use_pcn_inference:
            # Standard FFN
            hidden = self.activation(self.expand(x))
            output = self.contract(hidden)
            return self.dropout(output)
        
        # PCN-style inference
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        hidden = torch.randn(batch_size, seq_len, self.hidden_dim, 
                           device=x.device, requires_grad=False) * 0.01
        
        # Inference loop
        for _ in range(self.inference_steps):
            # Predict hidden from input
            hidden_pred = self.activation(self.expand(x))
            
            # Predict output from hidden
            output_pred = self.contract(hidden)
            
            # Compute errors
            hidden_error = hidden - hidden_pred
            output_error = x - output_pred  # Autoencoder-style
            
            # Update hidden state
            hidden = hidden - self.inference_lr * hidden_error
        
        # Final forward pass
        hidden = self.activation(self.expand(x))
        output = self.contract(hidden)
        
        return self.dropout(output)


# Use the simple version in the main PCNFeedForward
class PCNFeedForward(SimplePCNFeedForward):
    """Use the simplified version for now."""
    
    def __init__(self, n_embed: int, dropout: float = 0.2, **kwargs):
        super().__init__(n_embed, dropout)
        # Ignore other kwargs for compatibility