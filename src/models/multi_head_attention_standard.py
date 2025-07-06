import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionStandard(nn.Module):
    """
    Standard multi-head attention implementation as used in GPT-2/3, BERT, etc.
    Uses single projection matrices for all heads, then reshapes.
    """
    
    def __init__(self, n_embed, n_heads, block_size, dropout=0.2, bias=True):
        super().__init__()
        assert n_embed % n_heads == 0, "n_embed must be divisible by n_heads"
        
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.block_size = block_size
        
        # Single projection matrices for all heads combined
        self.q_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        
    def forward(self, x, return_attention=False):
        B, T, C = x.shape
        
        # Project Q, K, V for all heads at once
        q = self.q_proj(x)  # (B, T, n_embed)
        k = self.k_proj(x)  # (B, T, n_embed)
        v = self.v_proj(x)  # (B, T, n_embed)
        
        # Reshape to separate heads: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Store attention weights for visualization if requested
        if return_attention:
            # Average attention weights across heads for visualization
            attention_weights = attn_weights.mean(dim=1)  # (B, T, T)
        
        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values: (B, n_heads, T, head_dim)
        out = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, n_heads, head_dim) -> (B, T, n_embed)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        if return_attention:
            return out, attention_weights
        return out
    
    def count_parameters(self):
        """Count parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)