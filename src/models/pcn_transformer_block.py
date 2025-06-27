"""
PCN-enhanced Transformer Block with configurable feedforward network.

This module extends the standard transformer block to support PCN-based
feedforward networks while maintaining compatibility with standard FFN.
"""

import torch
import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.normalization import LayerNorm
from models.pcn_feedforward_v2 import PCNFeedForward, PCNFeedForwardWithMemory
from models.hidden_layers import FeedForward


class PCNTransformerBlock(nn.Module):
    """
    Transformer block that can use either standard or PCN-based feedforward.
    
    Args:
        n_embed: Embedding dimension
        n_heads: Number of attention heads
        batch_size: Batch size
        block_size: Maximum sequence length
        use_layer_norm: Whether to use layer normalization
        use_residual: Whether to use residual connections
        norm_position: "pre" or "post" normalization
        dropout: Dropout probability
        use_pcn_ff: Whether to use PCN feedforward instead of standard
        pcn_config: Configuration dict for PCN feedforward
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        batch_size: int,
        block_size: int,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        norm_position: str = "pre",
        dropout: float = 0.2,
        use_pcn_ff: bool = False,
        pcn_config: dict = None
    ):
        super().__init__()
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.norm_position = norm_position
        self.use_pcn_ff = use_pcn_ff
        
        # Multi-head attention
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(
            n_heads, n_embed, head_size, batch_size, block_size, dropout
        )
        
        # Feedforward network (PCN or standard)
        if use_pcn_ff:
            pcn_cfg = pcn_config or {}
            # Remove n_pcn_layers from config if present (not used in v2)
            pcn_cfg = {k: v for k, v in pcn_cfg.items() if k != 'n_pcn_layers'}
            self.feed_forward = PCNFeedForward(
                n_embed=n_embed,
                dropout=dropout,
                **pcn_cfg
            )
        else:
            self.feed_forward = FeedForward(n_embed, dropout)
        
        # Layer normalization
        if use_layer_norm:
            self.ln1 = LayerNorm(n_embed)
            self.ln2 = LayerNorm(n_embed)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embed)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embed)
        """
        if self.use_layer_norm and self.norm_position == "pre":
            # Pre-norm: LayerNorm → Attention → Residual
            attn_out = self.attention(self.ln1(x))
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            # Pre-norm: LayerNorm → FFN → Residual
            ff_out = self.feed_forward(self.ln2(x))
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        elif self.use_layer_norm and self.norm_position == "post":
            # Post-norm: Attention → Residual → LayerNorm
            attn_out = self.attention(x)
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            x = self.ln1(x)
            
            # Post-norm: FFN → Residual → LayerNorm
            ff_out = self.feed_forward(x)
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
            x = self.ln2(x)
        
        else:
            # No layer norm
            attn_out = self.attention(x)
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            ff_out = self.feed_forward(x)
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        return x
    
    def get_pcn_energy(self) -> torch.Tensor:
        """
        Get the PCN energy (prediction error) if using PCN feedforward.
        
        Returns None if using standard feedforward.
        """
        if self.use_pcn_ff and hasattr(self.feed_forward, 'get_energy'):
            return self.feed_forward.get_energy()
        return None


class AlternatingPCNTransformerBlock(nn.Module):
    """
    Transformer block that alternates between attention and PCN processing.
    
    Instead of attention→FFN, this does either:
    - Attention only (odd blocks)
    - PCN processing only (even blocks)
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        batch_size: int,
        block_size: int,
        block_type: str = "attention",  # "attention" or "pcn"
        use_layer_norm: bool = True,
        use_residual: bool = True,
        norm_position: str = "pre",
        dropout: float = 0.2,
        pcn_config: dict = None
    ):
        super().__init__()
        
        self.block_type = block_type
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.norm_position = norm_position
        
        if block_type == "attention":
            # Attention-only block
            head_size = n_embed // n_heads
            self.main_layer = MultiHeadAttention(
                n_heads, n_embed, head_size, batch_size, block_size, dropout
            )
        else:
            # PCN-only block
            pcn_cfg = pcn_config or {}
            # Remove n_pcn_layers from config if present (not used in v2)
            pcn_cfg = {k: v for k, v in pcn_cfg.items() if k != 'n_pcn_layers'}
            self.main_layer = PCNFeedForward(
                n_embed=n_embed,
                dropout=dropout,
                **pcn_cfg
            )
        
        # Layer normalization
        if use_layer_norm:
            self.ln = LayerNorm(n_embed)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through alternating block."""
        if self.use_layer_norm and self.norm_position == "pre":
            # Pre-norm
            out = self.main_layer(self.ln(x))
            x = x + self.dropout(out) if self.use_residual else self.dropout(out)
        
        elif self.use_layer_norm and self.norm_position == "post":
            # Post-norm
            out = self.main_layer(x)
            x = x + self.dropout(out) if self.use_residual else self.dropout(out)
            x = self.ln(x)
        
        else:
            # No layer norm
            out = self.main_layer(x)
            x = x + self.dropout(out) if self.use_residual else self.dropout(out)
        
        return x