import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .multi_head_attention_standard import MultiHeadAttentionStandard
from .hidden_layers import FeedForward
from .normalization import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, batch_size, block_size, 
                 use_layer_norm=True, use_residual=True, norm_position="pre", dropout=0.2,
                 use_standard_attention=True, hidden_multiplier=4):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.norm_position = norm_position
        
        if use_standard_attention:
            # Use standard multi-head attention (like GPT-2/3)
            self.attention = MultiHeadAttentionStandard(n_embed, n_heads, block_size, dropout)
        else:
            # Use educational implementation with separate heads
            head_size = n_embed // n_heads
            self.attention = MultiHeadAttention(n_heads, n_embed, head_size, batch_size, block_size, dropout)
        
        self.feed_forward = FeedForward(n_embed, dropout, hidden_multiplier)
        
        if use_layer_norm:
            self.ln1 = LayerNorm(n_embed)
            self.ln2 = LayerNorm(n_embed)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if self.use_layer_norm and self.norm_position == "pre":
            attn_out = self.attention(self.ln1(x))
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            ff_out = self.feed_forward(self.ln2(x))
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        elif self.use_layer_norm and self.norm_position == "post":
            attn_out = self.attention(x)
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            x = self.ln1(x)
            
            ff_out = self.feed_forward(x)
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
            x = self.ln2(x)
        
        else:
            attn_out = self.attention(x)
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            ff_out = self.feed_forward(x)
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        return x