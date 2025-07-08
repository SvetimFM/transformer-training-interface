import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, batch_size, block_size, dropout=0.2):
        super().__init__()
        self.n_embed = n_embed
        self.batch_size = batch_size
        self.block_size = block_size
        self.head_size = head_size

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.proj = nn.Linear(head_size, head_size)  # output projection
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(self.block_size, self.block_size)),
        )  # adding a tril module
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B, T, C = x.shape  # from here on, adopting the shared notation

        k = self.key(x)  # (B T HEAD_W)
        q = self.query(x)  # (B T HEAD_W)

        # activate plus normalized
        # softmax converges to one hot vectors w.o normalization- (with convergence towards max, we end up looking just at one node)
        weights = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B T 16) @ (B 16 T) -> (B, T , T)

        # impose future token mask-out
        # weights = torch.zeros((self.block_size, self.block_size))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(weights, dim=-1)
        
        # Store pre-dropout weights for visualization
        if return_attention:
            attention_weights = weights.clone()
        
        weights = self.attn_dropout(weights)  # dropout on attention weights
        
        v = self.value(x)
        
        out = weights @ v  # (B, T, T) @ (B, T, HEAD_W) -> (B, T, HEAD_W)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        if return_attention:
            return out, attention_weights
        return out


class MultiHeadAttention(nn.Module):
    """
    Educational implementation of multi-head attention.
    Creates separate attention heads as distinct modules.
    
    Note: This is not the standard implementation. For the standard
    implementation used in GPT/BERT, see MultiHeadAttentionStandard.
    This version is easier to understand but less efficient.
    """
    def __init__(self, num_heads, n_embed, head_size, batch_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(head_size, n_embed, batch_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            outputs = []
            attentions = []
            for head in self.heads:
                out, attn = head(x, return_attention=True)
                outputs.append(out)
                attentions.append(attn)
            return torch.cat(outputs, dim=-1), attentions
        else:
            return torch.cat([head(x) for head in self.heads], dim=-1)
