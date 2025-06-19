from models.self_attention import SelfAttentionHead
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
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
