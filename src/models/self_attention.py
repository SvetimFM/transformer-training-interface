import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, batch_size, block_size):
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

    def forward(self, x):
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
        v = self.value(x)
        out = weights @ v
        out = self.proj(out)  # project back to n_embed dimensions

        return out
