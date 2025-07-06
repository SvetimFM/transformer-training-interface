import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.2, hidden_multiplier=4):
        super().__init__()
        hidden_dim = hidden_multiplier * n_embed
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)