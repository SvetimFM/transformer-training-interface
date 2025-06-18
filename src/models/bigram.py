from models.multi_head_attention import MultiHeadAttention
from models.hidden_layers import FeedForward
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


# Bigram Language Model
# But what is happening here per tutorial is essentially we take our training data and use it to train a bigram model that just looks at N token to predict N+1 token

n_embed = 256
num_heads = 8
head_size = 32


class BigramLM(nn.Module):
    def __init__(self, vocab_size, batch_size, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embed
        )  # n_embed is number of embedded dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = MultiHeadAttention(
            num_heads, n_embed, head_size, batch_size, block_size
        )
        self.ffwd = FeedForward(n_embed)
        self.decoder_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        token_embeddings = token_embeddings.to("cuda")
        posit_embeddings = self.position_embedding_table(torch.arange(T, device="cuda"))
        posit_embeddings = posit_embeddings.to("cuda")

        # (T,C)
        # now we are looking at our sequences through embeddings of their relationships
        x = token_embeddings + posit_embeddings  # (B, T, C)
        x = self.sa_head(x)
        x = self.ffwd(x)

        logits = self.decoder_head(x)  # (B,T, vocab_size)

        # generation mode
        if targets is None:
            loss = None
        # training mode
        else:
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # rotation to make pytorch play nice with the B, T, C logit

            targets = targets.view(B * T)
            loss = F.cross_entropy(
                logits, targets
            )  # error calculation - distance between target and what we got instead

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]

            # get predictions
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            # activate
            probs = F.softmax(logits, dim=-1)  # B, C

            # sample the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1

            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1 -> append to the sequence
        return idx
