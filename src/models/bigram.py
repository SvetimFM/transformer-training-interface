import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


# Bigram Language Model
# But what is happening here per tutorial is essentially we take our training data and use it to train a bigram model that just looks at N token to predict N+1 token


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            # get predictions
            logits, loss = self(idx)

            # bigram - looking just at the last token in the time sequence
            logits = logits[:, -1, :]  # B, C - bye bye T dim! (for now)

            # activate
            probs = F.softmax(logits, dim=-1)  # B, C

            # sample the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1

            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1 -> append to the sequence
        return idx
