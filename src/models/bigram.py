from models.transformer_block import TransformerBlock
from models.normalization import LayerNorm
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLM(nn.Module):
    def __init__(self, vocab_size, batch_size, block_size, config=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Use config if provided, otherwise use defaults
        if config:
            n_embed = config.model.n_embed
            n_heads = config.model.n_heads
            n_layers = config.model.n_layers
            dropout = config.model.dropout
            use_layer_norm = config.model.use_layer_norm
            use_residual = config.model.use_residual
            norm_position = config.model.norm_position
            self.device = config.training.device
        else:
            n_embed = 256
            n_heads = 8
            n_layers = 1
            dropout = 0.2
            use_layer_norm = False
            use_residual = False
            norm_position = "pre"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=n_embed,
                n_heads=n_heads,
                batch_size=batch_size,
                block_size=block_size,
                use_layer_norm=use_layer_norm,
                use_residual=use_residual,
                norm_position=norm_position,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final layer norm if using normalization
        self.ln_f = LayerNorm(n_embed) if use_layer_norm else nn.Identity()
        
        self.decoder_head = nn.Linear(n_embed, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx)
        posit_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        
        x = self.dropout(token_embeddings + posit_embeddings)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        logits = self.decoder_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        return idx