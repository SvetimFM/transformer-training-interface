"""
Enhanced BigramLM with visualization delays for observing each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .bigram import BigramLM
from .visualization_transformer_block import VisualizationTransformerBlock
from .normalization import LayerNorm

class VisualizationBigramLM(nn.Module):
    """BigramLM with added delays in visualization mode"""
    
    def __init__(self, vocab_size, batch_size, block_size, config):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = config.training.device
        
        # Extract model configuration
        n_embed = config.model.n_embed
        n_heads = config.model.n_heads
        n_layers = config.model.n_layers
        dropout = config.model.dropout
        use_layer_norm = config.model.use_layer_norm
        use_residual = config.model.use_residual
        norm_position = config.model.norm_position
        
        self.use_layer_norm = use_layer_norm
        
        # Create embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Create transformer blocks with visualization support
        self.blocks = nn.ModuleList([
            VisualizationTransformerBlock(
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
        
        self._viz_config = None
        self._viz_callback = None
    
    def set_viz_config(self, config, callback=None):
        """Set visualization configuration and optional callback"""
        self._viz_config = config
        self._viz_callback = callback
        
        # Propagate to transformer blocks
        for block in self.blocks:
            block.set_viz_config(config, callback)
    
    def _get_viz_delay(self):
        """Calculate visualization delay based on speed ratio"""
        if self._viz_config and self._viz_config.visualization_mode and self._viz_config.visualization_speed_ratio > 0:
            # Smaller delay for internal steps
            delay = (1 - self._viz_config.visualization_speed_ratio) / self._viz_config.visualization_speed_ratio * 0.02
            return min(delay, 1.0)  # Cap at 1 second per sub-step
        return 0
    
    def _viz_sleep(self, phase_name=None):
        """Sleep if in visualization mode, optionally announce phase"""
        delay = self._get_viz_delay()
        if delay > 0:
            if phase_name and self._viz_callback:
                self._viz_callback({"phase": phase_name, "delay": delay})
            time.sleep(delay)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Step 1: Token embeddings
        self._viz_sleep("Token Embeddings")
        token_embeddings = self.token_embedding_table(idx)
        
        # Step 2: Position embeddings
        self._viz_sleep("Position Embeddings")
        posit_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        
        # Step 3: Combine embeddings
        self._viz_sleep("Embedding Addition")
        x = self.dropout(token_embeddings + posit_embeddings)
        
        # Step 4: Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            self._viz_sleep(f"Transformer Block {i+1}")
            x = block(x)
        
        # Step 5: Final layer norm
        if self.use_layer_norm:
            self._viz_sleep("Final Layer Norm")
        x = self.ln_f(x)
        
        # Step 6: Decoder head projection
        # Note: We skip delay here as embedding projection is naturally slow
        logits = self.decoder_head(x)
        
        if targets is None:
            loss = None
        else:
            # Step 7: Calculate loss
            self._viz_sleep("Loss Calculation")
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate text (no visualization delays during generation)"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        return idx