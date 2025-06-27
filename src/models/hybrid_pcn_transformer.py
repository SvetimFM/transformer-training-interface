"""
Hybrid PCN-Transformer Model

Base model class that supports various PCN-Transformer hybrid architectures.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List, Dict

from models.pcn_transformer_block import PCNTransformerBlock, AlternatingPCNTransformerBlock
from models.transformer_block import TransformerBlock
from models.normalization import LayerNorm

import sys
sys.path.append('../..')
from pcn_model.network import PredictiveCodingNetwork
from pcn_model.layers import PCNLayer


class HybridPCNTransformer(nn.Module):
    """
    Base class for hybrid PCN-Transformer architectures.
    
    Supports multiple architecture types:
    1. pcn_ff: Standard transformer with PCN feedforward
    2. alternating: Alternating attention and PCN layers
    3. hierarchical: PCN layers followed by transformer layers
    4. dual_stream: Parallel PCN and transformer paths
    5. pcn_positional: PCN-based positional encoding
    
    Args:
        vocab_size: Vocabulary size
        batch_size: Batch size
        block_size: Maximum sequence length
        architecture: Type of hybrid architecture
        config: Configuration dict
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        architecture: str = "pcn_ff",
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.architecture = architecture
        
        # Parse config
        cfg = config or {}
        self.n_embed = cfg.get('n_embed', 256)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 6)
        self.dropout = cfg.get('dropout', 0.2)
        self.use_layer_norm = cfg.get('use_layer_norm', True)
        self.use_residual = cfg.get('use_residual', True)
        self.norm_position = cfg.get('norm_position', 'pre')
        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # PCN-specific config
        self.pcn_config = cfg.get('pcn_config', {
            'n_pcn_layers': 2,
            'inference_steps': 5,
            'inference_lr': 0.1
        })
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embed)
        
        # Build architecture-specific components
        if architecture == "pcn_positional":
            # Use PCN for positional encoding
            self._build_pcn_positional()
        else:
            # Standard positional embedding
            self.position_embedding_table = nn.Embedding(block_size, self.n_embed)
        
        # Build main architecture
        if architecture == "pcn_ff":
            self._build_pcn_ff()
        elif architecture == "alternating":
            self._build_alternating()
        elif architecture == "hierarchical":
            self._build_hierarchical()
        elif architecture == "dual_stream":
            self._build_dual_stream()
        elif architecture == "pcn_positional":
            self._build_standard_with_pcn_pos()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Output head
        self.decoder_head = nn.Linear(self.n_embed, vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def _build_pcn_ff(self):
        """Build transformer with PCN feedforward layers."""
        self.blocks = nn.ModuleList([
            PCNTransformerBlock(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                batch_size=self.batch_size,
                block_size=self.block_size,
                use_layer_norm=self.use_layer_norm,
                use_residual=self.use_residual,
                norm_position=self.norm_position,
                dropout=self.dropout,
                use_pcn_ff=True,
                pcn_config=self.pcn_config
            ) for _ in range(self.n_layers)
        ])
        
        self.ln_f = LayerNorm(self.n_embed) if self.use_layer_norm else nn.Identity()
    
    def _build_alternating(self):
        """Build alternating attention and PCN layers."""
        self.blocks = nn.ModuleList()
        
        for i in range(self.n_layers):
            block_type = "attention" if i % 2 == 0 else "pcn"
            self.blocks.append(
                AlternatingPCNTransformerBlock(
                    n_embed=self.n_embed,
                    n_heads=self.n_heads,
                    batch_size=self.batch_size,
                    block_size=self.block_size,
                    block_type=block_type,
                    use_layer_norm=self.use_layer_norm,
                    use_residual=self.use_residual,
                    norm_position=self.norm_position,
                    dropout=self.dropout,
                    pcn_config=self.pcn_config
                )
            )
        
        self.ln_f = LayerNorm(self.n_embed) if self.use_layer_norm else nn.Identity()
    
    def _build_hierarchical(self):
        """Build PCN layers followed by transformer layers."""
        # PCN layers for feature extraction
        n_pcn_layers = self.n_layers // 2
        n_transformer_layers = self.n_layers - n_pcn_layers
        
        # PCN hierarchy
        pcn_dims = [self.n_embed]
        for i in range(n_pcn_layers):
            # Gradually increase dimension
            pcn_dims.append(self.n_embed * (2 ** min(i, 2)))
        
        self.pcn_layers = nn.ModuleList([
            PCNLayer(
                in_dim=pcn_dims[i+1],
                out_dim=pcn_dims[i],
                activation_fn=torch.relu
            )
            for i in range(n_pcn_layers)
        ])
        
        # Project PCN output back to n_embed
        self.pcn_projection = nn.Linear(pcn_dims[-1], self.n_embed)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                batch_size=self.batch_size,
                block_size=self.block_size,
                use_layer_norm=self.use_layer_norm,
                use_residual=self.use_residual,
                norm_position=self.norm_position,
                dropout=self.dropout
            ) for _ in range(n_transformer_layers)
        ])
        
        self.ln_f = LayerNorm(self.n_embed) if self.use_layer_norm else nn.Identity()
    
    def _build_dual_stream(self):
        """Build parallel PCN and transformer streams."""
        # Transformer stream
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                batch_size=self.batch_size,
                block_size=self.block_size,
                use_layer_norm=self.use_layer_norm,
                use_residual=self.use_residual,
                norm_position=self.norm_position,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])
        
        # PCN stream (simplified PCN network)
        pcn_dims = [self.n_embed, self.n_embed * 2, self.n_embed]
        self.pcn_network = PredictiveCodingNetwork(
            dims=pcn_dims,
            output_dim=self.n_embed
        )
        
        # Fusion layer (learnable gating)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.n_embed * 2, self.n_embed),
            nn.Sigmoid()
        )
        
        self.ln_f = LayerNorm(self.n_embed) if self.use_layer_norm else nn.Identity()
    
    def _build_pcn_positional(self):
        """Build PCN-based positional encoding."""
        # PCN for learning position representations
        self.position_pcn = nn.ModuleList([
            PCNLayer(
                in_dim=self.n_embed,
                out_dim=self.n_embed,
                activation_fn=torch.relu
            ) for _ in range(2)
        ])
        
        # Learnable position queries
        self.position_queries = nn.Parameter(
            torch.randn(self.block_size, self.n_embed) * 0.01
        )
    
    def _build_standard_with_pcn_pos(self):
        """Build standard transformer with PCN positional encoding."""
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=self.n_embed,
                n_heads=self.n_heads,
                batch_size=self.batch_size,
                block_size=self.block_size,
                use_layer_norm=self.use_layer_norm,
                use_residual=self.use_residual,
                norm_position=self.norm_position,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])
        
        self.ln_f = LayerNorm(self.n_embed) if self.use_layer_norm else nn.Identity()
    
    def get_positional_embeddings(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings (standard or PCN-based)."""
        if self.architecture == "pcn_positional":
            # Use PCN to generate position embeddings
            positions = self.position_queries[:seq_len].unsqueeze(0)  # (1, seq_len, n_embed)
            
            # Run through PCN layers
            for layer in self.position_pcn:
                pred, _ = layer(positions)
                positions = pred
            
            return positions.squeeze(0)  # (seq_len, n_embed)
        else:
            # Standard positional embeddings
            return self.position_embedding_table(
                torch.arange(seq_len, device=self.device)
            )
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass through the hybrid model.
        
        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target token indices for loss computation
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape
        
        # Token embeddings
        token_embeddings = self.token_embedding_table(idx)
        
        # Positional embeddings
        posit_embeddings = self.get_positional_embeddings(T)
        
        # Combine embeddings
        x = self.dropout_layer(token_embeddings + posit_embeddings)
        
        # Architecture-specific forward pass
        if self.architecture == "hierarchical":
            x = self._forward_hierarchical(x)
        elif self.architecture == "dual_stream":
            x = self._forward_dual_stream(x)
        else:
            # Standard forward (pcn_ff, alternating, pcn_positional)
            for block in self.blocks:
                x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.decoder_head(x)
        
        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def _forward_hierarchical(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for hierarchical architecture."""
        batch_size, seq_len, _ = x.shape
        
        # Initialize PCN latents
        latents = [x]
        for i in range(len(self.pcn_layers)):
            dim = self.pcn_layers[i].in_dim
            latent = torch.randn(batch_size, seq_len, dim, device=x.device) * 0.01
            latents.append(latent)
        
        # PCN inference (simplified - just a few steps)
        for _ in range(3):
            for i, layer in enumerate(self.pcn_layers):
                pred, _ = layer(latents[i + 1])
                error = latents[i] - pred
                latents[i + 1] = latents[i + 1] - 0.1 * error
        
        # Project PCN output
        pcn_out = self.pcn_projection(latents[-1])
        
        # Pass through transformer blocks
        x = pcn_out
        for block in self.transformer_blocks:
            x = block(x)
        
        return x
    
    def _forward_dual_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for dual stream architecture."""
        # Transformer stream
        trans_out = x
        for block in self.transformer_blocks:
            trans_out = block(trans_out)
        
        # PCN stream (simplified forward)
        batch_size, seq_len, _ = x.shape
        pcn_latents = self.pcn_network.init_latents(batch_size * seq_len, x.device)
        
        # Flatten for PCN processing
        x_flat = x.view(-1, self.n_embed)
        inputs_and_latents = [x_flat] + pcn_latents
        
        # Quick PCN inference
        for _ in range(3):
            errors, gm_errors = self.pcn_network.compute_errors(inputs_and_latents)
            for i in range(len(pcn_latents)):
                inputs_and_latents[i + 1] = inputs_and_latents[i + 1] - 0.1 * errors[i]
        
        # PCN output
        pcn_out = self.pcn_network.readout(inputs_and_latents[-1])
        pcn_out = pcn_out.view(batch_size, seq_len, self.n_embed)
        
        # Fusion
        combined = torch.cat([trans_out, pcn_out], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * trans_out + (1 - gate) * pcn_out
        
        return fused
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last position logits
            logits = logits[:, -1, :]
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx