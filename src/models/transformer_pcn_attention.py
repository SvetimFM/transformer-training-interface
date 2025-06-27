"""
Transformer with PCN-Guided Attention

This model uses PCN exploration in the attention mechanism to discover
alternative attention patterns through energy minimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from models.pcn_attention_exploration import PCNGuidedMultiHeadAttention
from models.normalization import LayerNorm


class TransformerBlockPCNAttention(nn.Module):
    """
    Transformer block with PCN-guided attention.
    """
    
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        dropout: float = 0.2,
        # PCN parameters
        use_pcn_exploration: bool = True,
        n_exploration_samples: int = 3,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        explore_queries: bool = True,
        explore_keys: bool = False,
        explore_values: bool = False
    ):
        super().__init__()
        
        # Layer normalization
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
        
        # PCN-guided attention
        self.attn = PCNGuidedMultiHeadAttention(
            hidden_dim=n_embed,
            n_heads=n_heads,
            dropout=dropout,
            use_pcn_exploration=use_pcn_exploration,
            n_exploration_samples=n_exploration_samples,
            n_refinement_steps=n_refinement_steps,
            exploration_noise=exploration_noise,
            explore_queries=explore_queries,
            explore_keys=explore_keys,
            explore_values=explore_values
        )
        
        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_exploration_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional exploration statistics.
        """
        # Attention with residual
        attn_out, stats = self.attn(self.ln1(x), mask, return_exploration_stats)
        x = x + attn_out
        
        # Feedforward with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, stats


class TransformerWithPCNAttention(nn.Module):
    """
    Full transformer model with PCN-guided attention exploration.
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.2,
        # PCN exploration parameters
        use_pcn_exploration: bool = True,
        pcn_layers: Optional[List[int]] = None,  # Which layers to apply PCN to
        n_exploration_samples: int = 3,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        explore_queries: bool = True,
        explore_keys: bool = False,
        explore_values: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.device = device
        self.use_pcn_exploration = use_pcn_exploration
        
        # If pcn_layers not specified, apply to all layers
        if pcn_layers is None:
            pcn_layers = list(range(n_layers))
        self.pcn_layers = pcn_layers
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockPCNAttention(
                n_embed=n_embed,
                n_heads=n_heads,
                dropout=dropout,
                use_pcn_exploration=(use_pcn_exploration and i in pcn_layers),
                n_exploration_samples=n_exploration_samples,
                n_refinement_steps=n_refinement_steps,
                exploration_noise=exploration_noise,
                explore_queries=explore_queries,
                explore_keys=explore_keys,
                explore_values=explore_values
            ) for i in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_exploration_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with optional exploration statistics.
        """
        B, T = idx.shape
        device = idx.device
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos)
        
        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        # Collect exploration statistics
        all_stats = {
            'query_energies': [],
            'query_diversity': [],
            'query_probs': []
        }
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x, stats = block(x, mask, return_exploration_stats and i in self.pcn_layers)
            
            if stats is not None:
                for key in stats:
                    if key in all_stats:
                        all_stats[key].extend(stats[key])
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            logits = logits.view(B, T, C)
        
        if return_exploration_stats and all_stats['query_energies']:
            return logits, loss, all_stats
        else:
            return logits, loss, None
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens using the model.
        """
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _, _ = self(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample or take argmax
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def analyze_attention_patterns(
        self,
        idx: torch.Tensor,
        layer: int = 0
    ) -> Dict:
        """
        Analyze attention patterns for a given input.
        """
        self.eval()
        
        # Store original state
        original_use_pcn = self.use_pcn_exploration
        
        # Get attention patterns with and without PCN
        patterns = {}
        
        for use_pcn in [False, True]:
            self.use_pcn_exploration = use_pcn
            for block in self.blocks:
                if hasattr(block.attn, 'use_pcn_exploration'):
                    block.attn.use_pcn_exploration = use_pcn
            
            with torch.no_grad():
                # Forward pass to get attention weights
                # This is simplified - in practice would need to extract from attention layers
                _, _, stats = self(idx, return_exploration_stats=True)
                
            patterns[f'pcn_{use_pcn}'] = stats if stats else {}
        
        # Restore original state
        self.use_pcn_exploration = original_use_pcn
        for block in self.blocks:
            if hasattr(block.attn, 'use_pcn_exploration'):
                block.attn.use_pcn_exploration = original_use_pcn
        
        return patterns