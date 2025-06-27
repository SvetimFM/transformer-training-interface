"""
Energy-Guided Transformer

A transformer architecture that uses PCN energy-based guidance to improve
token selection through refined attention patterns and distribution optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from models.guided_attention import PCNGuidedTransformerBlock
from models.pcn_token_guidance import PCNTokenGuidance, ContrastivePCNGuidance
from models.normalization import LayerNorm


class EnergyGuidedTransformer(nn.Module):
    """
    Transformer with PCN energy-guided token selection.
    
    Key features:
    - PCN-guided attention in transformer blocks
    - Energy-based refinement of token distributions
    - Contrastive learning for energy function
    - Multi-hypothesis exploration before final prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.2,
        # PCN guidance parameters
        use_pcn_guidance: bool = True,
        pcn_latent_dim: int = 256,
        n_refinement_steps: int = 5,
        guidance_strength: float = 0.1,
        use_contrastive: bool = True,
        # Which layers to apply guidance
        guided_layers: Optional[List[int]] = None,
        # Device
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.use_pcn_guidance = use_pcn_guidance
        self.device = device
        
        # If no specific layers specified, guide all layers
        if guided_layers is None:
            self.guided_layers = list(range(n_layers))
        else:
            self.guided_layers = guided_layers
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PCNGuidedTransformerBlock(
                n_embed=n_embed,
                n_heads=n_heads,
                block_size=block_size,
                dropout=dropout,
                use_pcn_guidance=(use_pcn_guidance and i in self.guided_layers),
                guidance_strength=guidance_strength
            )
            for i in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(n_embed)
        
        # Output projection
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # PCN token guidance module
        if use_pcn_guidance:
            if use_contrastive:
                self.pcn_guidance = ContrastivePCNGuidance(
                    hidden_dim=n_embed,
                    vocab_size=vocab_size,
                    latent_dim=pcn_latent_dim,
                    n_refinement_steps=n_refinement_steps,
                    n_heads=n_heads
                )
            else:
                self.pcn_guidance = PCNTokenGuidance(
                    hidden_dim=n_embed,
                    vocab_size=vocab_size,
                    latent_dim=pcn_latent_dim,
                    n_refinement_steps=n_refinement_steps,
                    n_heads=n_heads
                )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
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
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass with energy-guided token prediction.
        
        Args:
            idx: Input token indices (batch, seq_len)
            targets: Target tokens for training
            return_stats: Whether to return PCN statistics
            
        Returns:
            logits: Token predictions
            loss: Loss value (if targets provided)
            stats: Dictionary of statistics
        """
        B, T = idx.shape
        device = idx.device
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)
        x = self.embedding_dropout(tok_emb + pos_emb)
        
        # Stats collection
        stats = {
            'pcn_stats': [],
            'attention_stats': []
        }
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if self.use_pcn_guidance and i in self.guided_layers:
                # Get initial prediction for guidance
                with torch.no_grad():
                    # Quick forward pass to get context
                    temp_x = self.ln_f(x)
                    initial_logits = self.lm_head(temp_x[:, -1, :])  # Last position
                    context = temp_x[:, -1, :]  # Use last position as context
                
                # Get PCN guidance
                refined_logits, (q_bias, k_bias, v_bias), pcn_stats = self.pcn_guidance(
                    initial_logits, context, seq_len=T, return_stats=True
                )
                
                if return_stats:
                    stats['pcn_stats'].append(pcn_stats)
                
                # Apply guided attention
                x = block(x, pcn_biases=(q_bias, k_bias, v_bias))
            else:
                # Standard attention
                x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Apply final PCN refinement to last position
        if self.use_pcn_guidance and T > 0:
            # Refine the last position's predictions
            last_logits = logits[:, -1, :]
            context = x[:, -1, :]
            
            refined_last_logits, _, final_stats = self.pcn_guidance(
                last_logits, context, seq_len=1, return_stats=True
            )
            
            # Replace last position with refined predictions
            logits[:, -1, :] = refined_last_logits
            
            if return_stats:
                stats['final_refinement'] = final_stats
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
            
            # Add contrastive loss if using contrastive PCN
            if self.use_pcn_guidance and isinstance(self.pcn_guidance, ContrastivePCNGuidance):
                # Sample negative examples
                with torch.no_grad():
                    # Get top-k predictions that are NOT the target
                    top_k = torch.topk(logits[:, -1, :], k=10, dim=-1)
                    
                    # Create negative logits by swapping probabilities
                    negative_logits = logits[:, -1, :].clone()
                    
                    # Simple negative: uniform distribution
                    negative_logits = torch.ones_like(negative_logits) / self.vocab_size
                
                # Positive logits: one-hot at target position
                positive_logits = torch.zeros_like(logits[:, -1, :])
                positive_logits.scatter_(1, targets[:, -1:], 1.0)
                
                # Contrastive loss
                contrastive_loss = self.pcn_guidance.contrastive_loss(
                    positive_logits,
                    negative_logits,
                    x[:, -1, :]
                )
                
                # Combine losses
                loss = loss + 0.1 * contrastive_loss
                
                if return_stats:
                    stats['contrastive_loss'] = contrastive_loss.item()
        
        return logits, loss, stats
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        use_energy_sampling: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens using energy-guided selection.
        
        Args:
            idx: Initial context tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use argmax
            top_k: Top-k filtering
            use_energy_sampling: Whether to use energy-based sampling
            
        Returns:
            Generated token sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _, stats = self(idx_cond, return_stats=True)
            
            # Focus on last position
            logits = logits[:, -1, :]
            
            if self.use_pcn_guidance and use_energy_sampling and 'final_refinement' in stats:
                # Use energy-weighted sampling
                final_energy = stats['final_refinement']['final_energy']
                
                # Convert energy to additional logit bias (lower energy = higher probability)
                energy_bias = -final_energy * 0.5  # Scale factor
                
                # Apply energy bias to logits
                # This is a simple approach; could be more sophisticated
                logits = logits + energy_bias
            
            # Apply temperature
            logits = logits / temperature
            
            # Optional top-k filtering
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


class LightweightEnergyTransformer(nn.Module):
    """
    A simplified version for faster experimentation.
    
    Only applies PCN guidance at the final layer for token prediction,
    not throughout the attention mechanism.
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.2,
        pcn_latent_dim: int = 128,
        n_refinement_steps: int = 3,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.device = device
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)
        
        # Standard transformer blocks (no PCN guidance)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embed,
                nhead=n_heads,
                dim_feedforward=4 * n_embed,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(n_embed)
        
        # Output projection
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # Lightweight PCN guidance (only for final prediction)
        self.pcn_guidance = PCNTokenGuidance(
            hidden_dim=n_embed,
            vocab_size=vocab_size,
            latent_dim=pcn_latent_dim,
            n_refinement_steps=n_refinement_steps,
            n_heads=n_heads
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Simplified forward pass."""
        B, T = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask for transformer
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Get initial logits
        logits = self.lm_head(x)
        
        # Apply PCN refinement only to last position
        if T > 0:
            last_logits = logits[:, -1, :]
            context = x[:, -1, :]
            
            refined_logits, _, stats = self.pcn_guidance(
                last_logits, context, seq_len=1, return_stats=True
            )
            
            logits[:, -1, :] = refined_logits
        else:
            stats = {}
        
        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss, stats
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Standard generation with PCN refinement."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx