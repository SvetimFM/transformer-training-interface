"""
Transformer with PCN-Guided Latent Exploration

This model integrates PCN exploration mechanisms with standard transformers,
allowing for richer sampling of the embedding space before token generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from models.transformer_block import TransformerBlock
from models.normalization import LayerNorm
from models.pcn_exploration import PCNExplorationLayer, MultiScalePCNExploration


class TransformerWithPCNExploration(nn.Module):
    """
    Transformer model enhanced with PCN-guided latent exploration.
    
    Adds exploration layers that sample and refine multiple hypotheses
    in the continuous latent space before projecting to tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.2,
        # PCN exploration parameters
        use_pcn_exploration: bool = True,
        exploration_points: str = "final",  # "final", "all", "middle"
        n_exploration_samples: int = 5,
        n_exploration_steps: int = 10,
        exploration_noise: float = 0.1,
        use_hierarchical: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.device = device
        self.use_pcn_exploration = use_pcn_exploration
        self.exploration_points = exploration_points
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embed=n_embed,
                n_heads=n_heads,
                batch_size=batch_size,
                block_size=block_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.ln_f = LayerNorm(n_embed)
        
        # PCN exploration layers
        if use_pcn_exploration:
            if use_hierarchical:
                # Single hierarchical exploration at the end
                self.exploration = MultiScalePCNExploration(
                    hidden_dim=n_embed,
                    scales=[n_embed // 2, n_embed // 4],
                    n_samples=n_exploration_samples,
                    n_steps=n_exploration_steps
                )
                self.exploration_layers = None
            else:
                # Standard exploration
                if exploration_points == "all":
                    # Exploration after each transformer block
                    self.exploration_layers = nn.ModuleList([
                        PCNExplorationLayer(
                            hidden_dim=n_embed,
                            n_samples=n_exploration_samples,
                            n_steps=n_exploration_steps,
                            exploration_noise=exploration_noise
                        ) for _ in range(n_layers)
                    ])
                    self.exploration = None
                elif exploration_points == "middle":
                    # Exploration at middle layer
                    self.exploration_middle = PCNExplorationLayer(
                        hidden_dim=n_embed,
                        n_samples=n_exploration_samples,
                        n_steps=n_exploration_steps,
                        exploration_noise=exploration_noise
                    )
                    self.exploration = None
                    self.exploration_layers = None
                else:  # "final"
                    # Single exploration layer at the end
                    self.exploration = PCNExplorationLayer(
                        hidden_dim=n_embed,
                        n_samples=n_exploration_samples,
                        n_steps=n_exploration_steps,
                        exploration_noise=exploration_noise
                    )
                    self.exploration_layers = None
        else:
            self.exploration = None
            self.exploration_layers = None
        
        # Output projection
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
        Forward pass with optional PCN exploration.
        
        Args:
            idx: Input token indices (batch_size, seq_len)
            targets: Target tokens for loss computation
            return_exploration_stats: Return statistics about exploration
            
        Returns:
            logits: Output logits
            loss: Cross-entropy loss if targets provided
            stats: Exploration statistics if requested
        """
        B, T = idx.shape
        device = idx.device
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embed)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos)  # (1, T, n_embed)
        
        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)
        
        # Store exploration stats
        exploration_stats = {
            'energies': [],
            'diversity': [],
            'refinement_delta': []
        }
        
        # Process through transformer blocks with optional exploration
        if self.use_pcn_exploration and self.exploration_points == "all":
            for i, (block, explorer) in enumerate(zip(self.blocks, self.exploration_layers)):
                x = block(x)
                
                # Apply exploration
                x_before = x.clone()
                x = explorer(x)
                
                if return_exploration_stats:
                    delta = (x - x_before).norm(dim=-1).mean()
                    exploration_stats['refinement_delta'].append(delta.item())
        
        elif self.use_pcn_exploration and self.exploration_points == "middle":
            middle_idx = len(self.blocks) // 2
            for i, block in enumerate(self.blocks):
                x = block(x)
                
                if i == middle_idx:
                    x_before = x.clone()
                    x = self.exploration_middle(x)
                    
                    if return_exploration_stats:
                        delta = (x - x_before).norm(dim=-1).mean()
                        exploration_stats['refinement_delta'].append(delta.item())
        
        else:
            # Standard transformer processing
            for block in self.blocks:
                x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Apply final exploration if configured
        if self.use_pcn_exploration and self.exploration is not None:
            x_before = x.clone()
            
            if hasattr(self.exploration, 'forward') and 'return_all_samples' in self.exploration.forward.__code__.co_varnames:
                # Get all samples for analysis
                x_samples, energies = self.exploration(x, return_all_samples=True)
                
                if return_exploration_stats:
                    # Compute diversity of explored samples
                    if x_samples.shape[0] > 1:
                        diversity = 0
                        n_samples = x_samples.shape[0]
                        for i in range(n_samples):
                            for j in range(i+1, n_samples):
                                sim = F.cosine_similarity(
                                    x_samples[i].flatten(1),
                                    x_samples[j].flatten(1),
                                    dim=1
                                ).mean()
                                diversity += (1 - sim)
                        diversity = diversity / (n_samples * (n_samples - 1) / 2)
                        exploration_stats['diversity'].append(diversity.item())
                    
                    exploration_stats['energies'].extend(energies.mean(dim=1).tolist())
                
                # Use best sample
                best_idx = energies.mean(dim=1).argmin()
                x = x_samples[best_idx]
            else:
                x = self.exploration(x)
            
            if return_exploration_stats:
                delta = (x - x_before).norm(dim=-1).mean()
                exploration_stats['refinement_delta'].append(delta.item())
        
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
        
        # Reshape logits back
        logits = logits.view(B, T, -1)
        
        if return_exploration_stats:
            return logits, loss, exploration_stats
        else:
            return logits, loss
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        use_exploration: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Generate tokens with optional PCN exploration.
        
        Args:
            idx: Initial context tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use argmax
            top_k: Top-k filtering
            use_exploration: Override model's exploration setting
            
        Returns:
            Generated token sequence
        """
        # Decide whether to use exploration
        if use_exploration is None:
            use_exploration = self.use_pcn_exploration
        
        # Temporarily set exploration mode
        original_use_exploration = self.use_pcn_exploration
        self.use_pcn_exploration = use_exploration
        
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
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
        
        # Restore original exploration setting
        self.use_pcn_exploration = original_use_exploration
        
        return idx
    
    def get_exploration_analysis(
        self,
        idx: torch.Tensor
    ) -> Dict:
        """
        Analyze exploration behavior on given input.
        """
        self.eval()
        with torch.no_grad():
            _, _, stats = self(idx, return_exploration_stats=True)
        
        analysis = {
            'mean_energy': sum(stats['energies']) / len(stats['energies']) if stats['energies'] else 0,
            'mean_diversity': sum(stats['diversity']) / len(stats['diversity']) if stats['diversity'] else 0,
            'mean_refinement': sum(stats['refinement_delta']) / len(stats['refinement_delta']) if stats['refinement_delta'] else 0,
            'exploration_active': self.use_pcn_exploration
        }
        
        return analysis