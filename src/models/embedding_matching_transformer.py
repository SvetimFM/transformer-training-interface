"""
Embedding-Matching Transformer

A transformer architecture that predicts future embedding sequences rather than
directly predicting tokens. Uses PCN-guided refinement for better exploration
of the continuous embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from models.transformer_block import TransformerBlock
from models.normalization import LayerNorm
from models.pcn_embedding_guidance import (
    PCNEmbeddingPredictor,
    EmbeddingRefinementLayer,
    MultiScaleEmbeddingPredictor
)


class EmbeddingMatchingTransformer(nn.Module):
    """
    Transformer that predicts embeddings instead of tokens.
    
    Key differences from standard transformer:
    1. Predicts sequences of embeddings using PCN guidance
    2. Uses embedding distance loss instead of cross-entropy
    3. Optional token projection for generation
    4. Multi-hypothesis exploration in continuous space
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.2,
        # Embedding prediction parameters
        n_future_predict: int = 5,
        n_hypotheses: int = 3,
        n_pcn_iterations: int = 10,
        use_multiscale: bool = False,
        use_refinement: bool = True,
        # Loss parameters
        embedding_loss_weight: float = 1.0,
        token_loss_weight: float = 0.1,
        # Device
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_future_predict = n_future_predict
        self.device = device
        
        # Loss weights
        self.embedding_loss_weight = embedding_loss_weight
        self.token_loss_weight = token_loss_weight
        
        # Token embeddings and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                batch_size=1,  # Will be set dynamically
                n_embed=n_embed,
                n_heads=n_heads,
                block_size=block_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.ln_f = LayerNorm(n_embed)
        
        # Embedding prediction components
        if use_multiscale:
            self.embedding_predictor = MultiScaleEmbeddingPredictor(
                embed_dim=n_embed,
                scales=[20, 10, 5],
                n_iterations=n_pcn_iterations
            )
        else:
            self.embedding_predictor = PCNEmbeddingPredictor(
                embed_dim=n_embed,
                n_future=n_future_predict,
                n_hypotheses=n_hypotheses,
                n_iterations=n_pcn_iterations,
                dropout=dropout
            )
        
        # Optional embedding refinement
        if use_refinement:
            self.embedding_refiner = EmbeddingRefinementLayer(
                embed_dim=n_embed,
                n_iterations=5,
                dropout=dropout
            )
        else:
            self.embedding_refiner = None
        
        # Output projection (for token generation)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input tokens."""
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        pos_emb = self.position_embedding_table(pos)
        
        # Combine
        x = tok_emb + pos_emb
        return x
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with embedding prediction.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for training
            return_embeddings: Return predicted embeddings
            
        Returns:
            logits: Token logits (B, T, vocab_size)
            loss: Combined loss (if targets provided)
            stats: Dictionary with losses and statistics
        """
        B, T = idx.shape
        device = idx.device
        
        # Get input embeddings
        x = self.get_embeddings(idx)
        
        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Optional embedding refinement
        if self.embedding_refiner is not None:
            x = self.embedding_refiner(x)
        
        # Initialize stats
        stats = {}
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # 1. Embedding prediction loss
            if T > self.n_future_predict:
                # For each position, predict next n_future embeddings
                embedding_losses = []
                
                for i in range(T - self.n_future_predict):
                    # Current context
                    context_embeddings = x[:, :i+1]
                    
                    # True future embeddings
                    future_tokens = targets[:, i+1:i+1+self.n_future_predict]
                    future_embeddings_true = self.token_embedding_table(future_tokens)
                    
                    # Predict future embeddings
                    predicted_embeddings, pred_stats = self.embedding_predictor(
                        context_embeddings,
                        future_true=future_embeddings_true
                    )
                    
                    # Embedding distance loss
                    # Use cosine distance + L2
                    cosine_loss = 1 - F.cosine_similarity(
                        predicted_embeddings.reshape(-1, self.n_embed),
                        future_embeddings_true.reshape(-1, self.n_embed)
                    ).mean()
                    
                    l2_loss = F.mse_loss(predicted_embeddings, future_embeddings_true)
                    
                    embedding_loss = 0.5 * cosine_loss + 0.5 * l2_loss
                    embedding_losses.append(embedding_loss)
                
                avg_embedding_loss = torch.stack(embedding_losses).mean()
                stats['embedding_loss'] = avg_embedding_loss.item()
            else:
                avg_embedding_loss = torch.tensor(0.0, device=device)
                stats['embedding_loss'] = 0.0
            
            # 2. Token prediction loss (auxiliary)
            logits = self.lm_head(x)
            
            if self.token_loss_weight > 0:
                token_loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1)
                )
                stats['token_loss'] = token_loss.item()
            else:
                token_loss = torch.tensor(0.0, device=device)
                stats['token_loss'] = 0.0
            
            # Combined loss
            loss = (self.embedding_loss_weight * avg_embedding_loss + 
                   self.token_loss_weight * token_loss)
            
            stats['total_loss'] = loss.item()
        else:
            # Just get logits for generation
            logits = self.lm_head(x)
        
        if return_embeddings:
            return logits, loss, stats, x
        else:
            return logits, loss, stats
    
    def generate_with_embedding_guidance(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        return_trajectories: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Generate tokens using embedding-space guidance.
        
        Instead of greedy token selection, uses PCN to explore embedding
        trajectories and select tokens based on embedding quality.
        """
        device = idx.device
        trajectories = [] if return_trajectories else None
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get current embeddings
            x = self.get_embeddings(idx_cond)
            x = self.blocks(x)
            x = self.ln_f(x)
            
            if self.embedding_refiner is not None:
                x = self.embedding_refiner(x)
            
            # Predict future embeddings
            predicted_embeddings, stats = self.embedding_predictor(
                x, return_all_hypotheses=True
            )  # Returns (B, n_hypotheses, n_future, embed_dim)
            
            # For each hypothesis, compute token probabilities
            # We'll use the first predicted embedding
            first_embeddings = predicted_embeddings[:, :, 0, :]  # (B, n_hypotheses, embed_dim)
            
            # Convert to logits
            hypothesis_logits = []
            for h in range(first_embeddings.shape[1]):
                h_logits = self.lm_head(first_embeddings[:, h])
                hypothesis_logits.append(h_logits)
            
            hypothesis_logits = torch.stack(hypothesis_logits, dim=1)  # (B, n_hyp, vocab)
            
            # Use energy-weighted average
            energies = stats['energies']  # (B, n_hypotheses)
            weights = F.softmax(-energies / temperature, dim=1)
            
            # Weighted average of logits
            logits = torch.sum(
                hypothesis_logits * weights.unsqueeze(-1),
                dim=1
            )  # (B, vocab)
            
            # Apply temperature
            logits = logits / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Save trajectory if requested
            if return_trajectories:
                trajectories.append(predicted_embeddings[0, :, 0, :].cpu())
        
        return idx, trajectories
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        use_embedding_guidance: bool = True
    ) -> torch.Tensor:
        """
        Standard generation interface.
        
        Can use either embedding-guided generation or standard generation.
        """
        if use_embedding_guidance:
            generated, _ = self.generate_with_embedding_guidance(
                idx, max_new_tokens, temperature, top_k
            )
            return generated
        else:
            # Standard generation
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


class StandardTransformer(nn.Module):
    """
    Standard transformer for baseline comparison.
    
    This is a simplified version without PCN components.
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.device = device
        
        # Token embeddings and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                batch_size=1,
                n_embed=n_embed,
                n_heads=n_heads,
                block_size=block_size,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Standard transformer forward pass."""
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embedding_table(idx)
        
        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        pos_emb = self.position_embedding_table(pos)
        
        # Combine and process
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        stats = {}
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )
            stats['loss'] = loss.item()
        
        return logits, loss, stats
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Standard generation."""
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