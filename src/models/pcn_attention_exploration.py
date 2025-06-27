"""
PCN-Guided Attention Exploration for Transformers

This module implements PCN-based exploration in the key-query-value space,
allowing discovery of alternative attention patterns through energy minimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class PCNAttentionExploration(nn.Module):
    """
    Explores alternative key-query-value configurations using PCN dynamics.
    
    Instead of exploring final embeddings, this explores the attention
    mechanism's latent space to discover alternative attention patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_exploration_samples: int = 3,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        step_size: float = 0.01,
        temperature: float = 1.0,
        explore_queries: bool = True,
        explore_keys: bool = False,
        explore_values: bool = False
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            n_heads: Number of attention heads
            n_exploration_samples: Number of alternative hypotheses to explore
            n_refinement_steps: Number of PCN refinement iterations
            exploration_noise: Initial noise for exploration
            step_size: Learning rate for refinement
            temperature: Temperature for energy-based selection
            explore_queries: Whether to explore query space
            explore_keys: Whether to explore key space
            explore_values: Whether to explore value space
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_exploration_samples = n_exploration_samples
        self.n_refinement_steps = n_refinement_steps
        self.exploration_noise = exploration_noise
        self.step_size = step_size
        self.temperature = temperature
        
        self.explore_queries = explore_queries
        self.explore_keys = explore_keys
        self.explore_values = explore_values
        
        # Energy networks for different components
        if explore_queries:
            self.query_energy_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Query refinement predictor
            self.query_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        if explore_keys:
            self.key_energy_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Note: We use a simpler energy function that doesn't require a network
        
        # Diversity weight
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
    def compute_attention_energy(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy for a given attention configuration.
        
        Lower energy = better attention pattern
        """
        B, n_heads, T, head_dim = queries.shape
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, values)
        
        # Simple energy based on attention statistics
        # 1. Mean attention output magnitude
        output_magnitude = attn_output.norm(dim=-1).mean(dim=(1, 2))
        
        # 2. Attention entropy (diversity)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean(dim=(1, 2))
        
        # 3. Query-key alignment
        alignment = (queries * keys).sum(dim=-1).mean(dim=(1, 2))
        
        # Combine into energy (lower is better)
        energy = output_magnitude - 0.1 * entropy - 0.05 * alignment
        
        return energy
    
    def explore_queries_pcn(
        self,
        queries_original: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """
        Explore alternative query configurations.
        """
        B, n_heads, T, head_dim = queries_original.shape
        device = queries_original.device
        
        # Initialize multiple query hypotheses
        query_hypotheses = []
        for _ in range(self.n_exploration_samples):
            noise = torch.randn_like(queries_original) * self.exploration_noise
            hypothesis = queries_original + noise
            query_hypotheses.append(hypothesis)
        
        # Get context from hidden states
        context = self.query_predictor(hidden_states)
        context = context.reshape(B, T, n_heads, head_dim).transpose(1, 2)
        
        # Refine queries through PCN dynamics
        refined_queries = []
        energies = []
        
        for hypothesis in query_hypotheses:
            current_q = hypothesis.clone()
            
            for step in range(self.n_refinement_steps):
                # Prediction error
                pred_error = current_q - context
                
                # Compute attention energy
                energy = self.compute_attention_energy(
                    current_q, keys, values, mask
                )
                
                # Update queries to minimize prediction error and energy
                update = -self.step_size * pred_error
                current_q = current_q + update
                
                # Add small noise for exploration
                if step < self.n_refinement_steps - 1:
                    noise = torch.randn_like(current_q) * 0.01
                    current_q = current_q + noise
            
            refined_queries.append(current_q)
            
            # Final energy
            final_energy = self.compute_attention_energy(
                current_q, keys, values, mask
            )
            energies.append(final_energy)
        
        # Stack energies
        energies = torch.stack(energies, dim=0)  # (n_samples, B)
        
        # Select best hypothesis using softmin
        probs = F.softmax(-energies / self.temperature, dim=0)
        
        # Weighted average of refined queries
        selected_queries = sum(
            p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * q
            for p, q in zip(probs, refined_queries)
        )
        
        # Compute diversity of hypotheses
        diversity = 0
        n = len(refined_queries)
        if n > 1:
            for i in range(n):
                for j in range(i+1, n):
                    sim = F.cosine_similarity(
                        refined_queries[i].flatten(1),
                        refined_queries[j].flatten(1),
                        dim=1
                    ).mean()
                    diversity += (1 - sim)
            diversity = diversity / (n * (n - 1) / 2)
        
        stats = {
            'query_energies': energies.mean(dim=1).tolist(),
            'query_diversity': [diversity.item()],
            'query_probs': probs.mean(dim=1).tolist()
        }
        
        return selected_queries, stats
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional exploration of K-Q-V space.
        
        Returns:
            queries, keys, values: Potentially refined tensors
            stats: Exploration statistics if requested
        """
        stats = {}
        
        # Explore queries if enabled
        if self.explore_queries:
            queries, query_stats = self.explore_queries_pcn(
                queries, keys, values, hidden_states, mask
            )
            stats.update(query_stats)
        
        # TODO: Implement key and value exploration if needed
        
        if return_stats:
            return queries, keys, values, stats
        else:
            return queries, keys, values, None


class PCNGuidedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with PCN-guided exploration of K-Q-V space.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        # PCN exploration parameters
        use_pcn_exploration: bool = True,
        n_exploration_samples: int = 3,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        explore_queries: bool = True,
        explore_keys: bool = False,
        explore_values: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_pcn_exploration = use_pcn_exploration
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # PCN exploration
        if use_pcn_exploration:
            self.pcn_explorer = PCNAttentionExploration(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_exploration_samples=n_exploration_samples,
                n_refinement_steps=n_refinement_steps,
                exploration_noise=exploration_noise,
                explore_queries=explore_queries,
                explore_keys=explore_keys,
                explore_values=explore_values
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_exploration_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with optional PCN exploration.
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        queries = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        exploration_stats = None
        
        # Apply PCN exploration if enabled
        if self.use_pcn_exploration:
            queries, keys, values, exploration_stats = self.pcn_explorer(
                queries, keys, values, x, mask, return_exploration_stats
            )
        
        # Compute attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, values)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        
        if return_exploration_stats:
            return output, exploration_stats
        else:
            return output, None