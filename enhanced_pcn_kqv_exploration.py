"""
Enhanced PCN-Guided K-Q-V Exploration for Transformers

This module implements comprehensive exploration across all attention components
(keys, queries, and values) with adaptive exploration strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class AdaptiveExplorationScheduler:
    """Manages exploration rate based on training progress."""
    
    def __init__(
        self,
        initial_noise: float = 0.2,
        final_noise: float = 0.05,
        initial_samples: int = 5,
        final_samples: int = 2,
        warmup_steps: int = 1000,
        decay_steps: int = 5000
    ):
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.initial_samples = initial_samples
        self.final_samples = final_samples
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.step = 0
        
    def get_params(self) -> Tuple[float, int]:
        """Get current exploration parameters."""
        self.step += 1
        
        if self.step < self.warmup_steps:
            # Warmup phase - increase exploration
            progress = self.step / self.warmup_steps
            noise = self.initial_noise * progress
            samples = self.initial_samples
        else:
            # Decay phase - reduce exploration
            decay_progress = min(1.0, (self.step - self.warmup_steps) / self.decay_steps)
            noise = self.initial_noise + (self.final_noise - self.initial_noise) * decay_progress
            samples = int(self.initial_samples - 
                         (self.initial_samples - self.final_samples) * decay_progress)
        
        return noise, max(2, samples)


class EnhancedPCNKQVExploration(nn.Module):
    """
    Enhanced exploration of Keys, Queries, and Values with separate strategies
    for each component and adaptive exploration scheduling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_exploration_samples: int = 5,
        n_refinement_steps: int = 5,
        exploration_noise: float = 0.1,
        step_size: float = 0.01,
        temperature: float = 1.0,
        # Component-specific settings
        explore_queries: bool = True,
        explore_keys: bool = True,
        explore_values: bool = True,
        # Advanced features
        use_adaptive_exploration: bool = True,
        share_exploration_across_heads: bool = False,
        exploration_dropout: float = 0.1
    ):
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
        self.use_adaptive_exploration = use_adaptive_exploration
        self.share_exploration_across_heads = share_exploration_across_heads
        
        # Adaptive scheduler
        if use_adaptive_exploration:
            self.scheduler = AdaptiveExplorationScheduler(
                initial_noise=exploration_noise,
                initial_samples=n_exploration_samples
            )
        
        # Component-specific networks
        if explore_queries:
            self.query_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.query_refinement = nn.Linear(hidden_dim, hidden_dim)
        
        if explore_keys:
            self.key_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.key_refinement = nn.Linear(hidden_dim, hidden_dim)
        
        if explore_values:
            self.value_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.value_refinement = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-component interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=exploration_dropout, batch_first=True
        )
        
        # Diversity encouragement
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
        # Exploration dropout
        self.dropout = nn.Dropout(exploration_dropout)
    
    def compute_exploration_energy(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy for K-Q-V configuration with richer metrics.
        """
        B, n_heads, T, head_dim = queries.shape
        
        # Attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)
        
        # Energy components
        # 1. Attention entropy (encourage diverse attention)
        entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean(dim=(1, 2))
        
        # 2. Output magnitude (stable representations)
        output_magnitude = attn_output.norm(dim=-1).mean(dim=(1, 2))
        
        # 3. Key-Query alignment variance (diverse patterns)
        alignment = (queries * keys).sum(dim=-1)
        alignment_var = alignment.var(dim=-1).mean(dim=1)
        
        # 4. Value diversity (different information per head)
        if not self.share_exploration_across_heads:
            value_diversity = 0
            for h1 in range(n_heads):
                for h2 in range(h1 + 1, n_heads):
                    similarity = F.cosine_similarity(
                        values[:, h1].flatten(1),
                        values[:, h2].flatten(1),
                        dim=1
                    )
                    value_diversity += (1 - similarity.mean())
            value_diversity = value_diversity / (n_heads * (n_heads - 1) / 2)
        else:
            value_diversity = 0
        
        # Combine energies (lower is better)
        energy = (
            output_magnitude 
            - 0.1 * entropy 
            - 0.05 * alignment_var
            - 0.1 * value_diversity
        )
        
        return energy
    
    def explore_component(
        self,
        component: torch.Tensor,
        component_type: str,  # 'query', 'key', or 'value'
        context: torch.Tensor,
        other_components: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Explore a specific attention component (Q, K, or V).
        """
        B, n_heads, T, head_dim = component.shape
        
        # Get adaptive parameters if enabled
        if self.use_adaptive_exploration:
            noise_scale, n_samples = self.scheduler.get_params()
        else:
            noise_scale = self.exploration_noise
            n_samples = self.n_exploration_samples
        
        # Get predictor for this component
        predictor = getattr(self, f'{component_type}_predictor')
        refinement = getattr(self, f'{component_type}_refinement')
        
        # Predict expected component from context
        context_pred = predictor(context)
        context_pred = context_pred.reshape(B, T, n_heads, head_dim).transpose(1, 2)
        
        # Initialize exploration samples
        samples = []
        for _ in range(n_samples):
            # Add exploration noise
            noise = torch.randn_like(component) * noise_scale
            sample = component + noise
            
            # Initial refinement based on context
            sample = sample + 0.5 * (context_pred - sample)
            samples.append(sample)
        
        # Iterative refinement
        for step in range(self.n_refinement_steps):
            refined_samples = []
            
            for i, sample in enumerate(samples):
                # Compute gradient-free update
                pred_error = sample - context_pred
                
                # Cross-component influence
                if component_type == 'query' and 'key' in other_components:
                    # Queries influenced by keys
                    influence = torch.matmul(
                        sample, 
                        other_components['key'].transpose(-2, -1)
                    ).mean(dim=-1, keepdim=True)
                    update = -self.step_size * (pred_error - 0.1 * influence)
                    
                elif component_type == 'key' and 'query' in other_components:
                    # Keys influenced by queries
                    influence = torch.matmul(
                        other_components['query'].transpose(-2, -1),
                        sample
                    ).mean(dim=-2, keepdim=True)
                    update = -self.step_size * (pred_error - 0.1 * influence)
                    
                elif component_type == 'value':
                    # Values aim for diversity
                    update = -self.step_size * pred_error
                else:
                    update = -self.step_size * pred_error
                
                # Apply update
                sample = sample + update
                
                # Add decreasing noise
                if step < self.n_refinement_steps - 1:
                    noise = torch.randn_like(sample) * (noise_scale * 0.5 ** (step + 1))
                    sample = sample + noise
                
                refined_samples.append(sample)
            
            samples = refined_samples
        
        # Compute energies for selection
        energies = []
        for sample in samples:
            # Build full K-Q-V configuration
            if component_type == 'query':
                q, k, v = sample, other_components.get('key', component), other_components.get('value', component)
            elif component_type == 'key':
                q, k, v = other_components.get('query', component), sample, other_components.get('value', component)
            else:  # value
                q, k, v = other_components.get('query', component), other_components.get('key', component), sample
            
            energy = self.compute_exploration_energy(q, k, v, mask)
            energies.append(energy)
        
        energies = torch.stack(energies)  # (n_samples, B)
        
        # Select best sample (or weighted average)
        if self.training:
            # During training, use weighted average
            probs = F.softmax(-energies / self.temperature, dim=0)
            selected = sum(p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * s 
                          for p, s in zip(probs, samples))
        else:
            # During inference, take best
            best_idx = energies.argmin(dim=0)
            selected = samples[0]  # Initialize
            for b in range(B):
                selected[b] = samples[best_idx[b]][b]
        
        # Apply final refinement
        selected = selected + refinement(selected.transpose(1, 2).reshape(B, T, -1)).reshape(B, T, n_heads, head_dim).transpose(1, 2)
        selected = self.dropout(selected)
        
        # Compute statistics
        stats = {
            f'{component_type}_energies': energies.mean(dim=1).tolist(),
            f'{component_type}_energy_std': energies.std(dim=1).tolist(),
            f'{component_type}_noise_scale': noise_scale,
            f'{component_type}_n_samples': n_samples
        }
        
        # Compute diversity if we have multiple samples
        if len(samples) > 1:
            diversity = 0
            n = len(samples)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = F.cosine_similarity(
                        samples[i].flatten(1),
                        samples[j].flatten(1),
                        dim=1
                    ).mean()
                    diversity += (1 - sim)
            diversity = diversity / (n * (n - 1) / 2)
            stats[f'{component_type}_diversity'] = diversity.item()
        
        return selected, stats
    
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
        Forward pass with exploration of enabled components.
        """
        all_stats = {}
        components = {'query': queries, 'key': keys, 'value': values}
        
        # Explore each enabled component
        if self.explore_queries:
            other_comps = {k: v for k, v in components.items() if k != 'query'}
            queries, q_stats = self.explore_component(
                queries, 'query', hidden_states, other_comps, mask
            )
            all_stats.update(q_stats)
            components['query'] = queries
        
        if self.explore_keys:
            other_comps = {k: v for k, v in components.items() if k != 'key'}
            keys, k_stats = self.explore_component(
                keys, 'key', hidden_states, other_comps, mask
            )
            all_stats.update(k_stats)
            components['key'] = keys
        
        if self.explore_values:
            other_comps = {k: v for k, v in components.items() if k != 'value'}
            values, v_stats = self.explore_component(
                values, 'value', hidden_states, other_comps, mask
            )
            all_stats.update(v_stats)
            components['value'] = values
        
        # Cross-component refinement (optional)
        if self.training and (self.explore_queries or self.explore_keys or self.explore_values):
            # Use cross-attention for final refinement
            B, n_heads, T, head_dim = queries.shape
            
            # Reshape for cross-attention
            q_flat = queries.transpose(1, 2).reshape(B, T, -1)
            k_flat = keys.transpose(1, 2).reshape(B, T, -1)
            v_flat = values.transpose(1, 2).reshape(B, T, -1)
            
            # Apply cross-attention
            refined_q, _ = self.cross_attention(q_flat, k_flat, v_flat)
            queries = refined_q.reshape(B, T, n_heads, head_dim).transpose(1, 2)
        
        if return_stats:
            return queries, keys, values, all_stats
        else:
            return queries, keys, values, None