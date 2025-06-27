"""
PCN-Guided Latent Exploration for Transformers

This module implements predictive coding-based exploration of the latent space
before token generation, allowing the model to sample and refine multiple
hypotheses in continuous space before discretization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PCNExplorationLayer(nn.Module):
    """
    Explores multiple latent hypotheses using PCN dynamics before token generation.
    
    This layer samples multiple candidates in the continuous latent space and
    refines them through energy minimization before selecting the best one.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_samples: int = 5,
        n_steps: int = 10,
        exploration_noise: float = 0.1,
        step_size: float = 0.01,
        temperature: float = 1.0
    ):
        """
        Args:
            hidden_dim: Dimension of hidden representations
            n_samples: Number of latent hypotheses to explore
            n_steps: Number of PCN refinement steps
            exploration_noise: Std dev of initial exploration noise
            step_size: Learning rate for PCN updates
            temperature: Temperature for energy-based selection
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.exploration_noise = exploration_noise
        self.step_size = step_size
        self.temperature = temperature
        
        # Context predictor - predicts expected representation from context
        self.context_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Energy network - computes quality/coherence of latent
        self.energy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Diversity encouragement
        self.diversity_weight = nn.Parameter(torch.tensor(0.1))
        
    def compute_energy(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        other_latents: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute energy (quality score) for a latent hypothesis.
        
        Lower energy = better hypothesis
        """
        # Base energy from coherence with context
        combined = torch.cat([latent, context], dim=-1)
        base_energy = self.energy_net(combined).squeeze(-1)
        
        # Add diversity penalty if other hypotheses provided
        if other_latents is not None and len(other_latents) > 0:
            # Penalize similarity to other hypotheses
            diversity_penalty = 0
            for other in other_latents:
                similarity = F.cosine_similarity(latent, other, dim=-1)
                diversity_penalty = diversity_penalty + similarity
            diversity_penalty = diversity_penalty / len(other_latents)
            
            # Lower energy for diverse hypotheses
            total_energy = base_energy - self.diversity_weight * diversity_penalty
        else:
            total_energy = base_energy
            
        return total_energy
    
    def pcn_refine(
        self,
        latents: List[torch.Tensor],
        context: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Refine latent hypotheses through PCN dynamics.
        """
        refined_latents = []
        
        # Get context prediction (what we expect based on context)
        context_pred = self.context_predictor(context)
        
        for i, latent in enumerate(latents):
            current_latent = latent.clone()
            
            # Get other latents for diversity
            other_latents = [l for j, l in enumerate(latents) if j != i]
            
            # PCN refinement steps - simplified without autograd
            for step in range(self.n_steps):
                # Compute prediction error
                pred_error = current_latent - context_pred.detach()
                
                # Simple gradient-free update based on prediction error
                # Move towards context prediction while maintaining some originality
                update = -self.step_size * pred_error
                current_latent = current_latent + update
                
                # Add small noise for diversity
                if step < self.n_steps - 1:
                    noise = torch.randn_like(current_latent) * 0.01
                    current_latent = current_latent + noise
            
            refined_latents.append(current_latent)
            
        return refined_latents
    
    def forward(
        self,
        hidden: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_all_samples: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: explore and refine multiple hypotheses.
        
        Args:
            hidden: Current hidden state (batch_size, seq_len, hidden_dim)
            context: Context representation for conditioning
            return_all_samples: If True, return all explored samples
            
        Returns:
            Best refined latent (or all samples if requested)
        """
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device
        
        # Use hidden as context if not provided
        if context is None:
            # Use mean of sequence as context
            context = hidden.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # Initialize multiple hypotheses with exploration noise
        hypotheses = []
        for _ in range(self.n_samples):
            noise = torch.randn_like(hidden) * self.exploration_noise
            hypothesis = hidden + noise
            hypotheses.append(hypothesis)
        
        # Refine hypotheses through PCN dynamics
        # Process each position in sequence
        refined_hypotheses = []
        for pos in range(seq_len):
            pos_latents = [h[:, pos, :] for h in hypotheses]
            pos_context = context[:, pos, :]
            
            refined_pos = self.pcn_refine(pos_latents, pos_context)
            refined_hypotheses.append(refined_pos)
        
        # Reconstruct full sequences
        final_hypotheses = []
        for i in range(self.n_samples):
            hypothesis_seq = torch.stack(
                [refined_hypotheses[pos][i] for pos in range(seq_len)], 
                dim=1
            )
            final_hypotheses.append(hypothesis_seq)
        
        # Compute final energies for selection
        energies = []
        for hypothesis in final_hypotheses:
            # Average energy across sequence
            energy_seq = []
            for pos in range(seq_len):
                e = self.compute_energy(
                    hypothesis[:, pos, :],
                    context[:, pos, :],
                    None  # Don't use diversity for final selection
                )
                energy_seq.append(e)
            
            avg_energy = torch.stack(energy_seq, dim=1).mean(dim=1)
            energies.append(avg_energy)
        
        energies = torch.stack(energies, dim=0)  # (n_samples, batch_size)
        
        if return_all_samples:
            return torch.stack(final_hypotheses, dim=0), energies
        
        # Select best hypothesis using softmin (lower energy = better)
        probs = F.softmax(-energies / self.temperature, dim=0)
        
        # Weighted average of hypotheses (soft selection)
        selected = sum(p.unsqueeze(-1).unsqueeze(-1) * h 
                      for p, h in zip(probs, final_hypotheses))
        
        return selected


class MultiScalePCNExploration(nn.Module):
    """
    Hierarchical PCN exploration at multiple scales.
    
    Explores high-level concepts first, then refines down to token level.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        scales: List[int] = [256, 128, 64],
        n_samples: int = 5,
        n_steps: int = 10
    ):
        super().__init__()
        
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # Projection layers for different scales
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim, scale) for scale in scales
        ])
        
        # PCN exploration at each scale
        self.scale_explorers = nn.ModuleList([
            PCNExplorationLayer(
                hidden_dim=scale,
                n_samples=n_samples,
                n_steps=n_steps
            ) for scale in scales
        ])
        
        # Upsampling layers
        self.upsamplers = nn.ModuleList([
            nn.Linear(scales[i], scales[i-1] if i > 0 else hidden_dim)
            for i in range(len(scales))
        ])
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical exploration from coarse to fine.
        """
        current = hidden
        
        # Explore at each scale
        for i, (proj, explorer, upsampler) in enumerate(
            zip(self.scale_projections, self.scale_explorers, self.upsamplers)
        ):
            # Project to current scale
            scaled = proj(current)
            
            # Explore at this scale
            explored = explorer(scaled)
            
            # Upsample for next scale (or final output)
            current = upsampler(explored)
            
            # Add residual connection if dimensions match
            if current.shape == hidden.shape:
                current = current + hidden
        
        return current


class EnergyGuidedBeamSearch(nn.Module):
    """
    Beam search in continuous latent space using PCN energy.
    
    Instead of discrete token beams, maintains continuous latent trajectories.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        beam_size: int = 5,
        exploration_layer: Optional[PCNExplorationLayer] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        
        # Use provided exploration layer or create default
        self.exploration = exploration_layer or PCNExplorationLayer(
            hidden_dim=hidden_dim,
            n_samples=beam_size,
            n_steps=10
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        initial_hidden: torch.Tensor,
        max_length: int,
        context_fn=None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate sequence using energy-guided beam search.
        
        Returns:
            tokens: Generated token sequence
            latent_trajectory: List of latent states
        """
        device = initial_hidden.device
        batch_size = initial_hidden.shape[0]
        
        # Initialize beams
        beam_latents = [initial_hidden.unsqueeze(1)]  # Add seq dimension
        beam_scores = torch.zeros(batch_size, 1).to(device)
        
        generated_tokens = []
        latent_trajectory = []
        
        for step in range(max_length):
            # Get context if function provided
            context = context_fn(beam_latents[-1]) if context_fn else None
            
            # Explore from current beam state
            current_latent = beam_latents[-1][:, -1, :]  # Last position
            explored_latents, energies = self.exploration(
                current_latent.unsqueeze(1),
                context,
                return_all_samples=True
            )
            
            # Convert to logits
            logits = self.output_projection(explored_latents)  # (n_samples, batch, 1, vocab)
            
            # Get top tokens for each beam
            # ... (beam search logic with continuous latents)
            
            # For now, simplified: take best exploration and greedy decode
            best_idx = energies.argmin(dim=0)
            best_latent = explored_latents[best_idx, torch.arange(batch_size)]
            
            latent_trajectory.append(best_latent.squeeze(1))
            
            # Generate token
            logits = self.output_projection(best_latent)
            tokens = logits.argmax(dim=-1)
            generated_tokens.append(tokens)
        
        return torch.cat(generated_tokens, dim=1), latent_trajectory