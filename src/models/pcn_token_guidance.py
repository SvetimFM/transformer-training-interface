"""
PCN Token Guidance Module

This module implements PCN-based learning of efficient token representations
and guides the model toward optimal token selection through energy minimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class PCNTokenGuidance(nn.Module):
    """
    Learns an efficient representation of the token distribution space
    and guides selection through energy-based refinement.
    
    Key innovations:
    - Operates in learned latent space (not token embeddings)
    - Energy function captures contextual fit and distribution quality
    - Generates attention biases to guide K-Q-V computations
    - Iterative refinement improves token selection
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        latent_dim: int = 256,
        n_refinement_steps: int = 5,
        learning_rate: float = 0.1,
        target_entropy: float = 2.0,
        temperature: float = 1.0,
        exploration_noise: float = 0.05,
        n_heads: int = 8
    ):
        """
        Args:
            hidden_dim: Transformer hidden dimension
            vocab_size: Size of token vocabulary
            latent_dim: Dimension of PCN latent space
            n_refinement_steps: Number of PCN iterations
            learning_rate: Learning rate for PCN updates
            target_entropy: Target entropy for distribution smoothness
            temperature: Temperature for energy computations
            exploration_noise: Noise for exploration during refinement
            n_heads: Number of attention heads (for bias generation)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.n_refinement_steps = n_refinement_steps
        self.learning_rate = learning_rate
        self.target_entropy = target_entropy
        self.temperature = temperature
        self.exploration_noise = exploration_noise
        self.n_heads = n_heads
        
        # Latent space encoder (context + initial logits → latent)
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim + vocab_size, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Latent space decoder (latent → refined logits)
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, vocab_size)
        )
        
        # Context encoder for energy computation
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )
        
        # Energy network components
        self.energy_context = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        
        self.energy_distribution = nn.Sequential(
            nn.Linear(vocab_size, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        
        # Refinement network (predicts latent updates)
        self.refinement_net = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Attention bias generators
        head_dim = hidden_dim // n_heads
        
        self.q_bias_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.k_bias_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.v_bias_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def compute_token_energy(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        token_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute energy for token distribution quality.
        
        Lower energy = better token distribution.
        
        Args:
            latent: Latent representation (batch, latent_dim)
            context: Context representation (batch, hidden_dim)
            token_logits: Token logits (batch, vocab_size)
            
        Returns:
            total_energy: Scalar energy value
            components: Dict of energy components for analysis
        """
        batch_size = latent.shape[0]
        
        # 1. Contextual coherence energy
        context_encoded = self.context_encoder(context)
        context_combined = torch.cat([latent, context_encoded], dim=-1)
        context_energy = self.energy_context(context_combined).squeeze(-1)
        
        # 2. Distribution quality energy
        # Compute entropy of distribution
        probs = F.softmax(token_logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Penalty for deviation from target entropy
        entropy_deviation = (entropy - self.target_entropy).pow(2)
        
        # 3. Top-k consistency energy
        # Top tokens should have coherent probabilities (not random)
        top_k = min(10, self.vocab_size)
        top_k_logits, top_k_indices = torch.topk(token_logits, k=top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Variance of top-k probabilities (lower = more consistent)
        top_k_variance = torch.var(top_k_probs, dim=-1)
        
        # 4. Sparsity energy (encourage focused distributions)
        # Using Gini coefficient as sparsity measure
        sorted_probs, _ = torch.sort(probs, dim=-1)
        index = torch.arange(1, self.vocab_size + 1, device=probs.device).float()
        gini = 1 - 2 * torch.sum(sorted_probs * index, dim=-1) / (self.vocab_size * probs.sum(dim=-1))
        sparsity_energy = (0.7 - gini).pow(2)  # Target Gini of 0.7
        
        # Combine energies
        total_energy = (
            context_energy + 
            0.5 * entropy_deviation + 
            0.3 * top_k_variance + 
            0.2 * sparsity_energy
        ).mean()
        
        components = {
            'context_energy': context_energy.mean().item(),
            'entropy_deviation': entropy_deviation.mean().item(),
            'top_k_variance': top_k_variance.mean().item(),
            'sparsity_energy': sparsity_energy.mean().item(),
            'total_energy': total_energy.item()
        }
        
        return total_energy, components
    
    def refine_token_distribution(
        self,
        initial_logits: torch.Tensor,
        context: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Refine token distribution through PCN iterations.
        
        Args:
            initial_logits: Initial token logits from transformer (batch, vocab_size)
            context: Context representation (batch, hidden_dim)
            return_trajectory: Whether to return refinement trajectory
            
        Returns:
            refined_logits: Improved token logits
            stats: Dictionary with refinement statistics
        """
        batch_size = initial_logits.shape[0]
        device = initial_logits.device
        
        # Encode to latent space
        encoder_input = torch.cat([context, initial_logits], dim=-1)
        latent = self.latent_encoder(encoder_input)
        
        # Track refinement trajectory
        trajectory = [latent.clone()] if return_trajectory else None
        energies = []
        
        # PCN refinement iterations
        for step in range(self.n_refinement_steps):
            # Decode current latent to logits
            current_logits = self.latent_decoder(latent)
            
            # Compute energy and components
            with torch.no_grad():
                energy, energy_components = self.compute_token_energy(
                    latent, context, current_logits
                )
                energies.append(energy.item())
            
            # Compute refinement update
            # Input: current latent, context, and energy gradient approximation
            context_encoded = self.context_encoder(context)
            
            # Approximate energy gradient through finite differences
            noise = torch.randn_like(latent) * 0.01
            latent_plus = latent + noise
            latent_minus = latent - noise
            
            logits_plus = self.latent_decoder(latent_plus)
            logits_minus = self.latent_decoder(latent_minus)
            
            energy_plus, _ = self.compute_token_energy(latent_plus, context, logits_plus)
            energy_minus, _ = self.compute_token_energy(latent_minus, context, logits_minus)
            
            # Gradient approximation
            energy_grad = (energy_plus - energy_minus) / (2 * 0.01) * noise
            
            # Refinement network input
            refinement_input = torch.cat([
                latent,
                context_encoded,
                energy_grad.detach()
            ], dim=-1)
            
            # Compute update
            update = self.refinement_net(refinement_input)
            
            # Apply update with learning rate
            latent = latent - self.learning_rate * update
            
            # Add exploration noise (except last step)
            if step < self.n_refinement_steps - 1:
                latent = latent + torch.randn_like(latent) * self.exploration_noise
            
            if return_trajectory:
                trajectory.append(latent.clone())
        
        # Final decoding
        refined_logits = self.latent_decoder(latent)
        
        # Compute final energy
        final_energy, final_components = self.compute_token_energy(
            latent, context, refined_logits
        )
        
        stats = {
            'initial_energy': energies[0] if energies else 0,
            'final_energy': final_energy.item(),
            'energy_reduction': energies[0] - final_energy.item() if energies else 0,
            'energy_trajectory': energies,
            'final_components': final_components,
            'latent_norm': latent.norm(dim=-1).mean().item()
        }
        
        if return_trajectory:
            stats['trajectory'] = trajectory
        
        return refined_logits, stats
    
    def generate_attention_biases(
        self,
        latent: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate attention biases from PCN latent representation.
        
        Args:
            latent: PCN latent representation (batch, latent_dim)
            seq_len: Sequence length for bias expansion
            
        Returns:
            q_bias: Query bias (batch, seq_len, hidden_dim)
            k_bias: Key bias (batch, seq_len, hidden_dim)
            v_bias: Value bias (batch, seq_len, hidden_dim)
        """
        batch_size = latent.shape[0]
        
        # Generate biases
        q_bias = self.q_bias_generator(latent)  # (batch, hidden_dim)
        k_bias = self.k_bias_generator(latent)  # (batch, hidden_dim)
        v_bias = self.v_bias_generator(latent)  # (batch, hidden_dim)
        
        # Expand to sequence length
        # Use different strategies for different biases
        
        # Query bias: stronger at later positions (where we're predicting)
        position_weights = torch.linspace(0.5, 1.0, seq_len, device=latent.device)
        q_bias = q_bias.unsqueeze(1) * position_weights.unsqueeze(0).unsqueeze(-1)
        
        # Key bias: uniform across sequence
        k_bias = k_bias.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Value bias: decay from recent positions
        decay_weights = torch.exp(-torch.arange(seq_len, dtype=torch.float32, device=latent.device) * 0.1)
        decay_weights = decay_weights.flip(0)  # Recent positions have higher weight
        v_bias = v_bias.unsqueeze(1) * decay_weights.unsqueeze(0).unsqueeze(-1)
        
        return q_bias, k_bias, v_bias
    
    def forward(
        self,
        initial_logits: torch.Tensor,
        context: torch.Tensor,
        seq_len: Optional[int] = None,
        return_stats: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[Dict]]:
        """
        Forward pass: refine token distribution and generate attention biases.
        
        Args:
            initial_logits: Initial token predictions (batch, vocab_size)
            context: Context representation (batch, hidden_dim)
            seq_len: Sequence length for attention biases
            return_stats: Whether to return refinement statistics
            
        Returns:
            refined_logits: Refined token predictions
            attention_biases: (q_bias, k_bias, v_bias) for attention guidance
            stats: Optional refinement statistics
        """
        # Refine token distribution
        refined_logits, stats = self.refine_token_distribution(
            initial_logits, context, return_trajectory=False
        )
        
        # Encode refined distribution to latent for attention guidance
        encoder_input = torch.cat([context, refined_logits], dim=-1)
        guidance_latent = self.latent_encoder(encoder_input)
        
        # Generate attention biases
        if seq_len is None:
            seq_len = 1  # Default for single token prediction
            
        attention_biases = self.generate_attention_biases(guidance_latent, seq_len)
        
        if return_stats:
            return refined_logits, attention_biases, stats
        else:
            return refined_logits, attention_biases, None


class ContrastivePCNGuidance(PCNTokenGuidance):
    """
    Extended version with contrastive learning objective.
    
    Learns to assign lower energy to correct tokens and higher energy
    to incorrect tokens through contrastive loss.
    """
    
    def __init__(self, *args, margin: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        
    def contrastive_loss(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between positive and negative examples.
        
        Args:
            positive_logits: Logits for correct next token
            negative_logits: Logits for incorrect tokens
            context: Context representation
            
        Returns:
            Contrastive loss value
        """
        # Encode to latent
        pos_latent = self.latent_encoder(torch.cat([context, positive_logits], dim=-1))
        neg_latent = self.latent_encoder(torch.cat([context, negative_logits], dim=-1))
        
        # Compute energies
        pos_energy, _ = self.compute_token_energy(pos_latent, context, positive_logits)
        neg_energy, _ = self.compute_token_energy(neg_latent, context, negative_logits)
        
        # Contrastive loss: positive should have lower energy
        loss = F.relu(self.margin + pos_energy - neg_energy)
        
        return loss.mean()