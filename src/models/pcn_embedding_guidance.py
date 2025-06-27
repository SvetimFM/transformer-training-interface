"""
PCN-Guided Embedding Guidance for Token Prediction

This module implements PCN-based exploration and refinement of embedding
trajectories for improved sequence generation. Instead of directly predicting
tokens, it predicts and refines embedding sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class PCNEmbeddingPredictor(nn.Module):
    """
    Predicts future embedding sequences using PCN iterative refinement.
    
    This module generates multiple hypotheses for future embeddings and
    refines them through energy minimization to find coherent trajectories.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_future: int = 5,
        n_hypotheses: int = 3,
        n_iterations: int = 10,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        exploration_noise: float = 0.1,
        learning_rate: float = 0.1
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            n_future: Number of future embeddings to predict
            n_hypotheses: Number of embedding hypotheses to explore
            n_iterations: Number of PCN refinement iterations
            hidden_dim: Hidden dimension for processing (default: 2 * embed_dim)
            dropout: Dropout rate
            temperature: Temperature for hypothesis selection
            exploration_noise: Noise for hypothesis initialization
            learning_rate: Learning rate for PCN updates
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_future = n_future
        self.n_hypotheses = n_hypotheses
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.exploration_noise = exploration_noise
        self.learning_rate = learning_rate
        
        hidden_dim = hidden_dim or embed_dim * 2
        
        # Context encoder - processes current embeddings to create context
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Trajectory predictor - generates initial embedding trajectories
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim * n_future)
        )
        
        # Energy network - evaluates quality of embedding sequences
        self.energy_net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Refinement network - predicts updates for PCN iterations
        self.refinement_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # current, context, neighbor
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def compute_embedding_energy(
        self,
        embeddings: torch.Tensor,
        context: torch.Tensor,
        future_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy (quality score) for embedding trajectories.
        Lower energy = better trajectory.
        
        Args:
            embeddings: Predicted embeddings (batch, n_future, embed_dim)
            context: Context representation (batch, hidden_dim)
            future_true: True future embeddings for supervised energy (optional)
            
        Returns:
            Energy scores (batch,)
        """
        batch_size = embeddings.shape[0]
        
        # 1. Context coherence energy
        # How well do predictions align with context?
        context_emb = context[:, :self.embed_dim]  # Project to embed dim
        context_coherence = 0
        for i in range(self.n_future):
            combined = torch.cat([embeddings[:, i], context_emb], dim=-1)
            coherence = self.energy_net(combined).squeeze(-1)
            context_coherence = context_coherence + coherence
        context_coherence = context_coherence / self.n_future
        
        # 2. Trajectory smoothness energy
        # Penalize abrupt changes in embedding space
        if self.n_future > 1:
            diffs = embeddings[:, 1:] - embeddings[:, :-1]
            smoothness = diffs.norm(dim=-1).mean(dim=-1)
        else:
            smoothness = torch.zeros(batch_size, device=embeddings.device)
        
        # 3. Diversity energy (if multiple hypotheses)
        # Encourage different hypotheses to explore different regions
        diversity = torch.zeros(batch_size, device=embeddings.device)
        
        # 4. Supervised energy (if true future provided)
        if future_true is not None:
            # Handle case where embeddings might be from multiple hypotheses
            if embeddings.shape[0] != future_true.shape[0]:
                # Expand future_true to match number of hypotheses
                n_hypotheses = embeddings.shape[0] // future_true.shape[0]
                future_true_expanded = future_true.repeat(n_hypotheses, 1, 1)
                supervision = F.mse_loss(embeddings, future_true_expanded, reduction='none')
            else:
                supervision = F.mse_loss(embeddings, future_true, reduction='none')
            supervision = supervision.mean(dim=[1, 2])
        else:
            supervision = torch.zeros(batch_size, device=embeddings.device)
        
        # Combine energies
        total_energy = context_coherence + 0.1 * smoothness + supervision
        
        return total_energy
    
    def refine_embeddings(
        self,
        embeddings: torch.Tensor,
        context: torch.Tensor,
        future_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Refine embedding predictions through PCN iterations.
        
        Args:
            embeddings: Initial embeddings (batch, n_hypotheses, n_future, embed_dim)
            context: Context representation (batch, hidden_dim)
            future_true: True future embeddings (optional)
            
        Returns:
            Refined embeddings (batch, n_hypotheses, n_future, embed_dim)
        """
        batch_size, n_hyp, n_future, embed_dim = embeddings.shape
        
        # Flatten for processing
        embeddings_flat = embeddings.view(batch_size * n_hyp, n_future, embed_dim)
        context_expanded = context.unsqueeze(1).expand(-1, n_hyp, -1)
        context_flat = context_expanded.reshape(batch_size * n_hyp, -1)
        
        # PCN refinement iterations
        for iteration in range(self.n_iterations):
            # Compute current energy
            with torch.no_grad():
                energy = self.compute_embedding_energy(
                    embeddings_flat, context_flat, future_true
                )
            
            # Refine each embedding in the trajectory
            updates = []
            for t in range(n_future):
                current_emb = embeddings_flat[:, t]
                
                # Get neighboring embeddings for context
                if t > 0:
                    prev_emb = embeddings_flat[:, t-1]
                else:
                    prev_emb = current_emb
                    
                if t < n_future - 1:
                    next_emb = embeddings_flat[:, t+1]
                else:
                    next_emb = current_emb
                
                # Compute refinement update
                neighbor_context = (prev_emb + next_emb) / 2
                refine_input = torch.cat([
                    current_emb,
                    context_flat[:, :embed_dim],
                    neighbor_context
                ], dim=-1)
                
                update = self.refinement_net(refine_input)
                updates.append(update)
            
            # Apply updates
            updates = torch.stack(updates, dim=1)
            embeddings_flat = embeddings_flat - self.learning_rate * updates
            
            # Add small noise for exploration (except last iteration)
            if iteration < self.n_iterations - 1:
                noise = torch.randn_like(embeddings_flat) * self.exploration_noise * 0.1
                embeddings_flat = embeddings_flat + noise
        
        # Reshape back
        refined = embeddings_flat.view(batch_size, n_hyp, n_future, embed_dim)
        return refined
    
    def forward(
        self,
        current_embeddings: torch.Tensor,
        future_true: Optional[torch.Tensor] = None,
        return_all_hypotheses: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict future embedding sequences.
        
        Args:
            current_embeddings: Current embeddings (batch, seq_len, embed_dim)
            future_true: True future embeddings for training (batch, n_future, embed_dim)
            return_all_hypotheses: Return all hypotheses or just the best
            
        Returns:
            predictions: Predicted embeddings (batch, n_future, embed_dim)
            stats: Dictionary with energy scores and other statistics
        """
        batch_size, seq_len, embed_dim = current_embeddings.shape
        device = current_embeddings.device
        
        # Encode context from current embeddings
        # Use mean pooling over sequence
        context = current_embeddings.mean(dim=1)
        context = self.context_encoder(context)
        
        # Generate initial trajectory predictions
        initial_trajectories = []
        for _ in range(self.n_hypotheses):
            # Predict trajectory
            trajectory = self.trajectory_predictor(context)
            trajectory = trajectory.view(batch_size, self.n_future, embed_dim)
            
            # Add exploration noise
            noise = torch.randn_like(trajectory) * self.exploration_noise
            trajectory = trajectory + noise
            
            initial_trajectories.append(trajectory)
        
        # Stack hypotheses
        hypotheses = torch.stack(initial_trajectories, dim=1)  # (batch, n_hyp, n_future, embed_dim)
        
        # Refine through PCN iterations
        refined_hypotheses = self.refine_embeddings(hypotheses, context, future_true)
        
        # Compute final energies for selection
        energies = []
        for h in range(self.n_hypotheses):
            energy = self.compute_embedding_energy(
                refined_hypotheses[:, h], context, future_true
            )
            energies.append(energy)
        
        energies = torch.stack(energies, dim=1)  # (batch, n_hypotheses)
        
        # Select best hypothesis or combine
        if return_all_hypotheses:
            predictions = refined_hypotheses
        else:
            # Soft selection using energy-based probabilities
            probs = F.softmax(-energies / self.temperature, dim=1)
            predictions = torch.sum(
                refined_hypotheses * probs.unsqueeze(-1).unsqueeze(-1),
                dim=1
            )
        
        # Prepare statistics
        stats = {
            'energies': energies,
            'best_energy': energies.min(dim=1).values,
            'selection_probs': probs if not return_all_hypotheses else None
        }
        
        return predictions, stats


class EmbeddingRefinementLayer(nn.Module):
    """
    Refines embeddings through PCN dynamics before token projection.
    
    This layer can be inserted between transformer blocks and output projection
    to improve embedding quality through iterative refinement.
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_iterations: int = 5,
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        
        hidden_dim = hidden_dim or embed_dim * 2
        
        # Self-prediction network
        self.self_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Context integration
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Refine embeddings through PCN iterations.
        
        Args:
            embeddings: Input embeddings (batch, seq_len, embed_dim)
            
        Returns:
            Refined embeddings (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Initialize refined embeddings
        refined = embeddings.clone()
        
        # PCN refinement iterations
        for _ in range(self.n_iterations):
            # Self-prediction
            predicted = self.self_predictor(refined)
            
            # Compute prediction error
            error = embeddings - predicted
            
            # Context-aware update
            # For each position, consider its neighbors
            context_updates = []
            for i in range(seq_len):
                if i > 0 and i < seq_len - 1:
                    # Use previous and next embeddings as context
                    context = (refined[:, i-1] + refined[:, i+1]) / 2
                elif i == 0 and seq_len > 1:
                    context = refined[:, i+1]
                elif i == seq_len - 1 and seq_len > 1:
                    context = refined[:, i-1]
                else:
                    context = refined[:, i]
                
                # Compute gated update
                gate_input = torch.cat([refined[:, i], context], dim=-1)
                gate = self.context_gate(gate_input)
                
                # Apply gated error correction
                update = gate * error[:, i]
                context_updates.append(refined[:, i] + self.learning_rate * update)
            
            # Update refined embeddings
            refined = torch.stack(context_updates, dim=1)
        
        return refined


class MultiScaleEmbeddingPredictor(nn.Module):
    """
    Hierarchical embedding prediction at multiple temporal scales.
    
    Predicts long-range structure first, then refines to detailed predictions.
    """
    
    def __init__(
        self,
        embed_dim: int,
        scales: List[int] = [20, 10, 5],
        n_iterations: int = 10,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.scales = scales
        
        # Predictors for each scale
        self.scale_predictors = nn.ModuleList([
            PCNEmbeddingPredictor(
                embed_dim=embed_dim,
                n_future=scale,
                n_iterations=n_iterations,
                hidden_dim=hidden_dim
            )
            for scale in scales
        ])
        
        # Scale combination network
        self.scale_combiner = nn.Sequential(
            nn.Linear(embed_dim * len(scales), hidden_dim or embed_dim * 2),
            nn.LayerNorm(hidden_dim or embed_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim or embed_dim * 2, embed_dim)
        )
        
    def forward(
        self,
        current_embeddings: torch.Tensor,
        target_length: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict embeddings at multiple scales and combine.
        
        Args:
            current_embeddings: Current embeddings (batch, seq_len, embed_dim)
            target_length: Desired output length
            
        Returns:
            predictions: Combined predictions (batch, target_length, embed_dim)
            stats: Statistics from each scale
        """
        batch_size = current_embeddings.shape[0]
        
        scale_predictions = []
        scale_stats = {}
        
        # Get predictions at each scale
        for i, (scale, predictor) in enumerate(zip(self.scales, self.scale_predictors)):
            pred, stats = predictor(current_embeddings)
            
            # Interpolate to target length if needed
            if scale != target_length:
                # Simple linear interpolation
                pred = F.interpolate(
                    pred.transpose(1, 2),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_predictions.append(pred)
            scale_stats[f'scale_{scale}'] = stats
        
        # Combine scale predictions
        combined_input = torch.cat(scale_predictions, dim=-1)
        combined_pred = self.scale_combiner(combined_input)
        
        return combined_pred, scale_stats