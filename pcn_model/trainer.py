"""
PCN Trainer Implementation

Implements the alternating minimization training procedure for Predictive Coding Networks.
"""

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .network import PredictiveCodingNetwork


class PCNTrainer:
    """
    Trainer class for Predictive Coding Networks.
    
    Implements the two-phase alternating optimization:
    1. Inference phase: Update latents with frozen weights
    2. Learning phase: Update weights with frozen latents
    """
    
    def __init__(
        self,
        model: PredictiveCodingNetwork,
        eta_infer: float = 0.05,
        eta_learn: float = 0.005,
        T_infer: int = 50,
        T_learn: int = 500,
        device: str = 'cuda'
    ):
        """
        Initialize PCN trainer.
        
        Args:
            model: The PredictiveCodingNetwork to train
            eta_infer: Learning rate for inference phase
            eta_learn: Learning rate for learning phase
            T_infer: Number of inference steps per sample
            T_learn: Number of learning steps per batch
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.eta_infer = eta_infer
        self.eta_learn = eta_learn
        self.T_infer = T_infer
        self.T_learn = T_learn
        self.device = torch.device(device)
        
        self.model.to(self.device)
        
    def train_epoch(
        self, 
        data_loader: DataLoader,
        track_energy: bool = True
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader for training data
            track_energy: Whether to track energy trajectories
            
        Returns:
            epoch_energies: Total energy trajectories for each batch
            epoch_supervised_energies: Supervised energy trajectories for each batch
        """
        self.model.train()
        
        epoch_energies = []
        epoch_supervised_energies = []
        
        for x_batch, y_batch in tqdm(data_loader):
            batch_energies, batch_supervised_energies = self.train_batch(
                x_batch, y_batch, track_energy
            )
            
            if track_energy:
                epoch_energies.append(batch_energies)
                epoch_supervised_energies.append(batch_supervised_energies)
                
        return epoch_energies, epoch_supervised_energies
    
    def train_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        track_energy: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Train on a single batch using the PCN algorithm.
        
        Args:
            x_batch: Input batch (B, C, H, W) or (B, D)
            y_batch: Target labels (B,)
            track_energy: Whether to track energy over time
            
        Returns:
            batch_energies: Total energy at each step
            batch_supervised_energies: Supervised energy at each step
        """
        B = x_batch.size(0)
        d_0 = self.model.dims[0]
        
        batch_energies = []
        batch_supervised_energies = []
        
        # Prepare inputs
        x_batch = x_batch.view(B, d_0).to(self.device)
        y_batch = F.one_hot(y_batch, num_classes=self.model.readout.out_features).float().to(self.device)
        
        # Initialize: inputs + latents
        inputs_latents = [x_batch] + self.model.init_latents(B, self.device)
        
        # Get weight references for in-place updates
        weights = [layer.W for layer in self.model.layers] + [self.model.readout.weight]
        
        # Initial prediction for energy at t=0
        errors, gain_modulated_errors = self.model.compute_errors(inputs_latents)
        y_hat = self.model.readout(inputs_latents[-1])
        eps_sup = y_hat - y_batch
        eps_L = eps_sup @ weights[-1]
        errors_extended = errors + [eps_L]
        
        # Record initial energy
        if track_energy:
            supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
            latent_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
            batch_supervised_energies.append(supervised_energy)
            batch_energies.append(latent_energy + supervised_energy)
        
        # INFERENCE PHASE - Update latents
        with torch.no_grad(), autocast(device_type='cuda'):
            for t in range(1, self.T_infer + 1):
                # Update latents using current errors
                for l in range(1, self.model.L + 1):  # l = 1, ..., L
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                    inputs_latents[l] -= self.eta_infer * grad_Xl
                
                # Recompute errors for next step and energy tracking
                errors, gain_modulated_errors = self.model.compute_errors(inputs_latents)
                y_hat = self.model.readout(inputs_latents[-1])
                eps_sup = y_hat - y_batch
                eps_L = eps_sup @ weights[-1]
                errors_extended = errors + [eps_L]
                
                # Track energy
                if track_energy:
                    supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
                    latent_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    batch_supervised_energies.append(supervised_energy)
                    batch_energies.append(latent_energy + supervised_energy)
        
        # LEARNING PHASE - Update weights
        with torch.no_grad():  # No autocast to keep precision in weight updates
            for t in range(self.T_infer + 1, self.T_learn + self.T_infer + 1):
                # Update generative weights
                for l in range(self.model.L):  # l = 0, ..., L-1
                    grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l+1]) / B
                    weights[l] -= self.eta_learn * grad_Wl
                
                # Update readout weights
                grad_Wout = eps_sup.T @ inputs_latents[-1] / B
                weights[-1] -= self.eta_learn * grad_Wout
                
                # Recompute errors for next step
                errors, gain_modulated_errors = self.model.compute_errors(inputs_latents)
                y_hat = self.model.readout(inputs_latents[-1])
                eps_sup = y_hat - y_batch
                
                # Track energy
                if track_energy:
                    supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
                    latent_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    batch_supervised_energies.append(supervised_energy)
                    batch_energies.append(latent_energy + supervised_energy)
        
        return batch_energies, batch_supervised_energies
    
    def train(
        self,
        data_loader: DataLoader,
        num_epochs: int,
        track_energy: bool = True
    ) -> Dict[str, List]:
        """
        Train the model for multiple epochs.
        
        Args:
            data_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            track_energy: Whether to track energy trajectories
            
        Returns:
            Dictionary containing training history
        """
        energy_history = []
        supervised_energy_history = []
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1} / {num_epochs}")
            
            epoch_energies, epoch_supervised_energies = self.train_epoch(
                data_loader, track_energy
            )
            
            if track_energy:
                energy_history.append(epoch_energies)
                supervised_energy_history.append(epoch_supervised_energies)
        
        return {
            'energy_history': energy_history,
            'supervised_energy_history': supervised_energy_history
        }
    
    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Test the model and compute accuracies.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            top1_acc: Top-1 accuracy
            top3_acc: Top-3 accuracy
        """
        self.model.eval()
        
        total = 0
        top1_correct = 0
        top3_correct = 0
        
        for x_batch, y_batch in tqdm(test_loader):
            B = x_batch.size(0)
            total += B
            d_0 = self.model.dims[0]
            
            # Prepare inputs
            x_batch = x_batch.view(B, d_0).to(self.device)
            y_labels = y_batch.to(self.device)
            y_batch = F.one_hot(y_labels, num_classes=self.model.readout.out_features).float().to(self.device)
            
            # Initialize latents
            inputs_latents = [x_batch] + self.model.init_latents(B, self.device)
            weights = [layer.W for layer in self.model.layers] + [self.model.readout.weight]
            
            # Run inference
            with autocast(device_type='cuda'):
                for t in range(1, self.T_infer + 1):
                    errors, gain_modulated_errors = self.model.compute_errors(inputs_latents)
                    y_hat = self.model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch
                    eps_L = eps_sup @ weights[-1]
                    errors_extended = errors + [eps_L]
                    
                    for l in range(1, self.model.L + 1):
                        grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                        inputs_latents[l] -= self.eta_infer * grad_Xl
            
            # Compute predictions
            logits = self.model.readout(inputs_latents[-1])
            
            # Top-1 accuracy
            preds1 = logits.argmax(dim=1)
            top1_correct += (preds1 == y_labels).sum().item()
            
            # Top-3 accuracy
            _, preds3 = logits.topk(3, dim=1)
            top3_correct += (preds3 == y_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        top1_acc = top1_correct / total
        top3_acc = top3_correct / total
        
        return top1_acc, top3_acc