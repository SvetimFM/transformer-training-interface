"""
Utility functions for PCN training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
try:
    import plotly.graph_objects as go
    from plotly import colors
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Model predictions (logits), shape (N, num_classes)
        targets: True labels, shape (N,)
        top_k: k for top-k accuracy
        
    Returns:
        Accuracy as a percentage
    """
    _, pred_indices = predictions.topk(top_k, dim=1)
    correct = (pred_indices == targets.unsqueeze(1)).any(dim=1).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def track_energy(
    errors: List[torch.Tensor],
    eps_sup: Optional[torch.Tensor] = None
) -> Tuple[float, float]:
    """
    Compute total and supervised energy from errors.
    
    Args:
        errors: List of prediction errors [E^(0), ..., E^(L-1)]
        eps_sup: Supervised error (optional)
        
    Returns:
        total_energy: Total energy (latent + supervised)
        supervised_energy: Supervised energy only
    """
    # Compute latent energy: 1/2 * sum(||E^(l)||^2)
    latent_energy = 0.5 * sum(e.pow(2).sum().item() for e in errors)
    
    # Compute supervised energy if provided
    supervised_energy = 0.0
    if eps_sup is not None:
        supervised_energy = 0.5 * eps_sup.pow(2).sum().item()
    
    total_energy = latent_energy + supervised_energy
    
    return total_energy, supervised_energy


def plot_energy_trajectories(
    energy_history: List[List[List[float]]],
    T_infer: int,
    T_learn: int,
    title: str = "Energy Trajectories",
    use_plotly: bool = True
):
    """
    Plot energy trajectories over training.
    
    Args:
        energy_history: Nested list [epochs][batches][steps]
        T_infer: Number of inference steps
        T_learn: Number of learning steps
        title: Plot title
        use_plotly: Whether to use plotly (interactive) or matplotlib
    """
    if use_plotly and PLOTLY_AVAILABLE:
        _plot_energy_plotly(energy_history, T_infer, T_learn, title)
    else:
        _plot_energy_matplotlib(energy_history, T_infer, T_learn, title)


def _plot_energy_matplotlib(
    energy_history: List[List[List[float]]],
    T_infer: int,
    T_learn: int,
    title: str
):
    """Plot energy trajectories using matplotlib."""
    plt.figure(figsize=(12, 8))
    
    num_epochs = len(energy_history)
    colors_list = plt.cm.viridis(np.linspace(0, 1, num_epochs))
    
    for epoch_idx, epoch_energies in enumerate(energy_history):
        color = colors_list[epoch_idx]
        for batch_energies in epoch_energies:
            steps = list(range(len(batch_energies)))
            plt.plot(steps, batch_energies, color=color, alpha=0.3, linewidth=1)
    
    # Add vertical line separating inference and learning
    plt.axvline(x=T_infer + 0.5, color='black', linestyle='--', label='Inference end')
    
    plt.xlabel('Step (Inference then Learning)')
    plt.ylabel('Energy')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_energy_plotly(
    energy_history: List[List[List[float]]],
    T_infer: int,
    T_learn: int,
    title: str
):
    """Plot interactive energy trajectories using plotly."""
    num_epochs = len(energy_history)
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i/(num_epochs-1) if num_epochs>1 else 0 for i in range(num_epochs)]
    )
    
    fig = go.Figure()
    
    for epoch_idx, epoch_energies in enumerate(energy_history):
        color = epoch_colors[epoch_idx]
        for batch_idx, batch_vals in enumerate(epoch_energies):
            steps = list(range(len(batch_vals)))
            customdata = [[epoch_idx+1, batch_idx+1] for _ in steps]
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=batch_vals,
                mode='lines',
                line=dict(color=color, width=1),
                hovertemplate=(
                    'Epoch %{customdata[0]}<br>'
                    'Batch %{customdata[1]}<br>'
                    'Step %{x}<br>'
                    'Energy %{y:.4f}<extra></extra>'
                ),
                customdata=customdata,
                showlegend=False
            ))
    
    # Vertical separator
    fig.add_vline(
        x=T_infer + 0.5,
        line_dash='dash',
        line_color='black',
        annotation_text='Inference end',
        annotation_position='top right'
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Step t (Inference then Learning)',
        yaxis_title='Energy',
        yaxis_type='log'
    )
    
    fig.show()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    checkpoint_path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        checkpoint_path: Path to save checkpoint
        additional_info: Any additional info to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with additional info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint