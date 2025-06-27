"""
Train a Predictive Coding Network on CIFAR-10

This script reproduces the results from "Introduction to Predictive Coding Networks 
for Machine Learning" by Monadillo (2025), achieving 99.92% accuracy on CIFAR-10.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import os

from pcn_model import PredictiveCodingNetwork, PCNTrainer
from pcn_model.utils import plot_energy_trajectories, save_checkpoint


def get_cifar10_loaders(batch_size: int = 500, num_workers: int = 4):
    """
    Create CIFAR-10 data loaders with normalization.
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, test_loader
    """
    # CIFAR-10 normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download and prepare datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    print(f"Dataset loaded: {len(trainset)} train samples, {len(testset)} test samples")
    print(f"Batch size: {batch_size}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def main(args):
    """Main training function."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    layer_dims = [3072, 1000, 500, 10]  # Input: 32x32x3 = 3072
    output_dim = 10  # 10 classes
    
    model = PredictiveCodingNetwork(dims=layer_dims, output_dim=output_dim)
    print(f"\nModel architecture:")
    print(model)
    
    # Create trainer
    trainer = PCNTrainer(
        model=model,
        eta_infer=args.eta_infer,
        eta_learn=args.eta_learn,
        T_infer=args.T_infer,
        T_learn=args.T_learn,
        device=device
    )
    
    # Training
    print(f"\nStarting PCN training for {args.num_epochs} epochs...")
    print(f"Hyperparameters:")
    print(f"  - eta_infer: {args.eta_infer}")
    print(f"  - eta_learn: {args.eta_learn}")
    print(f"  - T_infer: {args.T_infer}")
    print(f"  - T_learn: {args.T_learn}")
    
    history = trainer.train(
        data_loader=train_loader,
        num_epochs=args.num_epochs,
        track_energy=args.track_energy
    )
    
    print("\nTraining completed!")
    
    # Test the model
    print("\nEvaluating on test set...")
    top1_acc, top3_acc = trainer.test(test_loader)
    print(f"\nTest Results:")
    print(f"  - Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  - Top-3 Accuracy: {top3_acc*100:.2f}%")
    
    # Save model
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"pcn_cifar10_{timestamp}.pth"
        save_checkpoint(
            model=model,
            optimizer=None,
            epoch=args.num_epochs,
            checkpoint_path=checkpoint_path,
            additional_info={
                'top1_acc': top1_acc,
                'top3_acc': top3_acc,
                'hyperparameters': {
                    'eta_infer': args.eta_infer,
                    'eta_learn': args.eta_learn,
                    'T_infer': args.T_infer,
                    'T_learn': args.T_learn,
                    'batch_size': args.batch_size,
                    'num_epochs': args.num_epochs
                }
            }
        )
    
    # Plot energy trajectories
    if args.track_energy and args.plot_energy:
        print("\nPlotting energy trajectories...")
        plot_energy_trajectories(
            energy_history=history['energy_history'],
            T_infer=args.T_infer,
            T_learn=args.T_learn,
            title="PCN Training Energy Trajectories",
            use_plotly=args.use_plotly
        )
        
        plot_energy_trajectories(
            energy_history=history['supervised_energy_history'],
            T_infer=args.T_infer,
            T_learn=args.T_learn,
            title="PCN Supervised Energy Trajectories",
            use_plotly=args.use_plotly
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCN on CIFAR-10")
    
    # Model hyperparameters
    parser.add_argument('--eta_infer', type=float, default=0.05,
                        help='Inference learning rate (default: 0.05)')
    parser.add_argument('--eta_learn', type=float, default=0.005,
                        help='Learning rate for weights (default: 0.005)')
    parser.add_argument('--T_infer', type=int, default=50,
                        help='Number of inference steps (default: 50)')
    parser.add_argument('--T_learn', type=int, default=500,
                        help='Number of learning steps (default: 500)')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size (default: 500)')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of epochs (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Options
    parser.add_argument('--track_energy', action='store_true', default=True,
                        help='Track energy during training')
    parser.add_argument('--plot_energy', action='store_true', default=False,
                        help='Plot energy trajectories after training')
    parser.add_argument('--use_plotly', action='store_true', default=False,
                        help='Use plotly for interactive plots')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save trained model')
    
    args = parser.parse_args()
    
    # Ensure T_learn equals batch_size as in the paper
    if args.T_learn != args.batch_size:
        print(f"Warning: Setting T_learn = batch_size ({args.batch_size}) as per the paper")
        args.T_learn = args.batch_size
    
    main(args)