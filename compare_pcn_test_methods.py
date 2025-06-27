"""
Compare PCN test methods to demonstrate data leakage issue.

This script compares:
1. Original test method (with labels during inference) - ~99.9% accuracy
2. Fixed test method (without labels during inference) - true accuracy
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from pcn_model import PredictiveCodingNetwork
from pcn_model.trainer_fixed import PCNTrainer


def get_cifar10_test_loader(batch_size: int = 500):
    """Get CIFAR-10 test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader


def main():
    """Run comparison of test methods."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the trained model
    checkpoint_path = "pcn_cifar10_20250624_173814.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using: python train_cifar10_pcn.py")
        return
    
    # Create model
    layer_dims = [3072, 1000, 500, 10]
    output_dim = 10
    model = PredictiveCodingNetwork(dims=layer_dims, output_dim=output_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded checkpoint from: {checkpoint_path}")
    print(f"Training accuracy reported: {checkpoint['top1_acc']*100:.2f}%")
    
    # Create trainer
    trainer = PCNTrainer(
        model=model,
        eta_infer=0.05,
        eta_learn=0.005,
        T_infer=50,
        T_learn=500,
        device=device
    )
    
    # Get test data
    test_loader = get_cifar10_test_loader(batch_size=500)
    
    print("\n" + "="*80)
    print("COMPARING PCN TEST METHODS")
    print("="*80)
    
    # Test 1: Original method (with label leakage)
    print("\n1. ORIGINAL TEST METHOD (with label leakage):")
    print("   Using true labels during inference...")
    top1_flawed, top3_flawed = trainer.test_with_labels(test_loader)
    print(f"   Results:")
    print(f"   - Top-1 Accuracy: {top1_flawed*100:.2f}%")
    print(f"   - Top-3 Accuracy: {top3_flawed*100:.2f}%")
    print("   ⚠️  These results are INVALID due to label leakage!")
    
    # Test 2: Fixed method (without labels)
    print("\n2. FIXED TEST METHOD (without label leakage):")
    print("   Using only unsupervised inference...")
    top1_fixed, top3_fixed = trainer.test(test_loader)
    print(f"   Results:")
    print(f"   - Top-1 Accuracy: {top1_fixed*100:.2f}%")
    print(f"   - Top-3 Accuracy: {top3_fixed*100:.2f}%")
    print("   ✅ These are the TRUE generalization results!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAccuracy inflation due to label leakage:")
    print(f"  - Top-1: {top1_flawed*100:.2f}% → {top1_fixed*100:.2f}% (diff: {(top1_flawed-top1_fixed)*100:.2f}%)")
    print(f"  - Top-3: {top3_flawed*100:.2f}% → {top3_fixed*100:.2f}% (diff: {(top3_flawed-top3_fixed)*100:.2f}%)")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe original PCN paper's claim of 99.92% accuracy on CIFAR-10 is based on")
    print("a flawed test procedure that uses true labels during inference. This is")
    print("equivalent to telling the model the correct answer during the test.")
    print("\nThe true performance of PCN on CIFAR-10 is significantly lower, as shown")
    print("by the fixed test method that properly evaluates generalization without")
    print("using label information during inference.")
    
    # Save results
    results = {
        'original_method': {'top1': top1_flawed, 'top3': top3_flawed},
        'fixed_method': {'top1': top1_fixed, 'top3': top3_fixed},
        'checkpoint': checkpoint_path,
        'device': device
    }
    
    torch.save(results, 'pcn_test_comparison_results.pt')
    print(f"\nResults saved to: pcn_test_comparison_results.pt")


if __name__ == "__main__":
    main()