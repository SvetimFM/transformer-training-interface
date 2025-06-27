# Predictive Coding Networks (PCN) - PyTorch Implementation

This is a clean PyTorch implementation of Predictive Coding Networks based on the paper "Introduction to Predictive Coding Networks for Machine Learning" by Monadillo (2025).

## Overview

Predictive Coding Networks (PCNs) are biologically-inspired neural networks that learn through hierarchical predictive coding. Unlike traditional feedforward networks trained with backpropagation, PCNs use local learning rules and alternating optimization between inference and learning phases.

## Key Features

- **Hierarchical Architecture**: Layers predict the activity of layers below
- **Local Learning Rules**: Biologically plausible Hebbian-like updates
- **Two-Phase Training**:
  - Inference: Update latent representations (fast)
  - Learning: Update weights (slow)
- **Energy-Based**: Minimize prediction errors across the hierarchy

## Architecture

```
Input (x^0) → Layer 1 (x^1) → Layer 2 (x^2) → ... → Layer L (x^L) → Output (y)
       ↑           ↑            ↑                          ↑
    W^(0)       W^(1)        W^(2)                    W^(L-1)
```

Each layer attempts to predict the layer below using learned weights W^(l).

## Usage

### Basic Example

```python
from pcn_model import PredictiveCodingNetwork, PCNTrainer

# Create model
layer_dims = [784, 256, 128, 10]  # MNIST example
output_dim = 10
model = PredictiveCodingNetwork(dims=layer_dims, output_dim=output_dim)

# Create trainer
trainer = PCNTrainer(
    model=model,
    eta_infer=0.05,   # Inference learning rate
    eta_learn=0.005,  # Weight learning rate
    T_infer=50,       # Inference steps
    T_learn=500       # Learning steps
)

# Train
history = trainer.train(train_loader, num_epochs=5)

# Test
top1_acc, top3_acc = trainer.test(test_loader)
```

### CIFAR-10 Training

To reproduce the paper's results on CIFAR-10:

```bash
python train_cifar10_pcn.py --num_epochs 4 --batch_size 500
```

Expected results:
- Top-1 Accuracy: ~99.92%
- Top-3 Accuracy: ~99.99%
- Training time: ~4 minutes on GPU

## Implementation Details

### PCNLayer
- Implements top-down predictions with learnable weights
- Uses ReLU activation by default
- No bias terms (following the paper)

### PredictiveCodingNetwork
- Orchestrates multiple PCN layers
- Includes linear readout for supervised tasks
- Provides methods for error computation

### PCNTrainer
- Implements alternating minimization algorithm
- Supports batch processing
- Tracks energy trajectories during training

## Hyperparameters

Key hyperparameters from the paper:
- `eta_infer`: 0.05 (inference rate)
- `eta_learn`: 0.005 (learning rate)
- `T_infer`: 50 (inference steps per sample)
- `T_learn`: 500 (learning steps per batch)
- `batch_size`: 500

## Mathematical Foundation

The network minimizes the total energy:
```
L = 1/2 Σ||ε^(l)||² + 1/2||ε^sup||²
```

Where:
- ε^(l) = x^(l) - x̂^(l) are prediction errors
- ε^sup = ŷ - y is the supervised error

Update rules:
- **Inference**: x^(l) ← x^(l) - η_infer * ∇_x^(l) L
- **Learning**: W^(l) ← W^(l) - η_learn * ∇_W^(l) L

## References

- Original paper: "Introduction to Predictive Coding Networks for Machine Learning" (Monadillo, 2025)
- GitHub: https://github.com/Monadillo/pcn-intro