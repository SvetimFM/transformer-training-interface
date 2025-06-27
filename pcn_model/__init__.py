"""
Predictive Coding Networks (PCNs) - PyTorch Implementation

A biologically-inspired alternative to traditional neural networks based on 
hierarchical predictive coding principles.

Based on "Introduction to Predictive Coding Networks for Machine Learning" by Monadillo (2025)
"""

from .layers import PCNLayer
from .network import PredictiveCodingNetwork
from .trainer import PCNTrainer
from .utils import compute_accuracy, track_energy

__version__ = "0.1.0"
__all__ = ["PCNLayer", "PredictiveCodingNetwork", "PCNTrainer", "compute_accuracy", "track_energy"]