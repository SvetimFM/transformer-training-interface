"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.
Based on the paper: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that can be attached to any linear layer.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 8, 
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create low-rank matrices A and B with explicit naming
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Register parameters explicitly for easier identification
        self.register_parameter('lora_down', self.lora_A)
        self.register_parameter('lora_up', self.lora_B)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize LoRA weights using Kaiming uniform distribution."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # Apply dropout to input
        x = self.dropout(x)
        # Low-rank adaptation: (x @ A^T) @ B^T * scaling
        # Ensure LoRA parameters are on the same device as input
        lora_output = (x @ self.lora_A.T.to(x.device)) @ self.lora_B.T.to(x.device)
        return lora_output * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Wraps an existing linear layer and adds LoRA parameters.
    """
    def __init__(
        self, 
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        enabled: bool = True
    ):
        super().__init__()
        self.base_layer = linear_layer
        self.enabled = enabled
        
        if enabled:
            self.lora = LoRALayer(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base layer and LoRA."""
        base_output = self.base_layer(x)
        
        if self.enabled and hasattr(self, 'lora'):
            return base_output + self.lora(x)
        return base_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for inference."""
        if self.enabled and hasattr(self, 'lora'):
            # W' = W + BA * scaling
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_layer.weight.data += delta_w
            
    def extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights for saving."""
        if self.enabled and hasattr(self, 'lora'):
            return {
                'lora_A': self.lora.lora_A.data,
                'lora_B': self.lora.lora_B.data,
                'alpha': self.lora.alpha,
                'rank': self.lora.rank
            }
        return {}
    
    def load_lora_weights(self, weights: Dict[str, torch.Tensor]):
        """Load LoRA weights from dictionary."""
        if self.enabled and hasattr(self, 'lora'):
            self.lora.lora_A.data = weights['lora_A']
            self.lora.lora_B.data = weights['lora_B']
            # Verify dimensions match
            assert self.lora.alpha == weights['alpha']
            assert self.lora.rank == weights['rank']


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to modify
        target_modules: List of module names to apply LoRA to (e.g., ['query', 'value'])
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability
        
    Returns:
        Modified model with LoRA layers
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent = model if parent_name == '' else model.get_submodule(parent_name)
                
                # Replace with LoRA-wrapped layer
                lora_layer = LoRALinear(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                setattr(parent, attr_name, lora_layer)
                
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """Save all LoRA weights from a model."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            weights = module.extract_lora_weights()
            if weights:
                lora_state[name] = weights
    torch.save(lora_state, path)
    

def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights into a model."""
    lora_state = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            module.load_lora_weights(lora_state[name])