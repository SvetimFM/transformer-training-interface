"""
Specific Hybrid PCN-Transformer Architectures

This module provides ready-to-use implementations of the 5 proposed hybrid architectures:
1. PCN-FF: Transformer with PCN feedforward
2. Alternating: Alternating attention and PCN layers
3. Hierarchical: PCN feature extraction + Transformer
4. Dual-Stream: Parallel PCN and Transformer paths
5. PCN-Positional: Transformer with PCN positional encoding
"""

from typing import Dict, Optional
from models.hybrid_pcn_transformer import HybridPCNTransformer


class PCNFeedForwardTransformer(HybridPCNTransformer):
    """
    Architecture 1: Standard Transformer with PCN FeedForward layers.
    
    Replaces the standard 2-layer FFN with hierarchical PCN processing
    while keeping multi-head attention for global context.
    
    Benefits:
    - Biological plausibility in local computations
    - Maintains transformer's global attention
    - Drop-in replacement for standard transformer
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.2,
        pcn_layers: int = 2,
        pcn_inference_steps: int = 5,
        pcn_inference_lr: float = 0.1,
        device: str = "cuda"
    ):
        config = {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre',
            'device': device,
            'pcn_config': {
                'n_pcn_layers': pcn_layers,
                'inference_steps': pcn_inference_steps,
                'inference_lr': pcn_inference_lr
            }
        }
        
        super().__init__(
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            architecture="pcn_ff",
            config=config
        )


class AlternatingPCNTransformer(HybridPCNTransformer):
    """
    Architecture 2: Alternating Attention and PCN layers.
    
    Odd layers: Multi-head attention for global dependencies
    Even layers: PCN processing for local hierarchical features
    
    Benefits:
    - Balanced global/local processing
    - Reduced computational cost (no FFN after attention)
    - Natural separation of concerns
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,  # Should be even for balance
        dropout: float = 0.2,
        pcn_layers: int = 3,
        pcn_inference_steps: int = 5,
        pcn_inference_lr: float = 0.1,
        device: str = "cuda"
    ):
        config = {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre',
            'device': device,
            'pcn_config': {
                'n_pcn_layers': pcn_layers,
                'inference_steps': pcn_inference_steps,
                'inference_lr': pcn_inference_lr
            }
        }
        
        super().__init__(
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            architecture="alternating",
            config=config
        )


class HierarchicalPCNTransformer(HybridPCNTransformer):
    """
    Architecture 3: Hierarchical PCN followed by Transformer.
    
    Bottom layers: PCN hierarchy for feature extraction
    Top layers: Transformer for sequence modeling
    
    Benefits:
    - Biologically-inspired feature learning
    - Powerful sequence modeling on top
    - Clear separation of representation and sequence learning
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_pcn_layers: int = 3,
        n_transformer_layers: int = 3,
        dropout: float = 0.2,
        pcn_inference_steps: int = 10,
        pcn_inference_lr: float = 0.1,
        device: str = "cuda"
    ):
        config = {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_pcn_layers + n_transformer_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre',
            'device': device,
            'pcn_config': {
                'inference_steps': pcn_inference_steps,
                'inference_lr': pcn_inference_lr
            }
        }
        
        super().__init__(
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            architecture="hierarchical",
            config=config
        )


class DualStreamPCNTransformer(HybridPCNTransformer):
    """
    Architecture 4: Dual-Stream with PCN and Transformer paths.
    
    Stream 1: Full transformer stack
    Stream 2: PCN hierarchical processing
    Fusion: Learnable gating mechanism
    
    Benefits:
    - Best of both worlds
    - Adaptive mixing based on input
    - Can learn when to use global vs local processing
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.2,
        pcn_depth: int = 3,
        device: str = "cuda"
    ):
        config = {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre',
            'device': device,
            'pcn_config': {
                'n_layers': pcn_depth
            }
        }
        
        super().__init__(
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            architecture="dual_stream",
            config=config
        )


class PCNPositionalTransformer(HybridPCNTransformer):
    """
    Architecture 5: Transformer with PCN-based Positional Encoding.
    
    Uses PCN to learn adaptive positional representations
    instead of fixed sinusoidal or learned embeddings.
    
    Benefits:
    - Adaptive position encoding
    - Can learn complex positional patterns
    - Potentially better for variable-length sequences
    """
    
    def __init__(
        self,
        vocab_size: int,
        batch_size: int,
        block_size: int,
        n_embed: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.2,
        n_position_pcn_layers: int = 2,
        device: str = "cuda"
    ):
        config = {
            'n_embed': n_embed,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'use_layer_norm': True,
            'use_residual': True,
            'norm_position': 'pre',
            'device': device,
            'pcn_config': {
                'n_position_layers': n_position_pcn_layers
            }
        }
        
        super().__init__(
            vocab_size=vocab_size,
            batch_size=batch_size,
            block_size=block_size,
            architecture="pcn_positional",
            config=config
        )


# Factory function for easy instantiation
def create_hybrid_model(
    model_type: str,
    vocab_size: int,
    batch_size: int,
    block_size: int,
    **kwargs
) -> HybridPCNTransformer:
    """
    Factory function to create hybrid models.
    
    Args:
        model_type: One of ['pcn_ff', 'alternating', 'hierarchical', 
                          'dual_stream', 'pcn_positional']
        vocab_size: Vocabulary size
        batch_size: Batch size
        block_size: Maximum sequence length
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated hybrid model
    """
    
    models = {
        'pcn_ff': PCNFeedForwardTransformer,
        'alternating': AlternatingPCNTransformer,
        'hierarchical': HierarchicalPCNTransformer,
        'dual_stream': DualStreamPCNTransformer,
        'pcn_positional': PCNPositionalTransformer
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from {list(models.keys())}")
    
    return models[model_type](
        vocab_size=vocab_size,
        batch_size=batch_size,
        block_size=block_size,
        **kwargs
    )