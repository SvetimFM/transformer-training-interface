"""
Attention weight capture for visualization
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import threading
import numpy as np


class AttentionCapture:
    """Captures attention weights from model during forward passes"""
    
    def __init__(self):
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.head_info: Dict[str, Dict] = {}
        self.is_capturing = False
        self._lock = threading.Lock()
        self._hooks = []
        
    def start_capture(self):
        """Start capturing attention weights"""
        with self._lock:
            self.is_capturing = True
            self.attention_weights.clear()
            
    def stop_capture(self):
        """Stop capturing attention weights"""
        with self._lock:
            self.is_capturing = False
            
    def register_attention_head(self, module: nn.Module, head_id: str, head_idx: int, layer_idx: int):
        """Register hooks on an attention head to capture weights"""
        
        # Store head info
        with self._lock:
            # Check if this is a standard multi-head attention or individual head
            if hasattr(module, 'n_heads'):
                # Standard multi-head attention
                self.head_info[head_id] = {
                    'head_idx': head_idx,
                    'layer_idx': layer_idx,
                    'n_embed': module.n_embed,
                    'head_size': module.n_embed // module.n_heads,
                    'n_heads': module.n_heads
                }
            else:
                # Individual attention head
                self.head_info[head_id] = {
                    'head_idx': head_idx,
                    'layer_idx': layer_idx,
                    'n_embed': module.n_embed,
                    'head_size': module.head_size
                }
        
        # Monkey-patch the forward method to capture attention weights
        original_forward = module.forward
        
        def wrapped_forward(x, return_attention=False):
            if self.is_capturing:
                # Call with return_attention=True when capturing
                out, attention_weights = original_forward(x, return_attention=True)
                
                with self._lock:
                    # Store attention weights (B, T, T)
                    self.attention_weights[head_id] = attention_weights.detach()
                
                if return_attention:
                    return out, attention_weights
                return out
            else:
                # Normal forward when not capturing
                return original_forward(x, return_attention=return_attention)
        
        module.forward = wrapped_forward
        
    def get_attention_patterns(self) -> Dict[str, np.ndarray]:
        """Get captured attention patterns"""
        with self._lock:
            patterns = {}
            for head_id, weights in self.attention_weights.items():
                # Convert to numpy and average over batch
                avg_weights = weights.mean(dim=0).cpu().numpy()
                patterns[head_id] = avg_weights
            return patterns
            
    def get_head_info(self) -> Dict[str, Dict]:
        """Get information about registered heads"""
        with self._lock:
            return self.head_info.copy()
            
    def clear(self):
        """Clear all captured data and remove hooks"""
        with self._lock:
            self.attention_weights.clear()
            self.head_info.clear()
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()


# Global instance
attention_capture = AttentionCapture()