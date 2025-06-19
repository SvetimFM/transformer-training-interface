"""
Enhanced TransformerBlock with visualization delays
"""

import torch
import torch.nn as nn
import time
from .transformer_block import TransformerBlock

class VisualizationTransformerBlock(TransformerBlock):
    """TransformerBlock with visualization delays"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._viz_config = None
        self._viz_callback = None
    
    def set_viz_config(self, config, callback=None):
        """Set visualization configuration and optional callback"""
        self._viz_config = config
        self._viz_callback = callback
    
    def _get_viz_delay(self):
        """Calculate visualization delay based on speed ratio"""
        if self._viz_config and self._viz_config.visualization_mode and self._viz_config.visualization_speed_ratio > 0:
            # Very small delay for block internal steps
            delay = (1 - self._viz_config.visualization_speed_ratio) / self._viz_config.visualization_speed_ratio * 0.01
            return min(delay, 0.5)
        return 0
    
    def _viz_sleep(self, phase_name=None):
        """Sleep if in visualization mode"""
        delay = self._get_viz_delay()
        if delay > 0:
            if phase_name and self._viz_callback:
                self._viz_callback({"phase": phase_name, "delay": delay})
            time.sleep(delay)
    
    def forward(self, x):
        if self.use_layer_norm and self.norm_position == "pre":
            # Pre-norm attention
            self._viz_sleep("Pre-norm (Attention)")
            normed = self.ln1(x)
            
            self._viz_sleep("Multi-Head Attention")
            attn_out = self.attention(normed)
            
            self._viz_sleep("Residual + Dropout (Attention)")
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            # Pre-norm feedforward
            self._viz_sleep("Pre-norm (FFN)")
            normed = self.ln2(x)
            
            self._viz_sleep("Feed Forward")
            ff_out = self.feed_forward(normed)
            
            self._viz_sleep("Residual + Dropout (FFN)")
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        elif self.use_layer_norm and self.norm_position == "post":
            # Post-norm attention
            self._viz_sleep("Multi-Head Attention")
            attn_out = self.attention(x)
            
            self._viz_sleep("Residual + Dropout (Attention)")
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            self._viz_sleep("Post-norm (Attention)")
            x = self.ln1(x)
            
            # Post-norm feedforward
            self._viz_sleep("Feed Forward")
            ff_out = self.feed_forward(x)
            
            self._viz_sleep("Residual + Dropout (FFN)")
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
            
            self._viz_sleep("Post-norm (FFN)")
            x = self.ln2(x)
        
        else:
            # No normalization
            self._viz_sleep("Multi-Head Attention")
            attn_out = self.attention(x)
            
            self._viz_sleep("Residual + Dropout (Attention)")
            x = x + self.dropout(attn_out) if self.use_residual else self.dropout(attn_out)
            
            self._viz_sleep("Feed Forward")
            ff_out = self.feed_forward(x)
            
            self._viz_sleep("Residual + Dropout (FFN)")
            x = x + self.dropout(ff_out) if self.use_residual else self.dropout(ff_out)
        
        return x