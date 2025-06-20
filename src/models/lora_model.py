"""
LoRA-adapted version of the BigramLM model.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from models.bigram import BigramLM
from models.lora import apply_lora_to_model, get_lora_params, save_lora_weights, load_lora_weights


class LoRABigramLM(nn.Module):
    """
    BigramLM with LoRA adapters for parameter-efficient fine-tuning.
    """
    def __init__(
        self,
        base_model: BigramLM,
        lora_config: Dict,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        
        # Default target modules for transformer
        if target_modules is None:
            target_modules = ['key', 'query', 'value', 'proj', 'decoder_head']
        
        # Apply LoRA to target modules
        self.base_model = apply_lora_to_model(
            self.base_model,
            target_modules=target_modules,
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16),
            dropout=lora_config.get('dropout', 0.0)
        )
        
        # Freeze all non-LoRA parameters
        self._freeze_non_lora_params()
        
    def _freeze_non_lora_params(self):
        """Freeze all parameters except LoRA parameters."""
        # First, freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Then unfreeze LoRA parameters
        for param in get_lora_params(self.base_model):
            param.requires_grad = True
            
    def forward(self, idx, targets=None):
        """Forward pass through the LoRA-adapted model."""
        return self.base_model(idx, targets)
    
    def generate(self, idx, max_new_tokens):
        """Generate text using the LoRA-adapted model."""
        return self.base_model.generate(idx, max_new_tokens)
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def save_lora_checkpoint(self, path: str):
        """Save only LoRA weights."""
        save_lora_weights(self.base_model, path)
        
    def load_lora_checkpoint(self, path: str):
        """Load LoRA weights."""
        load_lora_weights(self.base_model, path)
        
    def merge_and_save(self, path: str):
        """Merge LoRA weights into base model and save full model."""
        # Merge LoRA weights
        for module in self.base_model.modules():
            if hasattr(module, 'merge_weights'):
                module.merge_weights()
        
        # Save merged model
        torch.save(self.base_model.state_dict(), path)
        
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        lora_checkpoint_path: Optional[str] = None,
        config=None,
        lora_config: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        """
        Load a model with optional LoRA weights.
        
        Args:
            base_model_path: Path to base model checkpoint
            lora_checkpoint_path: Optional path to LoRA weights
            config: Model configuration
            lora_config: LoRA configuration
            device: Device to load model on
        """
        # Load base model
        checkpoint = torch.load(base_model_path, map_location=device)
        
        # Create base model
        base_model = BigramLM(
            vocab_size=config.model.vocab_size,
            batch_size=config.training.batch_size,
            block_size=config.model.block_size,
            config=config
        )
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.to(device)
        
        # Create LoRA model
        if lora_config is None:
            lora_config = {'rank': 8, 'alpha': 16, 'dropout': 0.0}
            
        lora_model = cls(base_model, lora_config)
        
        # Load LoRA weights if provided
        if lora_checkpoint_path:
            lora_model.load_lora_checkpoint(lora_checkpoint_path)
            
        return lora_model