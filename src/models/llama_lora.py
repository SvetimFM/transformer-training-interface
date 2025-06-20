"""
LLaMA-specific LoRA implementation using PEFT library.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union
from transformers import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer
import json
from pathlib import Path


class LLaMALoRA:
    """
    Wrapper for LLaMA models with LoRA adapters using PEFT.
    """
    def __init__(
        self,
        base_model: LlamaForCausalLM,
        lora_config: Optional[Dict] = None,
        adapter_name: str = "default"
    ):
        """
        Initialize LLaMA with LoRA.
        
        Args:
            base_model: Pre-loaded LLaMA model
            lora_config: LoRA configuration dictionary
            adapter_name: Name for the LoRA adapter
        """
        self.base_model = base_model
        self.adapter_name = adapter_name
        
        # Default LoRA config
        if lora_config is None:
            lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': TaskType.CAUSAL_LM
            }
        
        # Create PEFT config
        self.peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', "none"),
            task_type=lora_config.get('task_type', TaskType.CAUSAL_LM),
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, self.peft_config)
        self.model.print_trainable_parameters()
        
    def get_model(self):
        """Get the PEFT model."""
        return self.model
        
    def save_pretrained(self, save_path: str):
        """Save LoRA adapter weights."""
        self.model.save_pretrained(save_path)
        
        # Save config
        config_path = Path(save_path) / "llama_lora_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'adapter_name': self.adapter_name,
                'peft_config': self.peft_config.to_dict()
            }, f, indent=2)
            
    def load_adapter(self, adapter_path: str, adapter_name: Optional[str] = None):
        """Load a saved LoRA adapter."""
        if adapter_name is None:
            adapter_name = self.adapter_name
            
        self.model.load_adapter(adapter_path, adapter_name)
        self.model.set_adapter(adapter_name)
        
    def merge_and_unload(self):
        """Merge LoRA weights into base model and unload PEFT."""
        return self.model.merge_and_unload()
        
    def enable_adapters(self):
        """Enable LoRA adapters."""
        self.model.enable_adapters()
        
    def disable_adapters(self):
        """Disable LoRA adapters (use base model only)."""
        self.model.disable_adapters()
        
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        adapter_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: Union[str, Dict] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        Load a model with pre-trained LoRA adapter.
        
        Args:
            model_name_or_path: Base model name or path
            adapter_path: Path to LoRA adapter
            load_in_8bit: Load in 8-bit precision
            load_in_4bit: Load in 4-bit precision
            device_map: Device mapping
            torch_dtype: Data type for model
            **kwargs: Additional arguments for model loading
            
        Returns:
            LLaMALoRA instance
        """
        # Load with PEFT
        model = PeftModel.from_pretrained(
            model_name_or_path,
            adapter_path,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        # Create wrapper
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.base_model = model.base_model
        wrapper.adapter_name = "default"
        
        return wrapper


def create_llama_lora(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = True,
    max_memory: Optional[Dict] = None
) -> tuple:
    """
    Convenience function to create a LLaMA model with LoRA in one step.
    
    Args:
        model_name: HuggingFace model name
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Modules to apply LoRA to
        use_gradient_checkpointing: Enable gradient checkpointing
        max_memory: Maximum memory per device
        
    Returns:
        (lora_model, tokenizer) tuple
    """
    from utils.llama_loader import load_llama_model, prepare_model_for_lora
    
    # Load base model and tokenizer
    base_model, tokenizer = load_llama_model(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        max_memory=max_memory
    )
    
    # Prepare for LoRA
    base_model = prepare_model_for_lora(
        base_model,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    # Default target modules for LLaMA
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Create LoRA config
    lora_config = {
        'r': lora_r,
        'lora_alpha': lora_alpha,
        'target_modules': target_modules,
        'lora_dropout': lora_dropout,
        'bias': "none",
        'task_type': TaskType.CAUSAL_LM
    }
    
    # Create LoRA model
    lora_model = LLaMALoRA(base_model, lora_config)
    
    return lora_model, tokenizer


def estimate_llama_lora_params(
    model_size: str = "7b",
    rank: int = 16,
    target_modules: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Estimate number of LoRA parameters for different LLaMA sizes.
    
    Args:
        model_size: Model size (7b, 13b, 30b, 65b)
        rank: LoRA rank
        target_modules: Target modules (default: attention modules)
        
    Returns:
        Dictionary with parameter counts
    """
    # Hidden dimensions for different model sizes
    hidden_dims = {
        "7b": 4096,
        "13b": 5120,
        "30b": 6656,
        "65b": 8192,
        "70b": 8192
    }
    
    # Number of layers
    n_layers = {
        "7b": 32,
        "13b": 40,
        "30b": 60,
        "65b": 80,
        "70b": 80
    }
    
    if model_size not in hidden_dims:
        raise ValueError(f"Unknown model size: {model_size}")
    
    hidden_dim = hidden_dims[model_size]
    layers = n_layers[model_size]
    
    # Default target modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Calculate LoRA parameters
    # Each LoRA adapter has A (rank x in_features) and B (out_features x rank)
    params_per_module = 2 * rank * hidden_dim
    total_lora_params = params_per_module * len(target_modules) * layers
    
    # Base model parameters (approximate)
    base_params = {
        "7b": 6.7e9,
        "13b": 13e9,
        "30b": 30e9,
        "65b": 65e9,
        "70b": 70e9
    }
    
    return {
        "base_parameters": int(base_params[model_size]),
        "lora_parameters": total_lora_params,
        "percentage": (total_lora_params / base_params[model_size]) * 100,
        "parameters_per_module": params_per_module,
        "total_modules": len(target_modules) * layers
    }