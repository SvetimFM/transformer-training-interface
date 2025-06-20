"""
Utilities for loading and managing LLaMA models with memory optimization.
"""

import torch
import gc
from pathlib import Path
from typing import Optional, Dict, Union
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    LlamaConfig,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_model_memory_footprint(model):
    """Calculate model memory footprint in MB."""
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_mb = mem / 1024 / 1024
    return mem_mb


def load_llama_model(
    model_name_or_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: Union[str, Dict] = "auto",
    max_memory: Optional[Dict] = None,
    torch_dtype: torch.dtype = torch.float16,
    use_cache: bool = True,
    trust_remote_code: bool = False
) -> tuple:
    """
    Load a LLaMA model with various optimization options.
    
    Args:
        model_name_or_path: HuggingFace model ID or local path
        load_in_8bit: Load model in 8-bit precision (requires bitsandbytes)
        load_in_4bit: Load model in 4-bit precision (requires bitsandbytes)
        device_map: Device mapping strategy
        max_memory: Maximum memory per device
        torch_dtype: Data type for model weights
        use_cache: Whether to use KV cache
        trust_remote_code: Whether to trust remote code
        
    Returns:
        model, tokenizer tuple
    """
    print(f"Loading LLaMA model: {model_name_or_path}")
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print(f"Using {'8-bit' if load_in_8bit else '4-bit'} quantization")
    
    # Set default max memory if not provided
    if max_memory is None and torch.cuda.is_available():
        # Leave some memory for activations and gradients
        free_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = {0: f"{int(free_memory * 0.85 / 1024**3)}GB", "cpu": "50GB"}
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        use_cache=use_cache
    )
    
    # Print model info
    print(f"Model loaded successfully!")
    print(f"Model size: {get_model_memory_footprint(model):.2f} MB")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    
    return model, tokenizer


def prepare_model_for_lora(
    model,
    use_gradient_checkpointing: bool = True,
    enable_input_require_grads: bool = True
):
    """
    Prepare a model for LoRA training with memory optimizations.
    
    Args:
        model: The model to prepare
        use_gradient_checkpointing: Enable gradient checkpointing
        enable_input_require_grads: Enable gradients for inputs
    """
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Prepare model for k-bit training if quantized
    if hasattr(model, 'quantization_method'):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        print("Model prepared for k-bit training")
    
    # Enable input gradients
    if enable_input_require_grads:
        model.enable_input_require_grads()
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    return model


def estimate_lora_memory_usage(
    model_size_gb: float,
    rank: int = 16,
    target_modules_fraction: float = 0.1,
    batch_size: int = 1,
    sequence_length: int = 2048,
    gradient_accumulation_steps: int = 4,
    quantized: bool = False
) -> Dict[str, float]:
    """
    Estimate memory usage for LoRA training.
    
    Args:
        model_size_gb: Base model size in GB
        rank: LoRA rank
        target_modules_fraction: Fraction of model parameters targeted
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        quantized: Whether model is quantized
        
    Returns:
        Dictionary with memory estimates
    """
    # Base model memory
    if quantized:
        model_memory = model_size_gb * 0.5  # 4-bit is ~50% of fp16
    else:
        model_memory = model_size_gb
    
    # LoRA parameters memory (rough estimate)
    lora_params = model_size_gb * target_modules_fraction * (2 * rank / 4096)  # Assuming 4096 hidden dim
    
    # Gradients memory
    gradients_memory = lora_params * 2  # Gradients are same size as parameters
    
    # Optimizer states (AdamW has 2 states per parameter)
    optimizer_memory = lora_params * 2 * 2
    
    # Activations memory (rough estimate)
    activations_per_sample = sequence_length * 4096 * 4 / 1024**3  # Assuming 4096 hidden dim
    activations_memory = activations_per_sample * batch_size
    
    # Total memory
    total_memory = (
        model_memory + 
        lora_params + 
        gradients_memory / gradient_accumulation_steps + 
        optimizer_memory + 
        activations_memory
    )
    
    return {
        "model_memory_gb": model_memory,
        "lora_params_gb": lora_params,
        "gradients_memory_gb": gradients_memory,
        "optimizer_memory_gb": optimizer_memory,
        "activations_memory_gb": activations_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2  # 20% overhead
    }


def get_llama_target_modules(model_type: str = "llama") -> list:
    """
    Get the target modules for LoRA based on model type.
    
    Args:
        model_type: Type of model (llama, mistral, etc.)
        
    Returns:
        List of module names to target
    """
    if model_type.lower() in ["llama", "llama2", "llama3"]:
        # LLaMA attention and MLP projections
        return [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    elif model_type.lower() == "mistral":
        return [
            "q_proj",
            "k_proj",
            "v_proj", 
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    else:
        # Default to attention modules only
        return ["q_proj", "v_proj"]


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()