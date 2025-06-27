from pydantic import BaseModel
from typing import Optional, List

class ModelConfig(BaseModel):
    vocab_size: int = 65
    n_embed: int = 256
    n_heads: int = 8
    n_layers: int = 1
    block_size: int = 256
    dropout: float = 0.2
    
    use_layer_norm: bool = False
    use_residual: bool = False
    norm_position: str = "pre"  # "pre" or "post"
    
    # Output layer configuration
    output_activation: str = "gelu"  # "relu", "gelu", "silu", "tanh"
    n_output_layers: int = 0  # Number of hidden layers after transformer blocks
    output_hidden_dim: int = 512  # Dimension of output hidden layers
    
    class Config:
        validate_assignment = True

class TrainingConfig(BaseModel):
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 10
    train_steps: Optional[int] = None  # If None, will be calculated from epochs
    eval_interval: int = 500
    train_split: float = 0.85
    device: str = "cuda"
    seed: int = 1337
    
    # Learning rate scheduler parameters
    scheduler_type: str = "warmup_cosine"  # warmup_cosine, warmup_linear, warmup_constant, onecycle
    warmup_steps: Optional[int] = None  # If None, uses warmup_ratio
    warmup_ratio: float = 0.05  # 5% warmup by default
    min_lr_ratio: float = 0.1  # Final LR = 10% of max_lr
    tail_ratio: float = 0.2  # Last 20% of training for fine polish
    tail_eval_multiplier: int = 5  # Validate 5x more frequently in tail phase
    
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000
    
    # Visualization mode settings
    visualization_mode: bool = False  # Slow training for visualization
    visualization_speed_ratio: float = 0.01  # 1% of normal speed by default
    
    # Optimization settings
    compile_model: bool = True  # Use torch.compile() for faster training
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    use_amp: bool = True  # Use automatic mixed precision
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    gradient_clip_norm: Optional[float] = 1.0  # Max gradient norm for clipping
    
    class Config:
        validate_assignment = True

class GenerationConfig(BaseModel):
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    class Config:
        validate_assignment = True

class LoRAConfig(BaseModel):
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = ["key", "query", "value", "proj", "decoder_head"]
    lora_lr_multiplier: float = 1.0  # Learning rate multiplier for LoRA parameters
    
    class Config:
        validate_assignment = True

class AppConfig:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.generation = GenerationConfig()
        self.lora = LoRAConfig()
        
    def to_dict(self):
        return {
            "model": self.model.model_dump(),
            "training": self.training.model_dump(),
            "generation": self.generation.model_dump(),
            "lora": self.lora.model_dump()
        }
    
    def update_from_dict(self, config_dict):
        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            self.training = TrainingConfig(**config_dict["training"])
        if "generation" in config_dict:
            self.generation = GenerationConfig(**config_dict["generation"])
        if "lora" in config_dict:
            self.lora = LoRAConfig(**config_dict["lora"])

app_config = AppConfig()