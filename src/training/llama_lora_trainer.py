"""
Optimized trainer for LLaMA models with LoRA fine-tuning.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from typing import Optional, Dict, List, Union, Callable
import os
from pathlib import Path
import json
import time
from dataclasses import dataclass
import wandb
from datasets import Dataset
import numpy as np
from tqdm import tqdm


@dataclass
class LLaMALoRAConfig:
    """Configuration for LLaMA LoRA training."""
    output_dir: str = "./llama_lora_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    group_by_length: bool = True
    dataloader_num_workers: int = 4
    report_to: List[str] = None
    remove_unused_columns: bool = False
    label_names: List[str] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    push_to_hub: bool = False
    max_seq_length: int = 2048
    packing: bool = False


class LLaMALoRATrainer:
    """
    Specialized trainer for LLaMA models with LoRA using HuggingFace Trainer.
    """
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[LLaMALoRAConfig] = None,
        data_collator: Optional[DataCollatorForLanguageModeling] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List] = None,
        optimizers: tuple = (None, None)
    ):
        """
        Initialize LLaMA LoRA trainer.
        
        Args:
            model: PEFT model with LoRA
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Training configuration
            data_collator: Data collator for batching
            compute_metrics: Metrics computation function
            callbacks: Training callbacks
            optimizers: Custom optimizer and scheduler
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or LLaMALoRAConfig()
        
        # Set up data collator
        if data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        else:
            self.data_collator = data_collator
        
        # Create training arguments
        self.training_args = self._create_training_args()
        
        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers
        )
        
        # Print training info
        self._print_training_info()
        
    def _create_training_args(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from config."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            group_by_length=self.config.group_by_length,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to or [],
            remove_unused_columns=self.config.remove_unused_columns,
            label_names=self.config.label_names,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            push_to_hub=self.config.push_to_hub,
        )
        
    def _print_training_info(self):
        """Print training configuration and model info."""
        print("\n" + "="*50)
        print("LLaMA LoRA Training Configuration")
        print("="*50)
        
        # Model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Dataset info
        print(f"\nTraining samples: {len(self.train_dataset):,}")
        if self.eval_dataset:
            print(f"Evaluation samples: {len(self.eval_dataset):,}")
        
        # Training config
        print(f"\nBatch size: {self.config.per_device_train_batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_train_epochs}")
        print(f"Max sequence length: {self.config.max_seq_length}")
        
        # Memory settings
        print(f"\nGradient checkpointing: {self.config.gradient_checkpointing}")
        print(f"Mixed precision: {'fp16' if self.config.fp16 else 'bf16' if self.config.bf16 else 'fp32'}")
        
        print("="*50 + "\n")
        
    def train(self):
        """Start training."""
        return self.trainer.train()
        
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate the model."""
        return self.trainer.evaluate(eval_dataset=eval_dataset)
        
    def predict(self, test_dataset: Dataset):
        """Make predictions on test dataset."""
        return self.trainer.predict(test_dataset)
        
    def save_model(self, output_dir: Optional[str] = None):
        """Save the model and tokenizer."""
        save_dir = output_dir or self.config.output_dir
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
    def push_to_hub(self, repo_name: str, **kwargs):
        """Push model to HuggingFace Hub."""
        self.trainer.push_to_hub(repo_name, **kwargs)


def prepare_dataset_for_llama(
    texts: List[str],
    tokenizer,
    max_length: int = 2048,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dataset:
    """
    Prepare dataset for LLaMA training.
    
    Args:
        texts: List of text strings
        tokenizer: LLaMA tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        return_tensors: Return type
        
    Returns:
        HuggingFace Dataset
    """
    def tokenize_function(examples):
        # Tokenize the texts
        outputs = tokenizer(
            examples["text"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=None  # Return lists for Dataset
        )
        
        # For causal LM, labels are the same as input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        
        return outputs
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def create_llama_lora_trainer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "./llama_lora_output",
    load_in_8bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_seq_length: int = 2048,
    validation_split: float = 0.1,
    seed: int = 42
) -> LLaMALoRATrainer:
    """
    Convenience function to create a complete LLaMA LoRA training setup.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to dataset file
        output_dir: Output directory
        load_in_8bit: Use 8-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Learning rate
        num_epochs: Number of epochs
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation
        max_seq_length: Maximum sequence length
        validation_split: Validation set fraction
        seed: Random seed
        
    Returns:
        LLaMALoRATrainer instance
    """
    from models.llama_lora import create_llama_lora
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model and tokenizer
    print("Loading model and applying LoRA...")
    lora_model, tokenizer = create_llama_lora(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        if dataset_path.endswith('.json'):
            texts = json.load(f)
            if isinstance(texts, dict):
                texts = texts.get('texts', [])
        else:
            texts = f.read().split('\n\n')  # Split by double newline
    
    # Split into train/val
    np.random.shuffle(texts)
    split_idx = int(len(texts) * (1 - validation_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Prepare datasets
    train_dataset = prepare_dataset_for_llama(
        train_texts, tokenizer, max_seq_length
    )
    eval_dataset = prepare_dataset_for_llama(
        val_texts, tokenizer, max_seq_length
    )
    
    # Create config
    config = LLaMALoRAConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        fp16=True,
        gradient_checkpointing=True
    )
    
    # Create trainer
    trainer = LLaMALoRATrainer(
        model=lora_model.get_model(),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )
    
    return trainer