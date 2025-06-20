"""
Specialized trainer for LoRA fine-tuning on custom datasets.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import os
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import json

from training.trainer import Trainer, TrainingMetrics
from models.lora_model import LoRABigramLM


@dataclass
class LoRATrainingMetrics(TrainingMetrics):
    """Extended metrics for LoRA training."""
    base_model_loss: Optional[float] = None
    improvement: Optional[float] = None  # Relative improvement over base model
    dataset_name: str = "custom"


class LoRATrainer(Trainer):
    """
    Trainer specifically for LoRA fine-tuning.
    Tracks validation performance on custom dataset.
    """
    def __init__(
        self,
        lora_model: LoRABigramLM,
        base_model: Optional[nn.Module],
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config,
        dataset_name: str = "custom",
        device: str = "cuda"
    ):
        # Initialize parent trainer with LoRA model
        super().__init__(lora_model, train_data, val_data, config, device)
        
        self.lora_model = lora_model
        self.base_model = base_model
        self.dataset_name = dataset_name
        
        # LoRA-specific configuration
        self.lora_lr_multiplier = config.training.get('lora_lr_multiplier', 1.0)
        
        # Recreate optimizer with LoRA-specific learning rate
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            lora_params,
            lr=config.training.learning_rate * self.lora_lr_multiplier
        )
        
        # Training info
        print(f"LoRA Training initialized for dataset: {dataset_name}")
        print(f"Trainable parameters: {self.lora_model.get_num_trainable_params():,}")
        print(f"Total parameters: {self.lora_model.get_num_total_params():,}")
        print(f"Parameter efficiency: {self.lora_model.get_num_trainable_params() / self.lora_model.get_num_total_params() * 100:.2f}%")
        
    @torch.no_grad()
    def evaluate_with_comparison(self, max_batches=10) -> Tuple[float, float, float]:
        """
        Evaluate both LoRA and base model for comparison.
        Returns: (lora_loss, base_loss, improvement_percentage)
        """
        # Evaluate LoRA model
        lora_loss = self.evaluate(max_batches)
        
        if self.base_model is None:
            return lora_loss, lora_loss, 0.0
            
        # Evaluate base model
        self.base_model.eval()
        base_losses = []
        
        for _ in range(min(max_batches, len(self.val_data) // self.config.training.batch_size)):
            xb, yb = self._get_batch(self.val_data)
            
            if self.config.training.use_amp:
                with autocast('cuda'):
                    _, loss = self.base_model(xb, yb)
            else:
                _, loss = self.base_model(xb, yb)
                
            base_losses.append(loss.item())
        
        self.base_model.train()
        base_loss = sum(base_losses) / len(base_losses) if base_losses else float('inf')
        
        # Calculate improvement
        improvement = ((base_loss - lora_loss) / base_loss * 100) if base_loss > 0 else 0.0
        
        return lora_loss, base_loss, improvement
    
    def _training_loop(self):
        """Extended training loop with comparison metrics."""
        self.model.train()
        
        # Calculate total steps
        if self.config.training.train_steps is None:
            total_steps = self.steps_per_epoch * self.config.training.epochs
        else:
            total_steps = self.config.training.train_steps
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            if self.should_stop:
                break
                
            epoch_start_time = time.time()
            
            for step_in_epoch in range(self.steps_per_epoch):
                if self.should_stop or global_step >= total_steps:
                    break
                
                # Track timing for performance metrics
                step_start_time = time.time()
                
                # Get batch
                xb, yb = self._get_batch(self.train_data)
                
                # Forward pass with AMP
                if self.config.training.use_amp:
                    with autocast('cuda'):
                        logits, loss = self.model(xb, yb)
                        loss = loss / self.accumulation_steps
                else:
                    logits, loss = self.model(xb, yb)
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                if self.config.training.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights only after accumulation steps
                if (global_step + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    gradient_norm = None
                    if self.config.training.gradient_clip_norm is not None:
                        if self.config.training.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.training.gradient_clip_norm
                        ).item()
                    
                    # Optimizer step
                    if self.config.training.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Calculate performance metrics
                step_time = time.time() - step_start_time
                batch_tokens = xb.numel()
                self.tokens_processed += batch_tokens
                tokens_per_second = batch_tokens / step_time if step_time > 0 else 0
                
                # Get GPU memory usage
                gpu_memory_mb = 0.0
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                
                # Step the learning rate scheduler
                current_lr = self.scheduler.step()
                
                # Evaluation with comparison
                if global_step % self.config.training.eval_interval == 0:
                    lora_loss, base_loss, improvement = self.evaluate_with_comparison()
                    self.last_val_loss = lora_loss
                    
                    # Save best model
                    if lora_loss < best_val_loss:
                        best_val_loss = lora_loss
                        self.save_lora_checkpoint(global_step, is_best=True)
                    
                    print(f"Step {global_step}: LoRA loss={lora_loss:.4f}, Base loss={base_loss:.4f}, Improvement={improvement:.2f}%")
                else:
                    lora_loss = self.last_val_loss
                    base_loss = None
                    improvement = None
                
                # Create metrics
                self.current_metrics = LoRATrainingMetrics(
                    step=global_step,
                    epoch=epoch,
                    train_loss=loss.item() * self.accumulation_steps,
                    val_loss=lora_loss,
                    learning_rate=current_lr,
                    perplexity=torch.exp(loss * self.accumulation_steps).item(),
                    val_perplexity=torch.exp(torch.tensor(lora_loss)).item() if lora_loss else None,
                    tokens_per_second=tokens_per_second,
                    gpu_memory_mb=gpu_memory_mb,
                    gradient_norm=gradient_norm if (global_step + 1) % self.accumulation_steps == 0 else None,
                    base_model_loss=base_loss,
                    improvement=improvement,
                    dataset_name=self.dataset_name
                )
                
                # Callbacks and history
                if global_step % self.config.training.eval_interval == 0:
                    self.metrics_history.append(self.current_metrics)
                    self._run_callbacks("on_eval", self.current_metrics)
                
                if global_step % self.config.training.save_interval == 0 and global_step > 0:
                    self.save_lora_checkpoint(global_step)
                    self._run_callbacks("on_checkpoint", global_step)
                
                self._run_callbacks("on_step", self.current_metrics)
                
                global_step += 1
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} completed in {epoch_time:.2f}s")
            self._run_callbacks("on_epoch_end", epoch, self.current_metrics)
        
        # Training completed
        self.is_training = False
        
        # Final evaluation
        final_lora_loss, final_base_loss, final_improvement = self.evaluate_with_comparison()
        
        print(f"\nTraining completed!")
        print(f"Final LoRA loss: {final_lora_loss:.4f}")
        print(f"Final Base loss: {final_base_loss:.4f}")
        print(f"Final Improvement: {final_improvement:.2f}%")
        
        # Save final checkpoint
        self.save_lora_checkpoint(global_step, final=True)
        
    def save_lora_checkpoint(self, step: int, final: bool = False, is_best: bool = False):
        """Save LoRA-specific checkpoint."""
        # Create checkpoint directory
        lora_dir = os.path.join(self.config.training.checkpoint_dir, f"lora_{self.dataset_name}")
        os.makedirs(lora_dir, exist_ok=True)
        
        # Determine filename
        if final:
            filename = "lora_final.pt"
        elif is_best:
            filename = "lora_best.pt"
        else:
            filename = f"lora_step_{step}.pt"
            
        path = os.path.join(lora_dir, filename)
        
        # Save LoRA weights only
        self.lora_model.save_lora_checkpoint(path)
        
        # Save training info
        info = {
            'step': step,
            'dataset_name': self.dataset_name,
            'lora_config': self.lora_model.lora_config,
            'best_val_loss': self.last_val_loss,
            'metrics_history': [m.to_dict() for m in self.metrics_history[-10:]]  # Last 10 metrics
        }
        
        info_path = os.path.join(lora_dir, "training_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"LoRA checkpoint saved to {path}")