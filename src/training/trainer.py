import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import threading
import time
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import json
from .lr_scheduler import create_scheduler

@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    perplexity: float = 0.0
    val_perplexity: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    tokens_per_second: float = 0.0
    gpu_memory_mb: float = 0.0
    gradient_norm: Optional[float] = None
    
    def to_dict(self):
        data = {
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "learning_rate": self.learning_rate,
            "perplexity": self.perplexity,
            "timestamp": self.timestamp,
            "tokens_per_second": self.tokens_per_second,
            "gpu_memory_mb": self.gpu_memory_mb
        }
        # Only include val_loss and val_perplexity if they exist
        if self.val_loss is not None:
            data["val_loss"] = self.val_loss
        if self.val_perplexity is not None:
            data["val_perplexity"] = self.val_perplexity
        if self.gradient_norm is not None:
            data["gradient_norm"] = self.gradient_norm
        return data

class Trainer:
    def __init__(self, model, train_data, val_data, config, device="cuda"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device
        
        # Compile model if requested
        if config.training.compile_model and hasattr(torch, 'compile'):
            print(f"Compiling model with mode: {config.training.compile_mode}")
            self.model = torch.compile(model, mode=config.training.compile_mode)
        
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
        
        # Initialize AMP scaler if enabled
        self.scaler = GradScaler() if config.training.use_amp else None
        self.accumulation_steps = config.training.gradient_accumulation_steps
        
        # Calculate steps per epoch first
        self.steps_per_epoch = len(train_data) // config.training.batch_size
        
        # Create learning rate scheduler
        self.scheduler = create_scheduler(
            config.training.scheduler_type,
            self.optimizer,
            config,
            steps_per_epoch=self.steps_per_epoch
        )
        
        self.metrics_history: List[TrainingMetrics] = []
        self.current_metrics = TrainingMetrics()
        self.last_val_loss = None
        
        self.is_training = False
        self.should_stop = False
        self.training_thread = None
        
        self.callbacks: Dict[str, List[Callable]] = {
            "on_step": [],
            "on_eval": [],
            "on_checkpoint": [],
            "on_epoch_end": [],
            "on_training_end": []
        }
        
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        # Performance tracking
        self.tokens_processed = 0
        self.last_time = time.time()
    
    def add_callback(self, event: str, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _run_callbacks(self, event: str, *args, **kwargs):
        for callback in self.callbacks.get(event, []):
            callback(*args, **kwargs)
    
    @torch.no_grad()
    def evaluate(self, max_batches=10):
        self.model.eval()
        losses = []
        
        for _ in range(min(max_batches, len(self.val_data) // self.config.training.batch_size)):
            xb, yb = self._get_batch(self.val_data)
            
            if self.config.training.use_amp:
                with autocast('cuda'):
                    _, loss = self.model(xb, yb)
            else:
                _, loss = self.model(xb, yb)
                
            losses.append(loss.item())
        
        self.model.train()
        return sum(losses) / len(losses) if losses else float('inf')
    
    def _get_batch(self, data):
        ix = torch.randint(len(data) - self.config.model.block_size, (self.config.training.batch_size,))
        x = torch.stack([data[i:i+self.config.model.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.model.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)
    
    def _get_viz_delay(self, multiplier=1.0):
        """Calculate visualization delay based on speed ratio"""
        if self.config.training.visualization_mode and self.config.training.visualization_speed_ratio > 0:
            # Base delay scaled by speed ratio and multiplier
            delay = (1 - self.config.training.visualization_speed_ratio) / self.config.training.visualization_speed_ratio * 0.05 * multiplier
            return min(delay, 2.0)  # Cap at 2 seconds
        return 0
    
    def _training_loop(self):
        self.model.train()
        
        # Calculate total steps
        if self.config.training.train_steps is None:
            total_steps = self.steps_per_epoch * self.config.training.epochs
        else:
            total_steps = self.config.training.train_steps
        
        global_step = 0
        
        for epoch in range(self.config.training.epochs):
            if self.should_stop:
                break
                
            epoch_start_time = time.time()
            
            for step_in_epoch in range(self.steps_per_epoch):
                if self.should_stop or global_step >= total_steps:
                    break
                
                # Track timing for performance metrics
                step_start_time = time.time()
                
                # Get batch with optional viz delay
                if self.config.training.visualization_mode:
                    time.sleep(self._get_viz_delay(0.5))  # Small delay for batch loading
                xb, yb = self._get_batch(self.train_data)
                
                # Forward pass with AMP
                if self.config.training.use_amp:
                    with autocast(device_type='cuda'):
                        logits, loss = self.model(xb, yb)
                        # Scale loss for gradient accumulation
                        loss = loss / self.accumulation_steps
                else:
                    logits, loss = self.model(xb, yb)
                    loss = loss / self.accumulation_steps
                
                # Backward pass with viz delay
                if self.config.training.visualization_mode:
                    time.sleep(self._get_viz_delay(1.0))  # Full delay for backward
                
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
                    
                    # Optimizer step with viz delay
                    if self.config.training.visualization_mode:
                        time.sleep(self._get_viz_delay(0.5))
                    
                    if self.config.training.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Zero gradients for next accumulation
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
                
                # Determine if we're in tail phase (last 20% of training)
                tail_start_step = int(total_steps * (1 - self.config.training.tail_ratio))
                is_tail_phase = global_step >= tail_start_step
                
                # Calculate dynamic eval interval
                if is_tail_phase:
                    current_eval_interval = max(1, self.config.training.eval_interval // self.config.training.tail_eval_multiplier)
                else:
                    current_eval_interval = self.config.training.eval_interval
                
                # Calculate validation loss and perplexity
                if global_step % current_eval_interval == 0:
                    val_loss = self.evaluate()
                    self.last_val_loss = val_loss
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                    
                    # Log when we're doing tail validation
                    if is_tail_phase and global_step % self.config.training.eval_interval != 0:
                        print(f"Tail phase validation at step {global_step}")
                else:
                    val_loss = self.last_val_loss
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item() if val_loss is not None else None
                
                self.current_metrics = TrainingMetrics(
                    step=global_step,
                    epoch=epoch,
                    train_loss=loss.item() * self.accumulation_steps,  # Unscale loss
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    perplexity=torch.exp(loss * self.accumulation_steps).item(),
                    val_perplexity=val_perplexity,
                    tokens_per_second=tokens_per_second,
                    gpu_memory_mb=gpu_memory_mb,
                    gradient_norm=gradient_norm if (global_step + 1) % self.accumulation_steps == 0 else None
                )
                
                # Only append to history on eval intervals (using dynamic interval)
                if global_step % current_eval_interval == 0:
                    self.metrics_history.append(self.current_metrics)
                    self._run_callbacks("on_eval", self.current_metrics)
                
                if global_step % self.config.training.save_interval == 0 and global_step > 0:
                    self.save_checkpoint(global_step)
                    self._run_callbacks("on_checkpoint", global_step)
                
                self._run_callbacks("on_step", self.current_metrics)
                
                # Add delay for visualization mode
                if self.config.training.visualization_mode:
                    # Calculate delay to achieve desired speed ratio
                    # If ratio is 0.01 (1%), we want to wait 99x the normal step time
                    if self.config.training.visualization_speed_ratio > 0:
                        delay = (1 - self.config.training.visualization_speed_ratio) / self.config.training.visualization_speed_ratio * 0.1
                        time.sleep(min(delay, 5.0))  # Cap at 5 seconds max delay
                
                global_step += 1
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} completed in {epoch_time:.2f}s")
            self._run_callbacks("on_epoch_end", epoch, self.current_metrics)
        
        # Training completed
        self.is_training = False
        
        # Final evaluation
        final_val_loss = self.evaluate()
        final_metrics = TrainingMetrics(
            step=global_step,
            epoch=self.config.training.epochs,
            train_loss=self.current_metrics.train_loss,
            val_loss=final_val_loss,
            learning_rate=self.config.training.learning_rate,
            perplexity=self.current_metrics.perplexity,
            val_perplexity=torch.exp(torch.tensor(final_val_loss)).item()
        )
        
        self.metrics_history.append(final_metrics)
        self._run_callbacks("on_training_end", final_metrics)
        
        # Save final checkpoint
        self.save_checkpoint(global_step, final=True)
    
    def start_training(self):
        if self.is_training:
            return False
        
        self.is_training = True
        self.should_stop = False
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
        return True
    
    def stop_training(self):
        if not self.is_training:
            return False
        
        self.should_stop = True
        if self.training_thread:
            self.training_thread.join()
        return True
    
    def save_checkpoint(self, step, final=False):
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics_history': [m.to_dict() for m in self.metrics_history]
        }
        
        filename = f"checkpoint_{'final' if final else f'step_{step}'}.pt"
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        # Also save metrics separately for easy access
        metrics_path = os.path.join(self.config.training.checkpoint_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'metrics_history' in checkpoint:
            self.metrics_history = [
                TrainingMetrics(**m) for m in checkpoint['metrics_history']
            ]
        
        return checkpoint.get('step', 0)
    
    def get_status(self):
        total_steps = self.config.training.train_steps
        if total_steps is None:
            total_steps = self.steps_per_epoch * self.config.training.epochs
            
        return {
            "is_training": self.is_training,
            "current_step": self.current_metrics.step,
            "current_epoch": self.current_metrics.epoch,
            "total_epochs": self.config.training.epochs,
            "current_metrics": self.current_metrics.to_dict(),
            "total_steps": total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "scheduler_info": self.scheduler.get_schedule_info() if hasattr(self.scheduler, 'get_schedule_info') else {}
        }
    
    def update_scheduler(self):
        """Update the learning rate scheduler with new configuration"""
        # Preserve current optimizer state
        current_step = self.scheduler.current_step if hasattr(self.scheduler, 'current_step') else 0
        
        # Create new scheduler with updated config
        self.scheduler = create_scheduler(
            self.config.training.scheduler_type,
            self.optimizer,
            self.config,
            steps_per_epoch=self.steps_per_epoch
        )
        
        # Restore step count if training was in progress
        if current_step > 0:
            self.scheduler.current_step = current_step
            # Update learning rate to match current position
            lr = self.scheduler.get_lr(current_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        print(f"Scheduler updated: {self.config.training.scheduler_type} with warmup={self.config.training.warmup_ratio}, min_lr={self.config.training.min_lr_ratio}")