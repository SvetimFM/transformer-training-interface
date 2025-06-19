"""
Enhanced trainer with visualization mode delays for observing each training step
"""

import torch
import time
from .trainer import Trainer, TrainingMetrics

class VisualizationTrainer(Trainer):
    """Trainer with added delays in visualization mode to observe each step"""
    
    def _get_viz_delay(self):
        """Calculate visualization delay based on speed ratio"""
        if self.config.training.visualization_mode and self.config.training.visualization_speed_ratio > 0:
            # Base delay of 0.1 seconds, scaled by speed ratio
            delay = (1 - self.config.training.visualization_speed_ratio) / self.config.training.visualization_speed_ratio * 0.05
            return min(delay, 2.0)  # Cap at 2 seconds per sub-step
        return 0
    
    def _viz_sleep(self, phase_name=None):
        """Sleep if in visualization mode, optionally announce phase"""
        delay = self._get_viz_delay()
        if delay > 0:
            if phase_name:
                self._run_callbacks("on_viz_phase", {"phase": phase_name, "delay": delay})
            time.sleep(delay)
    
    def _training_loop(self):
        """Enhanced training loop with visualization delays"""
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
                
                # Step 1: Get batch
                self._viz_sleep("Getting batch")
                xb, yb = self._get_batch(self.train_data)
                
                # Step 2: Forward pass (embedding + attention + feedforward)
                self._viz_sleep("Forward pass - Embeddings")
                logits, loss = self.model(xb, yb)
                
                # Step 3: Zero gradients
                self._viz_sleep("Zeroing gradients")
                self.optimizer.zero_grad()
                
                # Step 4: Backward pass
                self._viz_sleep("Backward pass")
                loss.backward()
                
                # Step 5: Optimizer step (update weights)
                self._viz_sleep("Optimizer step")
                self.optimizer.step()
                
                # Step 6: Update learning rate
                self._viz_sleep("LR scheduler step")
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
                    self._viz_sleep("Validation")
                    val_loss = self.evaluate()
                    self.last_val_loss = val_loss
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                    
                    # Log when we're doing tail validation
                    if is_tail_phase and global_step % self.config.training.eval_interval != 0:
                        print(f"Tail phase validation at step {global_step}")
                else:
                    val_loss = self.last_val_loss
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item() if val_loss > 0 else 0.0
                
                self.current_metrics = TrainingMetrics(
                    step=global_step,
                    epoch=epoch,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    perplexity=torch.exp(loss).item(),
                    val_perplexity=val_perplexity
                )
                
                # Only append to history on eval intervals (using dynamic interval)
                if global_step % current_eval_interval == 0:
                    self.metrics_history.append(self.current_metrics)
                    self._run_callbacks("on_eval", self.current_metrics)
                
                if global_step % self.config.training.save_interval == 0 and global_step > 0:
                    self._viz_sleep("Saving checkpoint")
                    self.save_checkpoint(global_step)
                    self._run_callbacks("on_checkpoint", global_step)
                
                self._run_callbacks("on_step", self.current_metrics)
                
                # Main visualization delay (in addition to step-specific delays)
                if self.config.training.visualization_mode:
                    # This is the main delay from the original implementation
                    if self.config.training.visualization_speed_ratio > 0:
                        delay = (1 - self.config.training.visualization_speed_ratio) / self.config.training.visualization_speed_ratio * 0.1
                        time.sleep(min(delay, 5.0))
                
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