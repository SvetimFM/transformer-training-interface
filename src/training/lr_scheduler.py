import math
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseLRScheduler(ABC):
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer, max_lr: float, total_steps: int, 
                 warmup_steps: Optional[int] = None, warmup_ratio: float = 0.05,
                 min_lr_ratio: float = 0.1, tail_ratio: float = 0.1):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.min_lr = max_lr * min_lr_ratio
        
        # Calculate warmup steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = int(total_steps * warmup_ratio)
        
        # Calculate phase boundaries
        self.tail_start = int(total_steps * (1 - tail_ratio))
        self.main_steps = self.tail_start - self.warmup_steps
        
        self.current_step = 0
        self._last_lr = 0.0
        
    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Calculate learning rate for given step"""
        pass
    
    def step(self):
        """Update learning rate for current step"""
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        self._last_lr = lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self._last_lr
    
    def get_schedule_info(self) -> dict:
        """Get information about the schedule"""
        return {
            "type": self.__class__.__name__,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "tail_start": self.tail_start,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "current_lr": self._last_lr
        }
    
    def get_full_schedule(self, num_points: int = 1000) -> tuple:
        """Get the full schedule for visualization"""
        steps = np.linspace(0, self.total_steps, num_points)
        lrs = [self.get_lr(int(step)) for step in steps]
        return steps.tolist(), lrs


class WarmupCosineScheduler(BaseLRScheduler):
    """Cosine scheduler with linear warmup and tail decay"""
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        
        elif step < self.tail_start:
            # Cosine decay in main phase
            progress = (step - self.warmup_steps) / self.main_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Decay from max_lr to a reasonable value before tail
            tail_start_lr = self.max_lr * 0.3  # 30% of max before tail
            return tail_start_lr + (self.max_lr - tail_start_lr) * cosine_factor
        
        else:
            # Tail phase - steep decay
            tail_progress = (step - self.tail_start) / (self.total_steps - self.tail_start)
            tail_start_lr = self.max_lr * 0.3
            
            # Cosine decay in tail for smooth finish
            cosine_factor = 0.5 * (1 + math.cos(math.pi * tail_progress))
            return self.min_lr + (tail_start_lr - self.min_lr) * cosine_factor


class WarmupLinearScheduler(BaseLRScheduler):
    """Linear scheduler with warmup and tail decay"""
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        
        elif step < self.tail_start:
            # Linear decay in main phase
            progress = (step - self.warmup_steps) / self.main_steps
            tail_start_lr = self.max_lr * 0.3
            return self.max_lr - (self.max_lr - tail_start_lr) * progress
        
        else:
            # Tail phase - steep linear decay
            tail_progress = (step - self.tail_start) / (self.total_steps - self.tail_start)
            tail_start_lr = self.max_lr * 0.3
            return tail_start_lr - (tail_start_lr - self.min_lr) * tail_progress


class WarmupConstantScheduler(BaseLRScheduler):
    """Constant learning rate with warmup and tail decay"""
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        
        elif step < self.tail_start:
            # Constant in main phase
            return self.max_lr
        
        else:
            # Tail phase - cosine decay for smooth finish
            tail_progress = (step - self.tail_start) / (self.total_steps - self.tail_start)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * tail_progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor


class OneCycleLRScheduler(BaseLRScheduler):
    """One Cycle learning rate scheduler (Super-convergence)"""
    
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        # Set warmup ratio based on pct_start
        super().__init__(optimizer, max_lr, total_steps, warmup_ratio=pct_start)
        
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        self.anneal_strategy = anneal_strategy
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Annealing up
            pct = step / self.warmup_steps
            return self.initial_lr + (self.max_lr - self.initial_lr) * pct
        
        else:
            # Annealing down
            step_num = step - self.warmup_steps
            step_max = self.total_steps - self.warmup_steps
            pct = step_num / step_max
            
            if self.anneal_strategy == 'cos':
                lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * pct))
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * pct
            
            return lr


def create_scheduler(scheduler_type: str, optimizer, config, steps_per_epoch: int = None) -> BaseLRScheduler:
    """Factory function to create scheduler based on type"""
    
    # Calculate total steps if not provided
    if config.training.train_steps is None:
        if steps_per_epoch is None:
            # Rough estimate if not provided
            steps_per_epoch = 1000  # Default estimate
        total_steps = steps_per_epoch * config.training.epochs
    else:
        total_steps = config.training.train_steps
    
    # Get warmup steps
    warmup_steps = getattr(config.training, 'warmup_steps', None)
    warmup_ratio = getattr(config.training, 'warmup_ratio', 0.05)
    min_lr_ratio = getattr(config.training, 'min_lr_ratio', 0.1)
    tail_ratio = getattr(config.training, 'tail_ratio', 0.1)
    
    schedulers = {
        'warmup_cosine': WarmupCosineScheduler,
        'warmup_linear': WarmupLinearScheduler,
        'warmup_constant': WarmupConstantScheduler,
        'onecycle': OneCycleLRScheduler
    }
    
    scheduler_class = schedulers.get(scheduler_type, WarmupCosineScheduler)
    
    if scheduler_type == 'onecycle':
        return OneCycleLRScheduler(
            optimizer=optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps
        )
    else:
        return scheduler_class(
            optimizer=optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
            tail_ratio=tail_ratio
        )