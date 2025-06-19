import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Optional
import time
from collections import deque
import threading
from .component_registry import ComponentState, component_registry

class ActivationTracker:
    def __init__(self, max_history: int = 100, update_interval: float = 0.1):
        self.hooks: List[Any] = []
        self.activation_history: deque = deque(maxlen=max_history)
        self.current_activations: Dict[str, ComponentState] = {}
        self.update_interval = update_interval
        self.last_update_time = 0
        self.is_tracking = False
        self._lock = threading.Lock()
        
    def start_tracking(self):
        """Start tracking activations"""
        self.is_tracking = True
        component_registry.reset_states()
        
    def stop_tracking(self):
        """Stop tracking activations"""
        self.is_tracking = False
        self.remove_all_hooks()
        
    def add_forward_hook(self, module: nn.Module, component_id: str):
        """Add forward hook to track when module is active"""
        def hook(module, input, output):
            if self.is_tracking:
                with self._lock:
                    self.current_activations[component_id] = ComponentState.FORWARD
                    component_registry.update_state(component_id, ComponentState.FORWARD)
                    
                    # Schedule state reset after a short delay
                    self._schedule_state_reset(component_id)
                    
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        
    def add_backward_hook(self, module: nn.Module, component_id: str):
        """Add backward hook to track gradient flow"""
        def hook(module, grad_input, grad_output):
            if self.is_tracking:
                with self._lock:
                    self.current_activations[component_id] = ComponentState.BACKWARD
                    component_registry.update_state(component_id, ComponentState.BACKWARD)
                    
                    # Schedule state reset after a short delay
                    self._schedule_state_reset(component_id)
                    
        if hasattr(module, 'register_full_backward_hook'):
            handle = module.register_full_backward_hook(hook)
            self.hooks.append(handle)
    
    def _schedule_state_reset(self, component_id: str):
        """Reset component state to inactive after a delay"""
        def reset():
            time.sleep(0.5)  # Keep active state for 500ms for visibility
            if component_id in self.current_activations:
                with self._lock:
                    if self.current_activations.get(component_id) in [ComponentState.FORWARD, ComponentState.BACKWARD]:
                        self.current_activations[component_id] = ComponentState.INACTIVE
                        component_registry.update_state(component_id, ComponentState.INACTIVE)
        
        # Run in a separate thread to avoid blocking
        threading.Thread(target=reset, daemon=True).start()
    
    def remove_all_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_current_state(self) -> Dict[str, str]:
        """Get current activation states"""
        with self._lock:
            return {
                component_id: state.value 
                for component_id, state in self.current_activations.items()
            }
    
    def should_broadcast(self) -> bool:
        """Check if enough time has passed to broadcast updates"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

# Global tracker instance
activation_tracker = ActivationTracker()