from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch.nn as nn
from enum import Enum

class ComponentType(Enum):
    EMBEDDING = "embedding"
    LINEAR = "linear"
    ATTENTION = "attention"
    ATTENTION_HEAD = "attention_head"
    LAYER_NORM = "layer_norm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    ADD = "add"
    SOFTMAX = "softmax"
    TRANSFORMER_BLOCK = "transformer_block"
    FEED_FORWARD = "feed_forward"

class ComponentState(Enum):
    INACTIVE = "inactive"
    FORWARD = "forward"
    BACKWARD = "backward"
    COMPUTING = "computing"
    DISABLED = "disabled"

@dataclass
class ComponentInfo:
    id: str
    name: str
    type: ComponentType
    module: Optional[nn.Module] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # layer, index within layer
    state: ComponentState = ComponentState.INACTIVE
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "params": self.params,
            "position": self.position,
            "state": self.state.value
        }

class ComponentRegistry:
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.module_to_id: Dict[nn.Module, str] = {}
        self._id_counter = 0
        
    def generate_id(self, prefix: str) -> str:
        """Generate unique component ID"""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"
    
    def register_component(
        self,
        module: Optional[nn.Module],
        name: str,
        component_type: ComponentType,
        parent_id: Optional[str] = None,
        **params
    ) -> str:
        """Register a component and return its ID"""
        component_id = self.generate_id(component_type.value)
        
        component = ComponentInfo(
            id=component_id,
            name=name,
            type=component_type,
            module=module,
            parent_id=parent_id,
            params=params
        )
        
        self.components[component_id] = component
        
        if module is not None:
            self.module_to_id[module] = component_id
        
        # Update parent's children list
        if parent_id and parent_id in self.components:
            self.components[parent_id].children_ids.append(component_id)
        
        return component_id
    
    def get_component_by_module(self, module: nn.Module) -> Optional[ComponentInfo]:
        """Get component info by PyTorch module"""
        component_id = self.module_to_id.get(module)
        return self.components.get(component_id) if component_id else None
    
    def get_component(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component by ID"""
        return self.components.get(component_id)
    
    def update_state(self, component_id: str, state: ComponentState):
        """Update component state"""
        if component_id in self.components:
            self.components[component_id].state = state
    
    def update_state_by_module(self, module: nn.Module, state: ComponentState):
        """Update component state by module"""
        component = self.get_component_by_module(module)
        if component:
            component.state = state
    
    def get_architecture_graph(self) -> Dict[str, Any]:
        """Get the full architecture graph for visualization"""
        return {
            "components": {cid: c.to_dict() for cid, c in self.components.items()},
            "root_components": [
                cid for cid, c in self.components.items() 
                if c.parent_id is None
            ]
        }
    
    def reset_states(self):
        """Reset all component states to inactive"""
        for component in self.components.values():
            component.state = ComponentState.INACTIVE
    
    def get_active_components(self) -> List[str]:
        """Get list of currently active component IDs"""
        return [
            cid for cid, c in self.components.items()
            if c.state != ComponentState.INACTIVE and c.state != ComponentState.DISABLED
        ]

# Global registry instance
component_registry = ComponentRegistry()