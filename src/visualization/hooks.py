import torch.nn as nn
from typing import Optional
from .component_registry import ComponentType, component_registry
from .activation_tracker import activation_tracker
from .attention_capture import attention_capture

def register_model_components(model: nn.Module, config=None):
    """Register all model components for visualization"""
    
    # Clear existing registrations
    component_registry.components.clear()
    component_registry.module_to_id.clear()
    activation_tracker.remove_all_hooks()
    attention_capture.clear()
    
    # Register main model
    model_id = component_registry.register_component(
        model, 
        "Transformer Model", 
        ComponentType.TRANSFORMER_BLOCK,
        n_layers=config.model.n_layers if config else 1
    )
    
    # Track the last component in the chain
    last_component_id = model_id
    
    # Register embeddings
    if hasattr(model, 'token_embedding_table'):
        tok_emb_id = component_registry.register_component(
            model.token_embedding_table,
            "Token Embeddings",
            ComponentType.EMBEDDING,
            parent_id=last_component_id,
            vocab_size=model.vocab_size,
            n_embed=model.token_embedding_table.embedding_dim
        )
        activation_tracker.add_forward_hook(model.token_embedding_table, tok_emb_id)
        last_component_id = tok_emb_id
    
    if hasattr(model, 'position_embedding_table'):
        pos_emb_id = component_registry.register_component(
            model.position_embedding_table,
            "Positional Embeddings",
            ComponentType.EMBEDDING,
            parent_id=model_id,  # Position embeddings are parallel to token embeddings
            block_size=model.block_size,
            n_embed=model.position_embedding_table.embedding_dim
        )
        activation_tracker.add_forward_hook(model.position_embedding_table, pos_emb_id)
        
        # Register embedding add operation (virtual component)
        embed_add_id = component_registry.register_component(
            None,
            "Embedding Add",
            ComponentType.ADD,
            parent_id=last_component_id
        )
        # Connect position embeddings to the add node as well
        component_registry.components[pos_emb_id].children_ids.append(embed_add_id)
        last_component_id = embed_add_id
    
    # Register dropout after embeddings
    if hasattr(model, 'dropout'):
        dropout_id = component_registry.register_component(
            model.dropout,
            "Embedding Dropout",
            ComponentType.DROPOUT,
            parent_id=last_component_id,
            p=model.dropout.p
        )
        activation_tracker.add_forward_hook(model.dropout, dropout_id)
        last_component_id = dropout_id
    
    # Register transformer blocks in sequence
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            block_id = register_transformer_block(block, f"Transformer Block {i+1}", last_component_id, i)
            last_component_id = block_id  # Each block becomes parent of the next
    
    # Register final layer norm
    if hasattr(model, 'ln_f') and not isinstance(model.ln_f, nn.Identity):
        ln_f_id = component_registry.register_component(
            model.ln_f,
            "Final Layer Norm",
            ComponentType.LAYER_NORM,
            parent_id=last_component_id
        )
        activation_tracker.add_forward_hook(model.ln_f, ln_f_id)
        last_component_id = ln_f_id
    
    # Register output activation
    if hasattr(model, 'output_activation'):
        activation_name = model.output_activation.__class__.__name__
        output_act_id = component_registry.register_component(
            model.output_activation,
            f"Output Activation ({activation_name})",
            ComponentType.ACTIVATION,
            parent_id=last_component_id
        )
        activation_tracker.add_forward_hook(model.output_activation, output_act_id)
        last_component_id = output_act_id
    
    # Register output hidden layers
    if hasattr(model, 'output_layers') and len(model.output_layers) > 0:
        layer_num = 1
        for i, layer in enumerate(model.output_layers):
            if isinstance(layer, nn.Linear):
                layer_id = component_registry.register_component(
                    layer,
                    f"Output Hidden Layer {layer_num}",
                    ComponentType.LINEAR,
                    parent_id=last_component_id,
                    in_features=layer.in_features,
                    out_features=layer.out_features
                )
                activation_tracker.add_forward_hook(layer, layer_id)
                last_component_id = layer_id
                layer_num += 1
            elif isinstance(layer, nn.Dropout):
                dropout_id = component_registry.register_component(
                    layer,
                    f"Output Dropout {layer_num-1}",
                    ComponentType.DROPOUT,
                    parent_id=last_component_id,
                    p=layer.p
                )
                activation_tracker.add_forward_hook(layer, dropout_id)
                last_component_id = dropout_id
            elif hasattr(layer, '__class__') and 'ReLU' in layer.__class__.__name__ or 'GELU' in layer.__class__.__name__ or 'SiLU' in layer.__class__.__name__:
                act_name = layer.__class__.__name__
                act_id = component_registry.register_component(
                    layer,
                    f"Hidden Activation ({act_name})",
                    ComponentType.ACTIVATION,
                    parent_id=last_component_id
                )
                activation_tracker.add_forward_hook(layer, act_id)
                last_component_id = act_id
    
    # Register decoder head with appropriate name
    if hasattr(model, 'decoder_head'):
        # Check if there are output hidden layers
        has_output_layers = hasattr(model, 'output_layers') and len(model.output_layers) > 0
        projection_name = "Vocabulary Projection" if has_output_layers else "Output Projection"
        
        decoder_id = component_registry.register_component(
            model.decoder_head,
            projection_name,
            ComponentType.LINEAR,
            parent_id=last_component_id,
            in_features=model.decoder_head.in_features,
            out_features=model.decoder_head.out_features
        )
        activation_tracker.add_forward_hook(model.decoder_head, decoder_id)
        activation_tracker.add_backward_hook(model.decoder_head, decoder_id)
    
    return model_id

def register_transformer_block(block: nn.Module, name: str, parent_id: str, layer_idx: int) -> str:
    """Register a transformer block and its components"""
    
    block_id = component_registry.register_component(
        block,
        name,
        ComponentType.TRANSFORMER_BLOCK,
        parent_id=parent_id,
        layer_idx=layer_idx
    )
    
    # Pre-norm 1
    if hasattr(block, 'ln1') and block.use_layer_norm and block.norm_position == "pre":
        ln1_id = component_registry.register_component(
            block.ln1,
            "Pre-Attention Norm",
            ComponentType.LAYER_NORM,
            parent_id=block_id
        )
        activation_tracker.add_forward_hook(block.ln1, ln1_id)
    
    # Multi-head attention
    if hasattr(block, 'attention'):
        attn_id = register_multi_head_attention(
            block.attention, 
            "Multi-Head Attention", 
            block_id,
            layer_idx
        )
    
    # Post-norm 1
    if hasattr(block, 'ln1') and block.use_layer_norm and block.norm_position == "post":
        ln1_id = component_registry.register_component(
            block.ln1,
            "Post-Attention Norm",
            ComponentType.LAYER_NORM,
            parent_id=block_id
        )
        activation_tracker.add_forward_hook(block.ln1, ln1_id)
    
    # Pre-norm 2
    if hasattr(block, 'ln2') and block.use_layer_norm and block.norm_position == "pre":
        ln2_id = component_registry.register_component(
            block.ln2,
            "Pre-FFN Norm",
            ComponentType.LAYER_NORM,
            parent_id=block_id
        )
        activation_tracker.add_forward_hook(block.ln2, ln2_id)
    
    # Feed-forward network
    if hasattr(block, 'feed_forward'):
        ffn_id = register_feed_forward(
            block.feed_forward,
            "Feed Forward",
            block_id
        )
    
    # Post-norm 2
    if hasattr(block, 'ln2') and block.use_layer_norm and block.norm_position == "post":
        ln2_id = component_registry.register_component(
            block.ln2,
            "Post-FFN Norm",
            ComponentType.LAYER_NORM,
            parent_id=block_id
        )
        activation_tracker.add_forward_hook(block.ln2, ln2_id)
    
    # Residual connections (virtual components)
    if block.use_residual:
        res1_id = component_registry.register_component(
            None,
            "Residual Connection 1",
            ComponentType.ADD,
            parent_id=block_id
        )
        
        res2_id = component_registry.register_component(
            None,
            "Residual Connection 2",
            ComponentType.ADD,
            parent_id=block_id
        )
    
    return block_id

def register_multi_head_attention(mha: nn.Module, name: str, parent_id: str, layer_idx: int = 0) -> str:
    """Register multi-head attention and its heads"""
    
    mha_id = component_registry.register_component(
        mha,
        name,
        ComponentType.ATTENTION,
        parent_id=parent_id,
        num_heads=len(mha.heads) if hasattr(mha, 'heads') else 0
    )
    
    # Register individual attention heads
    if hasattr(mha, 'heads'):
        for i, head in enumerate(mha.heads):
            head_id = component_registry.register_component(
                head,
                f"Head {i+1}",
                ComponentType.ATTENTION_HEAD,
                parent_id=mha_id,
                head_size=head.head_size if hasattr(head, 'head_size') else 0
            )
            activation_tracker.add_forward_hook(head, head_id)
            activation_tracker.add_backward_hook(head, head_id)
            
            # Register for attention capture
            attention_capture.register_attention_head(head, head_id, i, layer_idx)
    
    return mha_id

def register_feed_forward(ffn: nn.Module, name: str, parent_id: str) -> str:
    """Register feed-forward network components"""
    
    ffn_id = component_registry.register_component(
        ffn,
        name,
        ComponentType.FEED_FORWARD,
        parent_id=parent_id
    )
    
    activation_tracker.add_forward_hook(ffn, ffn_id)
    activation_tracker.add_backward_hook(ffn, ffn_id)
    
    return ffn_id