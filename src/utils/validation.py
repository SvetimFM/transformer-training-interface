"""
Validation utilities for model configuration
"""

def validate_model_config(n_embed: int, n_heads: int) -> tuple[bool, str]:
    """
    Validate that the model configuration is valid.
    
    Args:
        n_embed: Embedding dimension
        n_heads: Number of attention heads
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if n_embed is divisible by n_heads
    if n_embed % n_heads != 0:
        head_size = n_embed / n_heads
        error_msg = (
            f"Invalid configuration: n_embed ({n_embed}) must be divisible by n_heads ({n_heads}). "
            f"Current head_size would be {head_size:.2f}, but it must be an integer. "
            f"Try n_embed values like: {', '.join(str(n_heads * i) for i in range(8, 17))}"
        )
        return False, error_msg
    
    # Check minimum head size
    head_size = n_embed // n_heads
    if head_size < 4:
        error_msg = (
            f"Invalid configuration: head_size ({head_size}) is too small. "
            f"With n_heads={n_heads}, you need n_embed >= {n_heads * 4}. "
            f"Consider reducing n_heads or increasing n_embed."
        )
        return False, error_msg
    
    # Warn if head size is not a power of 2 (not required but recommended)
    if head_size & (head_size - 1) != 0:  # Check if not power of 2
        warning_msg = (
            f"Note: head_size ({head_size}) is not a power of 2. "
            f"While this works, powers of 2 (8, 16, 32, 64) are often more efficient."
        )
        return True, warning_msg
    
    return True, ""

def get_valid_embed_sizes(n_heads: int, min_head_size: int = 8, max_head_size: int = 128) -> list[int]:
    """
    Get a list of valid embedding sizes for a given number of heads.
    
    Args:
        n_heads: Number of attention heads
        min_head_size: Minimum size per head
        max_head_size: Maximum size per head
        
    Returns:
        List of valid n_embed values
    """
    valid_sizes = []
    head_size = min_head_size
    while head_size <= max_head_size:
        valid_sizes.append(n_heads * head_size)
        head_size *= 2  # Focus on powers of 2
    return valid_sizes