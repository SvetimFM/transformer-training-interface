# Embedding Shape and Head Size Validation

## Overview

The transformer architecture requires that the embedding dimension (`n_embed`) is evenly divisible by the number of attention heads (`n_heads`). This validation system prevents configuration errors before they cause runtime failures.

## Key Constraints

### 1. **Divisibility Requirement**
- `n_embed` must be divisible by `n_heads`
- Formula: `head_size = n_embed / n_heads` must be an integer
- Example: `n_embed=256, n_heads=8` → `head_size=32` ✓

### 2. **Minimum Head Size**
- Each attention head needs at least 4 dimensions
- `head_size >= 4`
- Example: `n_embed=32, n_heads=8` → `head_size=4` ✓ (minimum)

### 3. **Recommended Head Sizes** (Optional)
- Powers of 2 are recommended for efficiency: 8, 16, 32, 64, 128
- Non-power-of-2 sizes work but may be less efficient
- System shows a warning for non-optimal sizes

## User Interface Features

### Real-Time Validation
As users adjust the sliders, they see immediate feedback:
- ✓ **Green**: Valid configuration with optimal head size
- ℹ️ **Yellow**: Valid but not optimal (non-power-of-2)
- ⚠️ **Red**: Invalid configuration with helpful error message

### Error Prevention
- "Apply Changes" button is blocked for invalid configurations
- Clear error messages explain the problem
- Suggested valid values are provided

### Visual Feedback Examples
```
✓ head_size = 32                    # Perfect
ℹ️ head_size = 24 (works, but not power of 2)  # Warning
⚠️ Invalid: head_size = 32.5 (must be integer)  # Error
⚠️ head_size = 2 (too small, need >= 4)        # Error
```

## Common Configurations

### Small Models
- `n_embed=128, n_heads=4` → `head_size=32`
- `n_embed=256, n_heads=8` → `head_size=32`

### Medium Models
- `n_embed=512, n_heads=8` → `head_size=64`
- `n_embed=768, n_heads=12` → `head_size=64` (BERT-base)

### Large Models
- `n_embed=1024, n_heads=16` → `head_size=64`
- `n_embed=2048, n_heads=32` → `head_size=64`

## Quick Reference Table

| n_heads | Valid n_embed values | Head sizes |
|---------|---------------------|------------|
| 1       | 64, 128, 256, 512   | 64, 128, 256, 512 |
| 2       | 64, 128, 256, 512   | 32, 64, 128, 256 |
| 4       | 64, 128, 256, 512   | 16, 32, 64, 128 |
| 8       | 64, 128, 256, 512, 1024 | 8, 16, 32, 64, 128 |
| 12      | 96, 192, 384, 768   | 8, 16, 32, 64 |
| 16      | 128, 256, 512, 1024 | 8, 16, 32, 64 |

## Implementation Details

### Client-Side Validation (JavaScript)
```javascript
if (n_embed % n_heads !== 0) {
    // Show error with suggested values
}
```

### Server-Side Validation (Python)
```python
if n_embed % n_heads != 0:
    raise HTTPException(400, "Invalid configuration...")
```

## Benefits

1. **Prevents Runtime Errors**: Catches configuration issues before model creation
2. **Educational**: Helps users understand transformer constraints
3. **User-Friendly**: Provides immediate feedback and suggestions
4. **Robust**: Validates on both client and server sides

## Troubleshooting

### "Invalid configuration" error
- Check that n_embed is divisible by n_heads
- Try one of the suggested values
- Reduce n_heads or increase n_embed

### "Head size too small" error
- Increase n_embed or decrease n_heads
- Minimum viable: n_embed = n_heads × 4

### Performance warning
- Consider using powers of 2 for head_size
- Common optimal sizes: 8, 16, 32, 64