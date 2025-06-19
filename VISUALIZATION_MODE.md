# Visualization Mode

A special training mode that dramatically slows down the training process to allow visual observation of data flow through the transformer architecture.

## Features

### 1. **Speed Control**
- Toggle switch to enable/disable visualization mode
- Speed slider to control training speed (1% to 100% of normal speed)
- Default: 1% speed for maximum visibility

### 2. **Visual Indicators**
- **Active Component Highlighting**: Currently processing component glows with green pulse animation
- **Phase Display**: Shows "Forward Pass" or "Backward Pass" 
- **Step Counter**: Displays current training step
- **Enhanced Animations**: Stronger visual effects in visualization mode

### 3. **Architecture Integration**
- Real-time component state updates
- Persistent highlighting of recently active components
- Visual flow indicators showing data movement
- Attention head identification

## How to Use

1. **Enable Visualization Mode**
   - Toggle the "Visualization Mode" switch in the Architecture panel
   - The speed control slider will appear

2. **Adjust Speed**
   - Use the slider to set training speed (1-100%)
   - 1% = Very slow (best for observation)
   - 10% = Slow (good balance)
   - 50% = Half speed
   - 100% = Normal speed

3. **Start Training**
   - Click "Start Training" to begin
   - Watch as components light up during forward/backward passes
   - Observe the flow of data through attention heads, layer norms, etc.

## Implementation Details

### Training Delay Calculation
```python
if visualization_mode and visualization_speed_ratio > 0:
    delay = (1 - visualization_speed_ratio) / visualization_speed_ratio * 0.1
    time.sleep(min(delay, 5.0))  # Cap at 5 seconds
```

### Visual States
- **Green Glow + Pulse**: Active component during forward pass
- **Orange Glow**: Active component during backward pass
- **Strong Border**: Currently processing component in visualization mode

### Performance Impact
- Training runs at selected percentage of normal speed
- WebSocket updates are more frequent in visualization mode
- All visualization features can be toggled off for normal training

## Educational Benefits

1. **Understanding Transformer Architecture**
   - See exactly how data flows through the model
   - Observe attention mechanism in action
   - Understand forward and backward passes

2. **Debugging Aid**
   - Identify which components are active at each step
   - Spot potential bottlenecks or issues
   - Verify model architecture behavior

3. **Teaching Tool**
   - Demonstrate transformer mechanics to students
   - Show real-time neural network operation
   - Explain concepts with visual feedback