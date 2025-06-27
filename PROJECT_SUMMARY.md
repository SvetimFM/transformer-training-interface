# Transformer PCN UI - Project Summary

## Overview

This project provides an interactive web-based interface for training transformers and analyzing Predictive Coding Networks (PCNs), with a special focus on demonstrating the label leakage issue in recent PCN research.

## Key Components

### 1. Web Application (`src/web/`)
- **app.py**: FastAPI backend with WebSocket support for real-time updates
- **pcn_manager.py**: Manages PCN experiments and comparisons
- **templates/index.html**: Main UI with three tabs (Transformer, PCN, Hybrid)
- **static/**: JavaScript (Chart.js visualizations) and CSS files

### 2. Models (`src/models/`)
- **bigram.py**: GPT-style transformer implementation
- **pcn_layers.py**: Predictive Coding Network components
- **hybrid_architectures.py**: Various PCN-Transformer hybrid models

### 3. Training (`src/training/`)
- **trainer.py**: Training loop with mixed precision, gradient accumulation
- **schedulers.py**: Learning rate scheduling (warmup + cosine)

### 4. Visualization (`src/visualization/`)
- **hooks.py**: Model architecture visualization
- **activation_tracker.py**: Real-time activation monitoring
- **attention_capture.py**: Attention pattern analysis

## Main Features

### Standard Transformer Training
- Configurable architecture (layers, heads, dimensions)
- Real-time loss curves and metrics
- Text generation with temperature control
- Attention pattern visualization

### PCN Label Leakage Demonstration
Shows the critical flaw in PCN evaluation:
- **Problematic**: Uses true labels during test inference → 99.92% accuracy
- **Correct**: No label access during testing → 40-50% accuracy

The issue is in the test loop:
```python
# WRONG: Uses true labels during testing
eps_sup = y_hat - y_batch  # y_batch has true labels!
```

### Hybrid Models (WIP)
Experimental architectures combining PCN and transformers:
- PCN-FF (PCN replaces feedforward)
- Alternating layers
- Hierarchical processing
- Dual-stream
- PCN-based positional encoding

## Technical Stack
- **Backend**: FastAPI, PyTorch 2.0+
- **Frontend**: Vanilla JS, Chart.js, D3.js
- **Real-time**: WebSockets
- **Styling**: Dark theme with custom CSS

## Quick Start
```bash
# Setup (one time)
python setup.py

# Run
./run.sh  # Linux/Mac
run.bat   # Windows
```

## Project Structure
```
├── src/              # Source code
├── config.py         # Configuration
├── run_ui.py         # Entry point
├── setup.py          # Setup script
├── requirements.txt  # Dependencies
├── README.md         # Documentation
├── LICENSE           # MIT License
└── .gitignore        # Git ignore rules
```

## Key Insights

1. **PCN Label Leakage**: The paper's 99.92% accuracy claim is due to using test labels during inference, not superior architecture.

2. **Proper Evaluation**: Without label access, PCN performs similarly to other methods (~40-50% on CIFAR-10 with minimal training).

3. **Educational Value**: This UI helps researchers understand proper evaluation methodology and the importance of preventing data leakage.

## Future Enhancements
- Complete hybrid model implementations
- Add more datasets
- Model checkpointing
- Distributed training support
- Docker containerization

## Publication Ready
The project includes:
- Comprehensive README
- MIT License
- .gitignore for Python projects
- Setup scripts for easy installation
- Contributing guidelines
- Clean, documented code

Ready to publish on GitHub or other platforms!