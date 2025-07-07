# Transformer Training Interface

An educational implementation of transformer neural networks with a web-based training interface. This project demonstrates modern transformer architectures with interactive visualizations for learning and experimentation.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📚 Tutorial

New to transformers? Check out our comprehensive [Transformer Tutorial](docs/TRANSFORMER_TUTORIAL.md) that covers:
- Understanding transformer architecture
- Key concepts explained simply
- Hands-on training walkthrough
- Troubleshooting guide
- Advanced experiments

## Features

### Core Functionality
- **Interactive Transformer Training**: Train GPT-style decoder-only transformers with real-time metrics
- **Multiple Tokenization Options**: Character-level and BPE tokenization support
- **Dataset Flexibility**: Built-in datasets (Shakespeare, Wikipedia) or upload custom text
- **Real-time Visualizations**: 
  - Loss curves and training metrics
  - Architecture visualization with D3.js
  - Attention pattern analysis
  - Learning rate schedules
  - Activation states during training

### Educational Design
- **Comprehensive Tutorial**: Step-by-step guide from transformer basics to advanced concepts
- **Implementation Transparency**: Tooltips explain architectural choices and trade-offs
- **Modern Best Practices**: Implements current techniques (AdamW, cosine scheduling, pre-norm)
- **Proven Recipes**: Pre-configured settings for different compute budgets

## Quick Start

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended)
- 4GB+ RAM

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd transformer-training-interface

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.web.app
```

Navigate to `http://localhost:8000` to access the interface.

### Basic Usage

1. **Choose a Recipe**: Start with the "Beginner" configuration for quick results
2. **Select Dataset**: Shakespeare is pre-loaded and works well for demos
3. **Configure Model**: Adjust layers, heads, and embedding dimensions
4. **Start Training**: Click "Start Training" and watch real-time metrics
5. **Generate Text**: Use the generation panel to sample from your model
6. **Analyze Attention**: Click "View Attention Patterns" to visualize what the model learns

## Architecture

### Model Implementation
- **Decoder-only Transformer**: GPT-style architecture optimized for text generation
- **Learned Positional Embeddings**: More flexible than sinusoidal for specific tasks
- **Pre-LayerNorm**: More stable training than post-norm
- **Separate Attention Heads**: Educational implementation for better understanding

### Training Features
- **Mixed Precision Training**: Optional FP16 for faster training
- **Gradient Accumulation**: Simulate larger batches on limited hardware
- **Learning Rate Scheduling**: Warmup + cosine decay
- **PyTorch 2.0 Compilation**: Optional torch.compile for performance

## Project Structure

```
transformer-training-interface/
├── src/
│   ├── models/           # Transformer implementation
│   ├── training/         # Training loop and utilities
│   ├── visualization/    # Real-time metrics and viz
│   ├── tokenizers/       # Character and BPE tokenizers
│   ├── utils/            # Dataset handling, utilities
│   └── web/              # FastAPI web interface
├── docs/                 # Tutorials and documentation
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Configuration Options

### Model Parameters
- `n_layers`: Number of transformer blocks (1-12)
- `n_heads`: Attention heads per layer (1-16)
- `n_embed`: Embedding dimension (64-1024)
- `block_size`: Maximum sequence length (32-512)
- `dropout`: Dropout probability (0.0-0.5)

### Training Parameters
- `batch_size`: Training batch size (4-128)
- `learning_rate`: Initial learning rate (1e-5 to 1e-3)
- `num_epochs`: Training epochs (1-100)
- `grad_accumulation_steps`: Gradient accumulation (1-16)
- `warmup_steps`: Learning rate warmup steps
- `lr_scheduler`: "cosine" or "linear"

### Generation Parameters
- `temperature`: Sampling temperature (0.1-2.0)
- `top_k`: Top-k sampling (0 = disabled)
- `top_p`: Nucleus sampling threshold (0.0-1.0)

## Recipes for Success

### Tiny Shakespeare Dataset

#### Beginner (Fast Training)
- **Layers**: 2, **Heads**: 4, **Embed Dim**: 128
- **Batch Size**: 64, **Learning Rate**: 3e-4
- ~300K parameters, trains in minutes

#### Intermediate (Better Quality)
- **Layers**: 4, **Heads**: 8, **Embed Dim**: 256
- **Batch Size**: 32, **Learning Rate**: 2e-4
- ~2.5M parameters, good results

#### Advanced (Best Quality)
- **Layers**: 6, **Heads**: 6, **Embed Dim**: 384
- **Batch Size**: 16, **Learning Rate**: 1e-4
- ~10M parameters, authentic Shakespeare

## API Documentation

The application provides a REST API for programmatic access:

- `GET /api/config`: Get current configuration
- `POST /api/config`: Update configuration
- `POST /api/train`: Start/stop training
- `GET /api/train/status`: Get training status
- `POST /api/generate`: Generate text
- `GET /api/metrics/history`: Get training history
- `POST /api/attention/capture`: Capture attention patterns
- `GET /api/architecture`: Get model architecture graph

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the seminal "Attention Is All You Need" paper
- Inspired by Andrej Karpathy's educational implementations
- Uses modern best practices from the transformer community

## Citation

If you use this project in your research or teaching, please cite:

```bibtex
@software{transformer_training_interface,
  title = {Transformer Training Interface},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/yourusername/transformer-training-interface}
}
```