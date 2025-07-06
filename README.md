# Transformer Training UI with PCN Analysis

An interactive web-based interface for training transformers and analyzing Predictive Coding Networks (PCNs). This project includes a comprehensive demonstration of the label leakage issue found in recent PCN research.

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

### 🚀 Standard Transformer Training
- Real-time training visualization with loss curves
- Configurable architecture (layers, heads, dimensions)
- Learning rate scheduling with warmup and cosine annealing
- Gradient accumulation and mixed precision training
- Interactive architecture visualization
- Text generation with attention pattern analysis

### 🔬 PCN Experiments
- **Label Leakage Analysis**: Demonstrates the critical flaw in PCN evaluation methodology
- Side-by-side comparison of problematic vs correct implementation
- Real-time accuracy tracking showing:
  - Problematic implementation: ~99.92% accuracy (unrealistic)
  - Correct implementation: ~40-50% accuracy (realistic)
- Energy distribution and diversity score visualizations

### 🏗️ Hybrid Models (Experimental)
- PCN-Transformer hybrid architectures
- Performance comparison with standard transformers
- Multiple architecture variants:
  - PCN-FF: PCN replaces feedforward networks
  - Alternating: Attention ↔ PCN layers
  - Hierarchical: PCN features → Transformer
  - Dual-Stream: Parallel PCN + Transformer
  - PCN-Positional: Adaptive positional encoding

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended)
- 4GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SvetimFM/transformer-pcn-ui.git
cd transformer-pcn-ui
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Shakespeare dataset (auto-downloads on first run)

## Usage

### Starting the UI

```bash
python run_ui.py
```

The UI will be available at `http://localhost:8000`

### Configuration

The application uses a configuration system that can be modified through the UI or by editing `config.py`:

- **Model Configuration**: Adjust transformer architecture parameters
- **Training Configuration**: Set batch size, learning rate, epochs, etc.
- **Generation Configuration**: Control text generation parameters

### Key Features Usage

#### 1. Standard Transformer Training
- Navigate to the "Standard Transformer" tab
- Configure model architecture using the sliders
- Click "Apply Architecture" to update the model
- Click "Start Training" to begin training
- Monitor real-time metrics and loss curves

#### 2. PCN Label Leakage Experiment
- Navigate to the "PCN Experiments" tab
- Click "Run Comparison" to start the experiment
- Observe the dramatic difference between:
  - **With Label Leakage**: Shows unrealistic ~99.92% accuracy
  - **Without Label Leakage**: Shows realistic ~40-50% accuracy
- Review the code comparison showing the exact issue

#### 3. Text Generation
- After training, use the text generation panel
- Enter a prompt and adjust temperature/max tokens
- Click "Generate" to see model output
- Click "View Attention Patterns" for detailed analysis

## Understanding the PCN Label Leakage Issue

The PCN experiment demonstrates a critical flaw found in the paper ["Introduction to Predictive Coding Networks for Machine Learning"](https://arxiv.org/pdf/2506.06332):

### The Problem
During testing, the paper's implementation uses true labels to update internal representations:

```python
# Problematic code from the paper:
eps_sup = y_hat - y_batch  # y_batch contains TRUE LABELS during testing!
# This error is then used to update the model's internal state
```

### Why It's Wrong
1. Test labels should NEVER be available during inference
2. The model gets 50 steps to "correct" itself using the true answer
3. This leads to unrealistic 99.92% accuracy claims

### Correct Approach
Testing should only use model predictions without access to true labels:
```python
# Correct implementation:
logits = model(x)
predictions = logits.argmax(dim=1)
# No label information used during inference
```

## Project Structure

```
├── src/
│   ├── web/
│   │   ├── app.py              # FastAPI application
│   │   ├── pcn_manager.py      # PCN experiment manager
│   │   ├── static/             # Frontend assets
│   │   └── templates/          # HTML templates
│   ├── models/
│   │   ├── bigram.py           # Transformer implementation
│   │   └── pcn_layers.py       # PCN components
│   ├── training/
│   │   └── trainer.py          # Training logic
│   └── visualization/
│       └── hooks.py            # Architecture visualization
├── config.py                   # Configuration management
├── run_ui.py                   # Entry point
└── requirements.txt            # Dependencies
```

## API Endpoints

The application provides a REST API for programmatic access:

- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration
- `POST /api/train` - Start/stop training
- `GET /api/train/status` - Get training status
- `POST /api/generate` - Generate text
- `POST /api/pcn/start-experiment` - Start PCN experiment
- `WebSocket /ws` - Real-time metrics streaming

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{transformer_pcn_ui,
  title = {Transformer Training UI with PCN Analysis},
  year = {2024},
  url = {https://github.com/SvetimFM/transformer-pcn-ui}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch, FastAPI, and Chart.js
- Inspired by the need for transparent ML research
- Thanks to the open-source community

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model dimensions
2. **Port already in use**: Change port in `run_ui.py` or kill existing process
3. **Module not found**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG  # On Windows: set LOG_LEVEL=DEBUG
python run_ui.py
```

## Future Work

- [ ] Complete hybrid model implementations
- [ ] Add more datasets beyond Shakespeare
- [ ] Implement model checkpointing
- [ ] Add distributed training support
- [ ] Create Docker container for easy deployment

---

**Note**: The PCN label leakage demonstration is for educational purposes to highlight the importance of proper evaluation methodology in machine learning research.