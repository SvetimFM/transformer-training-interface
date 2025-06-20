# LoRA Fine-tuning for Transformer Models

This project now supports **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning of transformer models on custom datasets.

## Features

### 1. **LoRA Implementation**
- Parameter-efficient fine-tuning (typically <1% of model parameters)
- Configurable rank, alpha, and dropout
- Target specific modules (attention layers, output layers)
- Save/load LoRA weights separately from base model

### 2. **Dual UI System**
- **FastAPI** (http://localhost:8000): Original training visualization interface
- **Streamlit** (http://localhost:8501): New LoRA fine-tuning interface

### 3. **Dataset Management**
- Support for TXT, JSON, and CSV file formats
- Automatic text preprocessing and vocabulary alignment
- Dataset statistics and visualization
- Train/validation splitting

### 4. **Training Features**
- Compare LoRA model performance against base model
- Real-time loss tracking and improvement metrics
- Automatic checkpoint saving (best and periodic)
- Parameter efficiency tracking

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Both UIs
```bash
python run_both_ui.py
```

This will start:
- FastAPI server at http://localhost:8000 (base model training)
- Streamlit app at http://localhost:8501 (LoRA fine-tuning)

### 3. Train Base Model First
Use the FastAPI interface to train a base model if you haven't already.

### 4. Fine-tune with LoRA
1. Open Streamlit UI (http://localhost:8501)
2. Upload your custom dataset (Dataset tab)
3. Configure LoRA parameters in the sidebar
4. Initialize models and start training (LoRA Training tab)
5. Test generation with both models (Model Testing tab)
6. View and load saved checkpoints (Results tab)

## LoRA Configuration

Key parameters in the sidebar:
- **Rank**: Lower rank = fewer parameters (default: 8)
- **Alpha**: Scaling factor for LoRA updates (default: 16)
- **Dropout**: Regularization for LoRA layers (default: 0.0)
- **LR Multiplier**: Learning rate multiplier for LoRA parameters

## Architecture

### LoRA Modules Applied To:
- Self-attention layers (Query, Key, Value projections)
- Output projection layers
- Decoder head

### Parameter Efficiency Example:
- Base model: ~500K parameters
- LoRA parameters: ~10K parameters (2% of base)
- Training speedup: 5-10x faster than full fine-tuning

## File Structure

```
src/
├── models/
│   ├── lora.py           # Core LoRA implementation
│   └── lora_model.py     # LoRA-wrapped model
├── training/
│   └── lora_trainer.py   # Specialized LoRA trainer
├── utils/
│   └── lora_dataset.py   # Dataset utilities
└── web/
    └── streamlit_app.py  # Streamlit UI
```

## Saved Checkpoints

LoRA checkpoints are saved in:
```
checkpoints/
└── lora_[dataset_name]/
    ├── lora_best.pt      # Best validation loss
    ├── lora_final.pt     # Final checkpoint
    └── training_info.json # Training metadata
```

## Tips for Best Results

1. **Dataset Size**: LoRA works well even with small datasets (1000+ tokens)
2. **Rank Selection**: Start with rank 8-16, increase if underfitting
3. **Learning Rate**: Can use higher LR than full fine-tuning
4. **Target Modules**: Focus on attention layers for best efficiency

## Technical Details

- LoRA decomposes weight updates into low-rank matrices: ΔW = BA
- Only trains the low-rank matrices A and B
- Original weights remain frozen
- Can merge LoRA weights for deployment: W' = W + BA * (α/r)

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or LoRA rank
2. **No improvement**: Increase rank or learning rate
3. **Overfitting**: Add dropout or reduce training steps
4. **Slow training**: Ensure torch.compile is enabled in settings