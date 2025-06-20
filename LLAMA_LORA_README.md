# LLaMA LoRA Fine-tuning Guide

This project now supports fine-tuning Large Language Models (LLaMA, Mistral, etc.) using Low-Rank Adaptation (LoRA) with state-of-the-art optimizations.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This installs transformers, PEFT, bitsandbytes for quantization, and other required libraries.

### 2. Launch the UI
```bash
# Launch all three UIs
python run_all_ui.py

# Or just the LLaMA UI
cd src && streamlit run web/streamlit_llama.py --server.port 8502
```

### 3. Access the LLaMA LoRA UI
Open http://localhost:8502 in your browser

## 📊 Supported Models

### Pre-configured Models:
- **TinyLlama 1.1B** - Great for testing and development
- **Llama 2 7B** - Most popular size
- **Llama 2 13B** - Better quality, more VRAM
- **Llama 2 70B** - Best quality (requires 4-bit)
- **Mistral 7B** - Alternative architecture
- **Any HuggingFace model** - Use custom model path

### Memory Requirements:

| Model | FP16 | 8-bit | 4-bit | LoRA Training (8-bit) |
|-------|------|-------|-------|---------------------|
| TinyLlama 1.1B | 2 GB | 1 GB | 0.5 GB | 3-4 GB |
| Llama 2 7B | 14 GB | 7 GB | 3.5 GB | 10-12 GB |
| Llama 2 13B | 26 GB | 13 GB | 6.5 GB | 16-20 GB |
| Llama 2 70B | 140 GB | 70 GB | 35 GB | 40-48 GB |

## 🎯 LoRA Configuration

### Key Parameters:

- **Rank (r)**: 
  - Lower (4-8): Fewer parameters, faster training, may underfit
  - Medium (16-32): Good balance for most tasks
  - Higher (64+): More expressiveness, risk of overfitting

- **Alpha**: 
  - Typically set to 2x rank
  - Controls the scaling of LoRA updates

- **Target Modules**:
  - `q_proj, v_proj`: Minimum (fastest, least parameters)
  - `q_proj, k_proj, v_proj, o_proj`: Attention only (recommended)
  - All modules: Maximum quality, slowest training

### Recommended Settings by Task:

**Instruction Tuning**:
```python
rank = 16
alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
learning_rate = 2e-4
```

**Domain Adaptation**:
```python
rank = 32
alpha = 64
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
learning_rate = 1e-4
```

**Style Transfer**:
```python
rank = 8
alpha = 16
target_modules = ["q_proj", "v_proj"]
learning_rate = 3e-4
```

## 📁 Dataset Format

### Supported Formats:

1. **Plain Text** (.txt)
   - One sample per paragraph (separated by double newlines)
   
2. **JSON** (.json)
   ```json
   [
     "First training sample",
     "Second training sample",
     "Third training sample"
   ]
   ```
   
3. **JSONL** (.jsonl)
   ```json
   {"text": "First training sample"}
   {"text": "Second training sample"}
   ```

4. **CSV** (.csv)
   - Text column will be used

### Dataset Guidelines:

- **Minimum samples**: 100-1000 for good results
- **Sample length**: 50-2000 tokens each
- **Quality**: Clean, relevant examples
- **Diversity**: Varied examples prevent overfitting

## 🔧 Advanced Features

### Quantization (Memory Saving)

**8-bit Loading**:
- Reduces memory by ~50%
- Minimal quality loss
- Recommended for 7B/13B models

**4-bit Loading**:
- Reduces memory by ~75%
- Some quality loss
- Required for 70B models on consumer GPUs

### Training Optimizations

- **Gradient Accumulation**: Simulate larger batches
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision (FP16)**: Faster training
- **Flash Attention**: If available

### Multi-GPU Support

The system automatically uses `device_map="auto"` for multi-GPU setups. Model layers are distributed across available GPUs.

## 💻 Command Line Usage

### Basic Training Script
```python
from models.llama_lora import create_llama_lora
from training.llama_lora_trainer import create_llama_lora_trainer

# Create trainer
trainer = create_llama_lora_trainer(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_path="my_dataset.txt",
    output_dir="./my_lora_model",
    load_in_8bit=True,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    num_epochs=3
)

# Train
trainer.train()

# Save
trainer.save_model()
```

### Loading and Using LoRA Model
```python
from models.llama_lora import LLaMALoRA
from transformers import AutoTokenizer

# Load model with LoRA
model = LLaMALoRA.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    adapter_path="./my_lora_model",
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 📈 Performance Tips

1. **Start Small**: Test with TinyLlama before scaling up
2. **Use Quantization**: 8-bit for most cases, 4-bit for large models
3. **Batch Size 1**: With gradient accumulation for stability
4. **Monitor Memory**: Use `nvidia-smi` to track GPU usage
5. **Save Checkpoints**: Enable periodic saving
6. **Validation Split**: Keep 10% for validation

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size to 1
- Enable 8-bit or 4-bit loading
- Reduce max sequence length
- Use fewer target modules
- Enable gradient checkpointing

### Slow Training
- Ensure you're using GPU: `torch.cuda.is_available()`
- Enable torch.compile if supported
- Use mixed precision (FP16)
- Reduce sequence length

### Poor Results
- Increase training epochs
- Try higher rank (32-64)
- Add more target modules
- Check dataset quality
- Adjust learning rate

### Loading Errors
- Ensure model name is correct
- Check HuggingFace login: `huggingface-cli login`
- Verify sufficient disk space for model download

## 📊 Monitoring Training

The UI provides:
- Real-time loss tracking
- Training/validation curves
- Parameter efficiency metrics
- Memory usage monitoring
- Checkpoint management

Training history is saved in:
```
your_output_dir/
├── checkpoint-100/
├── checkpoint-200/
├── trainer_state.json  # Full training history
└── training_args.bin
```

## 🔗 Integration with HuggingFace

### Push to Hub:
```python
trainer.push_to_hub("username/my-llama-lora")
```

### Load from Hub:
```python
model = LLaMALoRA.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    adapter_path="username/my-llama-lora"
)
```

## 🎓 Understanding LoRA

LoRA works by decomposing weight updates:
- Instead of updating W (d×k), we update A (d×r) and B (r×k)
- Where r << min(d,k), typically r=16 for 7B models
- This reduces parameters from d×k to r×(d+k)

For Llama 2 7B:
- Full fine-tuning: 7B parameters
- LoRA (r=16, attention only): ~40M parameters (0.6%)
- Memory saved: >90%
- Quality retained: >95%

## 🚀 Next Steps

1. Start with TinyLlama to learn the workflow
2. Prepare your custom dataset
3. Experiment with different LoRA configurations
4. Scale up to larger models as needed
5. Share your adapters on HuggingFace Hub!

For more examples and advanced usage, check the `examples/` directory.