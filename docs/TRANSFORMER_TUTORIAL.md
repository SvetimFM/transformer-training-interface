# Understanding Transformers: A Hands-On Tutorial

Welcome to this interactive tutorial on transformer neural networks! This guide will take you from the basics to training your own transformer model for text generation.

## Table of Contents
1. [What Are Transformers?](#what-are-transformers)
2. [Key Concepts](#key-concepts)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Hands-On: Your First Transformer](#hands-on-your-first-transformer)
5. [Understanding the Training Process](#understanding-the-training-process)
6. [Advanced Experiments](#advanced-experiments)
7. [Troubleshooting](#troubleshooting)

## What Are Transformers?

Transformers are a revolutionary neural network architecture introduced in the 2017 paper "Attention is All You Need". They've become the foundation for modern language models like GPT, BERT, and T5.

### Why Transformers Matter
- **Parallel Processing**: Unlike RNNs, transformers process all tokens simultaneously
- **Long-Range Dependencies**: Can effectively model relationships between distant words
- **Transfer Learning**: Pre-trained transformers can be fine-tuned for specific tasks
- **Scalability**: Performance improves predictably with model size

## Key Concepts

### 1. Tokens and Embeddings
**Tokens** are the basic units of text that the model processes. They can be:
- Characters: 'h', 'e', 'l', 'l', 'o'
- Subwords: 'hel', 'lo'
- Words: 'hello'

**Embeddings** convert tokens into numerical vectors that capture semantic meaning.

```python
# Example: "Hello world" → [101, 2310, 2088] → [[0.2, -0.5, ...], [0.1, 0.8, ...], ...]
```

### 2. Attention Mechanism
The heart of transformers is the **attention mechanism**, which allows the model to focus on relevant parts of the input when processing each token.

**Self-Attention Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): What information am I looking for?
- K (Key): What information do I have?
- V (Value): The actual information content

### 3. Multi-Head Attention
Instead of one attention mechanism, transformers use multiple "heads" that can focus on different types of relationships:
- Head 1: Grammar relationships
- Head 2: Semantic meaning
- Head 3: Position/order
- etc.

### 4. Position Encodings
Since transformers process all tokens in parallel, they need a way to understand word order. Position encodings add information about each token's position in the sequence.

## Architecture Deep Dive

A transformer consists of stacked layers, each containing:

### 1. Multi-Head Attention Layer
- Computes attention between all token pairs
- Multiple heads capture different relationships
- Includes residual connection and layer normalization

### 2. Feed-Forward Network
- Two linear transformations with ReLU activation
- Processes each position independently
- Also includes residual connection and normalization

### 3. Layer Normalization
- Stabilizes training by normalizing activations
- Can be applied before (pre-norm) or after (post-norm) each sub-layer

## Hands-On: Your First Transformer

Let's train a character-level transformer on Shakespeare text!

### Step 1: Access the Web Interface
1. Open your browser to `http://localhost:8050`
2. You'll see the main training interface

### Step 2: Choose Your Configuration
For your first model, use the **"Tiny Shakespeare"** recipe:
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Layers**: 4
- **Sequence Length**: 256
- **Batch Size**: 64
- **Learning Rate**: 3e-4

### Step 3: Understanding the Parameters

#### Model Architecture
- **Embedding Dimension**: Size of token representations (higher = more expressive)
- **Number of Heads**: Parallel attention mechanisms (must divide embedding dimension)
- **Number of Layers**: Depth of the network (more = better but slower)
- **Dropout**: Regularization to prevent overfitting (0.1 = drop 10% of connections)

#### Training Configuration
- **Sequence Length**: How many characters the model sees at once
- **Batch Size**: Number of sequences processed together
- **Learning Rate**: How fast the model learns (too high = unstable, too low = slow)
- **Weight Decay**: Another regularization technique

### Step 4: Start Training
1. Click **"Initialize Model"** - This creates your transformer
2. Click **"Start Training"** - Watch the metrics!

### What to Watch For:
- **Loss**: Should decrease over time (lower is better)
- **Perplexity**: Measures prediction quality (lower is better)
- **Learning Rate**: Follows the schedule (warmup → training → fine-tuning)

### Step 5: Generate Text
Once loss drops below 2.0:
1. Enter a prompt like "To be or not to be"
2. Set temperature (0.8 is a good default)
3. Click **"Generate"** and see your model's creativity!

## Understanding the Training Process

### Loss Function
We use **cross-entropy loss**, which measures how wrong the model's predictions are. For each position, the model predicts the next character, and we compare this to the actual next character.

### Learning Rate Schedule
Our training uses three phases:
1. **Warmup** (10% of steps): Gradually increase learning rate
2. **Main Training** (80% of steps): Constant high learning rate
3. **Fine-tuning** (10% of steps): Lower learning rate for refinement

### Monitoring Progress
- **Loss < 3.0**: Model is learning basic patterns
- **Loss < 2.0**: Generating readable text
- **Loss < 1.5**: High-quality generation
- **Loss < 1.0**: Potential overfitting

## Advanced Experiments

### 1. Architecture Experiments
Try these variations:
- **Deeper Models**: Increase layers to 6 or 8
- **Wider Models**: Increase embedding dimension to 256 or 512
- **More Heads**: Try 8 or 16 attention heads

### 2. Training Experiments
- **Learning Rate**: Try 1e-3 (faster) or 1e-4 (more stable)
- **Batch Size**: Larger batches (128) for stability, smaller (32) for speed
- **Sequence Length**: Longer sequences (512) capture more context

### 3. Tokenization Experiments
Switch between:
- **Character-level**: Simple, great for learning
- **BPE (GPT-2)**: More efficient, better for real text

### 4. PCN Experiments (Advanced)
Explore our research on Predictive Coding Networks:
1. Switch to the **"PCN Experiments"** tab
2. Compare standard vs PCN-enhanced transformers
3. Investigate the "label leakage" phenomenon

## Troubleshooting

### High Loss / Not Learning
- **Check learning rate**: Too high causes instability, too low prevents learning
- **Verify data**: Ensure dataset loaded correctly
- **Reduce model size**: Start smaller, then scale up

### Out of Memory
- **Reduce batch size**: Halve it until training works
- **Reduce sequence length**: Try 128 instead of 256
- **Use smaller model**: Fewer layers or smaller embeddings

### Generating Gibberish
- **Train longer**: Model needs more time
- **Check temperature**: Too high (>1.5) causes randomness
- **Verify tokenizer**: Ensure encode/decode works correctly

### GPU Not Being Used
- Check GPU availability in the status panel
- Ensure PyTorch is installed with CUDA support
- Try restarting the application

## Next Steps

1. **Experiment with Different Datasets**: Upload your own text files
2. **Try LoRA Fine-tuning**: Efficient adaptation to new tasks
3. **Explore Attention Patterns**: Visualize what the model learns
4. **Read the Research**: Check our PCN experiments for cutting-edge ideas

## Additional Resources

- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Visualization Guide](./ATTENTION_VISUALIZATION.md)
- [PCN Research Summary](./analysis/reports/pcn_executive_summary.md)

## Quick Reference Card

### Recommended Settings by Goal

**Fast Learning (See results quickly)**:
- Tiny model (4 layers, 128 dim)
- High learning rate (3e-3)
- Character tokenization

**Best Quality (Production-like)**:
- Large model (8 layers, 512 dim)
- Moderate learning rate (3e-4)
- BPE tokenization
- Train for 10,000+ steps

**Research/Experimentation**:
- Try PCN variants
- Compare attention mechanisms
- Experiment with hybrid architectures

Remember: The best way to understand transformers is to experiment! Start with the tiny model, watch the metrics, and gradually increase complexity as you build intuition.

Happy training! 🚀