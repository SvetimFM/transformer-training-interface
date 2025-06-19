# Attention Pattern Visualization

This implementation visualizes how attention mechanisms sample and aggregate information from the embedding space.

## Key Features

1. **Attention Weight Capture**
   - Modified `SelfAttentionHead` to optionally return attention weights
   - Created `AttentionCapture` class to collect weights during forward passes
   - Integrated with existing model registration system

2. **Interactive Visualization**
   - Heatmap display of attention patterns for each head
   - Color-coded weights (dark blue = low attention, red = high attention)
   - Custom text input for analyzing different sequences
   - Shows how Query-Key dot products create sampling patterns

3. **Understanding K, Q, V**
   - **Query (Q)**: "What information am I looking for?"
   - **Key (K)**: "What information do I contain?"
   - **Value (V)**: "The actual content to aggregate"
   - Attention scores (Q·K^T) determine sampling weights
   - Final output is weighted sum of Values

## Usage

1. Navigate to http://localhost:8000/static/attention.html
2. Enter any text to analyze
3. Click "Capture Attention Patterns"
4. View the attention heatmaps for each head

## Technical Details

- Attention weights are captured without affecting training performance
- Weights are normalized (sum to 1) and show causal masking
- Each head learns different attention patterns
- GPU-efficient matrix multiplication: `Attention = softmax(QK^T/sqrt(d_k)) * V`

## Why Not Multiple Queries (Q1...Qn)?

- Current multi-head attention already provides multiple query perspectives
- Single Q per position is more GPU-efficient
- Multiple heads learn different Q, K, V projections
- This design balances expressiveness with computational efficiency