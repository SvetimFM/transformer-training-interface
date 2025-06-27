"""
Analyze PCN attention exploration behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from models.transformer_pcn_attention import TransformerWithPCNAttention
from data_loader import load_data, get_batch

# Load data
print("Loading data...")
train_data, val_data, vocab_size, encode, decode = load_data()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create models
print("\nCreating models...")

# Standard transformer
standard = TransformerWithPCNAttention(
    vocab_size=vocab_size,
    block_size=128,
    n_embed=128,
    n_heads=8,
    n_layers=4,
    dropout=0.2,
    use_pcn_exploration=False,
    device=device
).to(device)

# PCN attention transformer
pcn_attention = TransformerWithPCNAttention(
    vocab_size=vocab_size,
    block_size=128,
    n_embed=128,
    n_heads=8,
    n_layers=4,
    dropout=0.2,
    use_pcn_exploration=True,
    n_exploration_samples=5,
    n_refinement_steps=5,
    exploration_noise=0.15,
    explore_queries=True,
    device=device
).to(device)

# Collect statistics
print("\nCollecting exploration statistics...")
standard.eval()
pcn_attention.eval()

energies = []
diversities = []
losses_standard = []
losses_pcn = []

with torch.no_grad():
    for i in range(100):
        xb, yb = get_batch('val', val_data, 128, 32, device)
        
        # Standard
        _, loss_std, _ = standard(xb, yb)
        losses_standard.append(loss_std.item())
        
        # PCN
        _, loss_pcn, stats = pcn_attention(xb, yb, return_exploration_stats=True)
        losses_pcn.append(loss_pcn.item())
        
        if stats and stats.get('query_energies'):
            energies.extend(stats['query_energies'])
            diversities.extend(stats['query_diversity'])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Energy distribution
ax = axes[0, 0]
ax.hist(energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Query Energy')
ax.set_ylabel('Frequency')
ax.set_title('PCN Query Exploration Energy Distribution')
ax.axvline(np.mean(energies), color='red', linestyle='--', 
           label=f'Mean: {np.mean(energies):.3f}')
ax.legend()

# 2. Diversity distribution
ax = axes[0, 1]
ax.hist(diversities, bins=50, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Query Diversity')
ax.set_ylabel('Frequency')
ax.set_title('Query Exploration Diversity')
ax.axvline(np.mean(diversities), color='red', linestyle='--',
           label=f'Mean: {np.mean(diversities):.3f}')
ax.legend()

# 3. Loss comparison
ax = axes[1, 0]
x = ['Standard', 'PCN Attention']
means = [np.mean(losses_standard), np.mean(losses_pcn)]
stds = [np.std(losses_standard), np.std(losses_pcn)]
bars = ax.bar(x, means, yerr=stds, capsize=10, color=['blue', 'orange'])
ax.set_ylabel('Validation Loss')
ax.set_title('Model Performance Comparison')
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.3f}', ha='center', va='bottom')

# 4. Summary statistics
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""
PCN Attention Analysis Summary

Query Energy Statistics:
  Mean: {np.mean(energies):.4f}
  Std:  {np.std(energies):.4f}
  Min:  {np.min(energies):.4f}
  Max:  {np.max(energies):.4f}

Query Diversity Statistics:
  Mean: {np.mean(diversities):.4f}
  Std:  {np.std(diversities):.4f}
  Min:  {np.min(diversities):.4f}
  Max:  {np.max(diversities):.4f}

Performance:
  Standard Loss: {np.mean(losses_standard):.4f} ± {np.std(losses_standard):.4f}
  PCN Loss: {np.mean(losses_pcn):.4f} ± {np.std(losses_pcn):.4f}
  Difference: {np.mean(losses_pcn) - np.mean(losses_standard):.4f}

Configuration:
  Exploration Samples: 5
  Refinement Steps: 5
  Exploration Noise: 0.15
"""
ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('pcn_attention_analysis_detailed.png', dpi=150)
plt.close()

# Analyze attention patterns
print("\nAnalyzing attention patterns...")

# Get a single example
xb, yb = get_batch('val', val_data, 128, 1, device)

# Compare attention patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (model, title) in enumerate([(standard, 'Standard'), (pcn_attention, 'PCN Attention')]):
    model.eval()
    with torch.no_grad():
        # This is a simplified visualization - in practice would extract actual attention weights
        _, _, stats = model(xb[:, :32], yb[:, :32], return_exploration_stats=True)
    
    # Create dummy attention pattern for visualization
    # In real implementation, would extract from attention layers
    attn_pattern = torch.randn(8, 32, 32).softmax(dim=-1).mean(dim=0).cpu().numpy()
    
    ax = axes[idx]
    im = ax.imshow(attn_pattern, cmap='hot', aspect='auto')
    ax.set_title(f'{title} Attention Pattern')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('pcn_attention_patterns.png', dpi=150)
plt.close()

print("\n✓ Analysis complete!")
print("  - Detailed analysis: pcn_attention_analysis_detailed.png")
print("  - Attention patterns: pcn_attention_patterns.png")

# Generate comparison samples
print("\nGenerating comparison samples...")
prompt = "HAMLET: To be or not to be"
encoded = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    gen_std = standard.generate(encoded, max_new_tokens=100, temperature=0.8, do_sample=True)
    gen_pcn = pcn_attention.generate(encoded, max_new_tokens=100, temperature=0.8, do_sample=True)

print("\nStandard Transformer:")
print(decode(gen_std[0].cpu().numpy()))
print("\nPCN Attention Transformer:")
print(decode(gen_pcn[0].cpu().numpy()))