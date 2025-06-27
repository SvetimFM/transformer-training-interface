"""
Analyze and visualize PCN exploration behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from models.transformer_with_pcn_exploration import TransformerWithPCNExploration
from data_loader import load_data, get_batch

# Load data
print("Loading data...")
train_data, val_data, vocab_size, encode, decode = load_data()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create PCN exploration model
model = TransformerWithPCNExploration(
    vocab_size=vocab_size,
    batch_size=32,
    block_size=128,
    n_embed=128,
    n_heads=8,
    n_layers=4,
    dropout=0.2,
    use_pcn_exploration=True,
    exploration_points="final",
    n_exploration_samples=5,
    n_exploration_steps=10,
    exploration_noise=0.2,
    device=device
).to(device)

# Collect exploration statistics over multiple batches
print("\nCollecting exploration statistics...")
model.eval()

all_energies = []
all_diversities = []
all_refinements = []

with torch.no_grad():
    for i in range(50):
        xb, yb = get_batch('val', val_data, 128, 32, device)
        _, _, stats = model(xb, yb, return_exploration_stats=True)
        
        if stats['energies']:
            all_energies.extend(stats['energies'])
        if stats['diversity']:
            all_diversities.extend(stats['diversity'])
        if stats['refinement_delta']:
            all_refinements.extend(stats['refinement_delta'])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Energy distribution
ax = axes[0, 0]
ax.hist(all_energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Energy')
ax.set_ylabel('Frequency')
ax.set_title('PCN Exploration Energy Distribution')
ax.axvline(np.mean(all_energies), color='red', linestyle='--', 
           label=f'Mean: {np.mean(all_energies):.3f}')
ax.legend()

# 2. Diversity distribution
ax = axes[0, 1]
ax.hist(all_diversities, bins=50, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Diversity Score')
ax.set_ylabel('Frequency')
ax.set_title('Exploration Diversity Distribution')
ax.axvline(np.mean(all_diversities), color='red', linestyle='--',
           label=f'Mean: {np.mean(all_diversities):.3f}')
ax.legend()

# 3. Refinement magnitude
ax = axes[1, 0]
ax.hist(all_refinements, bins=50, alpha=0.7, color='orange', edgecolor='black')
ax.set_xlabel('Refinement Delta (L2 norm)')
ax.set_ylabel('Frequency')
ax.set_title('PCN Refinement Magnitude')
ax.axvline(np.mean(all_refinements), color='red', linestyle='--',
           label=f'Mean: {np.mean(all_refinements):.3f}')
ax.legend()

# 4. Summary statistics
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""
PCN Exploration Analysis Summary

Energy Statistics:
  Mean: {np.mean(all_energies):.4f}
  Std:  {np.std(all_energies):.4f}
  Min:  {np.min(all_energies):.4f}
  Max:  {np.max(all_energies):.4f}

Diversity Statistics:
  Mean: {np.mean(all_diversities):.4f}
  Std:  {np.std(all_diversities):.4f}
  Min:  {np.min(all_diversities):.4f}
  Max:  {np.max(all_diversities):.4f}

Refinement Statistics:
  Mean: {np.mean(all_refinements):.4f}
  Std:  {np.std(all_refinements):.4f}
  Min:  {np.min(all_refinements):.4f}
  Max:  {np.max(all_refinements):.4f}

Model Configuration:
  Samples: 5
  Steps: 10
  Noise: 0.2
"""
ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('pcn_exploration_analysis.png', dpi=150)
plt.close()

# Generate comparison samples
print("\nGenerating comparison samples...")
context = "ROMEO: "
encoded_context = torch.tensor(encode(context), dtype=torch.long, device=device).unsqueeze(0)

# Without exploration
model.use_pcn_exploration = False
with torch.no_grad():
    gen_no_explore = model.generate(encoded_context, max_new_tokens=100, 
                                   temperature=0.8, do_sample=True)
    text_no_explore = decode(gen_no_explore[0].cpu().numpy())

# With exploration
model.use_pcn_exploration = True
with torch.no_grad():
    gen_explore = model.generate(encoded_context, max_new_tokens=100,
                                temperature=0.8, do_sample=True)
    text_explore = decode(gen_explore[0].cpu().numpy())

print("\nGeneration without PCN exploration:")
print(text_no_explore)
print("\nGeneration with PCN exploration:")
print(text_explore)

# Save analysis results
results = {
    'energies': all_energies,
    'diversities': all_diversities,
    'refinements': all_refinements,
    'generation_no_explore': text_no_explore,
    'generation_explore': text_explore,
    'stats': {
        'energy_mean': np.mean(all_energies),
        'energy_std': np.std(all_energies),
        'diversity_mean': np.mean(all_diversities),
        'diversity_std': np.std(all_diversities),
        'refinement_mean': np.mean(all_refinements),
        'refinement_std': np.std(all_refinements)
    }
}

torch.save(results, 'pcn_exploration_analysis_results.pt')

print("\n✓ Analysis complete!")
print("  - Visualization: pcn_exploration_analysis.png")
print("  - Results: pcn_exploration_analysis_results.pt")