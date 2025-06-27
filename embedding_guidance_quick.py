"""
Quick Embedding Guidance Experiment

A faster version of the embedding guidance experiment for rapid testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('src')

from models.embedding_matching_transformer import (
    EmbeddingMatchingTransformer,
    StandardTransformer
)
from data_loader import load_data, get_batch


def quick_experiment():
    # Configuration for quick test
    config = {
        'batch_size': 16,
        'block_size': 64,
        'learning_rate': 3e-4,
        'max_steps': 200,  # Much fewer steps
        'eval_interval': 20,
        'eval_iters': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Quick Embedding Guidance Experiment")
    print("=" * 60)
    print(f"Device: {config['device']}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Model configuration (smaller models)
    model_config = {
        'vocab_size': vocab_size,
        'block_size': config['block_size'],
        'n_embed': 64,  # Even smaller
        'n_heads': 2,
        'n_layers': 2,
        'dropout': 0.1,
        'device': config['device']
    }
    
    # Create models
    print("\nCreating models...")
    
    # Standard Transformer
    standard_model = StandardTransformer(**model_config).to(config['device'])
    print(f"Standard model parameters: {sum(p.numel() for p in standard_model.parameters()):,}")
    
    # Embedding-Matching Transformer
    embedding_model = EmbeddingMatchingTransformer(
        **model_config,
        n_future_predict=3,  # Fewer predictions
        n_hypotheses=2,      # Fewer hypotheses
        n_pcn_iterations=3,  # Fewer iterations
        use_multiscale=False,
        use_refinement=False,  # Skip refinement for speed
        embedding_loss_weight=1.0,
        token_loss_weight=0.1
    ).to(config['device'])
    print(f"Embedding model parameters: {sum(p.numel() for p in embedding_model.parameters()):,}")
    
    # Training loop
    results = {'standard': {}, 'embedding': {}}
    
    for model_name, model in [('standard', standard_model), ('embedding', embedding_model)]:
        print(f"\n{'='*60}")
        print(f"Training {model_name} model...")
        print(f"{'='*60}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        
        train_losses = []
        val_losses = []
        start_time = time.time()
        
        for step in range(config['max_steps']):
            # Training
            model.train()
            xb, yb = get_batch('train', train_data, config['block_size'], 
                              config['batch_size'], config['device'])
            
            if model_name == 'embedding':
                logits, loss, stats = model(xb, yb)
            else:
                logits, loss, stats = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Evaluation
            if step % config['eval_interval'] == 0:
                model.eval()
                val_loss_batch = []
                
                for _ in range(config['eval_iters']):
                    xb, yb = get_batch('val', val_data, config['block_size'],
                                      config['batch_size'], config['device'])
                    with torch.no_grad():
                        _, vloss, _ = model(xb, yb)
                        val_loss_batch.append(vloss.item())
                
                val_loss = np.mean(val_loss_batch)
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                
                print(f"Step {step}: train {loss.item():.3f}, val {val_loss:.3f}")
        
        training_time = time.time() - start_time
        
        # Generate sample
        model.eval()
        with torch.no_grad():
            context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
            if model_name == 'embedding':
                generated = model.generate(
                    context, max_new_tokens=100, temperature=0.8, 
                    do_sample=True, use_embedding_guidance=True
                )
            else:
                generated = model.generate(
                    context, max_new_tokens=100, temperature=0.8, do_sample=True
                )
        
        results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1],
            'training_time': training_time,
            'sample': decode(generated[0].cpu().numpy())
        }
    
    # Visualization
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training curves
    steps = np.arange(0, len(results['standard']['val_losses'])) * config['eval_interval']
    ax1.plot(steps, results['standard']['val_losses'], 'b-', label='Standard', linewidth=2)
    ax1.plot(steps, results['embedding']['val_losses'], 'r-', label='Embedding-Matching', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final metrics
    models = ['Standard', 'Embedding']
    final_losses = [results['standard']['final_val_loss'], results['embedding']['final_val_loss']]
    times = [results['standard']['training_time'], results['embedding']['training_time']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.7)
    bars2 = ax2.bar(x + width/2, np.array(times)/times[0], width, 
                    label='Relative Time', alpha=0.7)
    
    ax2.set_xlabel('Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Value')
    ax2.set_title('Performance vs Efficiency')
    ax2.legend()
    
    # Add value labels
    for bars in [bars1]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('embedding_guidance_quick_results.png', dpi=150)
    plt.close()
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    improvement = ((results['standard']['final_val_loss'] - results['embedding']['final_val_loss']) / 
                  results['standard']['final_val_loss'] * 100)
    
    print(f"\nPerformance:")
    print(f"  Standard:    {results['standard']['final_val_loss']:.3f}")
    print(f"  Embedding:   {results['embedding']['final_val_loss']:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    print(f"\nEfficiency:")
    print(f"  Standard time:    {results['standard']['training_time']:.1f}s")
    print(f"  Embedding time:   {results['embedding']['training_time']:.1f}s")
    print(f"  Time overhead:    {results['embedding']['training_time']/results['standard']['training_time']:.1f}x")
    
    print("\nGenerated Samples:")
    print("-"*60)
    print("Standard:")
    print(results['standard']['sample'][:200])
    print("\nEmbedding-Matching:")
    print(results['embedding']['sample'][:200])
    
    print("\n✓ Quick experiment complete!")
    print("  Results saved to embedding_guidance_quick_results.png")


if __name__ == '__main__':
    quick_experiment()