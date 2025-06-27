"""
Data loader for TinyShakespeare dataset.
"""

import torch
import os
import requests
import numpy as np


def download_shakespeare():
    """Download TinyShakespeare if not present."""
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'shakespeare.txt')
    
    if not os.path.exists(file_path):
        print("Downloading TinyShakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open(file_path, 'w') as f:
            f.write(response.text)
        print(f"Downloaded to {file_path}")
    
    return file_path


def load_data():
    """Load and prepare TinyShakespeare data."""
    # Download if needed
    file_path = download_shakespeare()
    
    # Read text
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode/decode functions
    encode = lambda s: [char_to_idx[c] for c in s]
    decode = lambda l: ''.join([idx_to_char[i] for i in l])
    
    # Encode entire text
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split into train/val (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Dataset loaded: {len(data):,} characters, vocab size: {vocab_size}")
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    return train_data, val_data, vocab_size, encode, decode


def get_batch(split, data, block_size, batch_size, device):
    """Get a batch of data."""
    # Generate random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to device
    x, y = x.to(device), y.to(device)
    
    return x, y


if __name__ == '__main__':
    # Test data loading
    train_data, val_data, vocab_size, encode, decode = load_data()
    
    # Test batch generation
    x, y = get_batch('train', train_data, block_size=128, batch_size=4, device='cpu')
    print(f"\nBatch shape: {x.shape}")
    
    # Test encode/decode
    test_text = "To be or not to be"
    encoded = encode(test_text)
    decoded = decode(encoded)
    print(f"\nOriginal: '{test_text}'")
    print(f"Encoded: {encoded[:10]}...")
    print(f"Decoded: '{decoded}'")