"""
Text dataset utilities for PCN language modeling.
"""

import torch
from torch.utils.data import Dataset
import requests
import os
from typing import Tuple, Optional


class TextDataset(Dataset):
    """
    Character-level text dataset for PCN language modeling.
    
    Supports downloading TinyShakespeare or loading custom text files.
    """
    
    def __init__(
        self,
        text_source: str = "shakespeare",
        sequence_length: int = 128,
        train: bool = True,
        train_split: float = 0.9,
        cache_dir: str = "./data"
    ):
        """
        Initialize text dataset.
        
        Args:
            text_source: "shakespeare" or path to text file
            sequence_length: Length of sequences to return
            train: Whether this is training or validation set
            train_split: Fraction of data for training
            cache_dir: Directory to cache downloaded data
        """
        self.sequence_length = sequence_length
        self.train = train
        self.train_split = train_split
        
        # Load or download text
        if text_source == "shakespeare":
            self.text = self._download_shakespeare(cache_dir)
        else:
            with open(text_source, 'r', encoding='utf-8') as f:
                self.text = f.read()
        
        # Split into train/val
        split_idx = int(len(self.text) * train_split)
        if train:
            self.text = self.text[:split_idx]
        else:
            self.text = self.text[split_idx:]
        
        # Create character vocabulary
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        
        print(f"Dataset initialized:")
        print(f"  - Total characters: {len(self.text):,}")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Split: {'train' if train else 'validation'}")
    
    def _download_shakespeare(self, cache_dir: str) -> str:
        """Download TinyShakespeare dataset."""
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, "shakespeare.txt")
        
        if not os.path.exists(filepath):
            print("Downloading TinyShakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Downloaded to {filepath}")
        else:
            print(f"Loading cached dataset from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to tensor of indices."""
        return torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert tensor of indices to text."""
        return ''.join([self.idx_to_char[idx.item()] for idx in indices])
    
    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return len(self.text) - self.sequence_length - 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Returns:
            input_seq: Tensor of shape (sequence_length,)
            target_seq: Tensor of shape (sequence_length,)
        """
        chunk = self.text[idx:idx + self.sequence_length + 1]
        encoded = self.encode(chunk)
        
        input_seq = encoded[:-1]  # All but last
        target_seq = encoded[1:]  # All but first
        
        return input_seq, target_seq
    
    def get_random_chunk(self, length: int) -> str:
        """Get a random chunk of text for generation seeding."""
        import random
        start_idx = random.randint(0, len(self.text) - length - 1)
        return self.text[start_idx:start_idx + length]