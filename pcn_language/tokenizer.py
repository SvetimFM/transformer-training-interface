"""
Simple character-level tokenizer for PCN language modeling.
"""

import torch
from typing import List, Dict, Optional


class CharacterTokenizer:
    """
    Character-level tokenizer.
    
    Maps characters to indices and vice versa.
    """
    
    def __init__(self, chars: Optional[List[str]] = None):
        """
        Initialize tokenizer.
        
        Args:
            chars: List of characters in vocabulary. If None, will be set on first encoding.
        """
        if chars is not None:
            self.chars = sorted(list(set(chars)))
            self.vocab_size = len(self.chars)
            self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        else:
            self.chars = None
            self.vocab_size = 0
            self.char_to_idx = {}
            self.idx_to_char = {}
    
    def fit(self, text: str):
        """Fit tokenizer on text to build vocabulary."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        print(f"Tokenizer vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars[:50])}...")
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to tensor of indices.
        
        Args:
            text: Input text string
            
        Returns:
            Tensor of character indices
        """
        if self.chars is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        indices = []
        for ch in text:
            if ch in self.char_to_idx:
                indices.append(self.char_to_idx[ch])
            else:
                # Handle unknown characters by mapping to first char
                indices.append(0)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode tensor of indices to text.
        
        Args:
            indices: Tensor of character indices
            
        Returns:
            Decoded text string
        """
        if self.chars is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        text = []
        for idx in indices:
            idx_val = idx.item() if torch.is_tensor(idx) else idx
            if idx_val in self.idx_to_char:
                text.append(self.idx_to_char[idx_val])
            else:
                # Handle invalid indices
                text.append(self.chars[0])
        
        return ''.join(text)
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode a batch of texts with padding.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (pad/truncate to this)
            
        Returns:
            Batched tensor of shape (batch_size, max_length)
        """
        if max_length is None:
            max_length = max(len(text) for text in texts)
        
        batch = []
        for text in texts:
            encoded = self.encode(text)
            
            # Truncate if needed
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            
            # Pad if needed (using index 0)
            if len(encoded) < max_length:
                padding = torch.zeros(max_length - len(encoded), dtype=torch.long)
                encoded = torch.cat([encoded, padding])
            
            batch.append(encoded)
        
        return torch.stack(batch)
    
    def batch_decode(self, batch: torch.Tensor) -> List[str]:
        """
        Decode a batch of indices.
        
        Args:
            batch: Tensor of shape (batch_size, seq_length)
            
        Returns:
            List of decoded strings
        """
        texts = []
        for indices in batch:
            texts.append(self.decode(indices))
        return texts