from typing import List, Dict, Optional
import json
import os
from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer that maps each character to a unique ID."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        super().__init__()
        if vocab is not None:
            self.build_from_vocab(vocab)
        else:
            self.char_to_id = {}
            self.id_to_char = {}
    
    def build_from_vocab(self, vocab: List[str]):
        """Build tokenizer from a list of characters."""
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    def build_from_text(self, text: str, min_frequency: int = 1):
        """Build vocabulary from text, optionally filtering by frequency."""
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Filter by minimum frequency and sort
        vocab_chars = sorted([char for char, count in char_counts.items() 
                            if count >= min_frequency])
        
        # Add special tokens at the beginning
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for i, token in enumerate(special_tokens):
            self.special_tokens[token] = i
        
        # Build mappings
        self.char_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_char = {idx: token for idx, token in enumerate(special_tokens)}
        
        # Add regular characters
        for i, char in enumerate(vocab_chars):
            idx = len(special_tokens) + i
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs, using <UNK> for unknown characters."""
        tokens = []
        unk_id = self.special_tokens.get('<UNK>', 1)
        
        for char in text:
            tokens.append(self.char_to_id.get(char, unk_id))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        chars = []
        for token_id in tokens:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # Skip special tokens in decoding
                if char not in self.special_tokens:
                    chars.append(char)
            else:
                # Handle invalid token IDs gracefully
                chars.append('?')
        
        return ''.join(chars)
    
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.char_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return self.char_to_id.copy()
    
    def save(self, path: str):
        """Save tokenizer configuration and vocabulary."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config = {
            'type': self.__class__.__name__,
            'vocab_size': self.vocab_size(),
            'special_tokens': self.special_tokens,
            'char_to_id': self.char_to_id,
            'vocab': list(self.char_to_id.keys())
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """Load tokenizer configuration and vocabulary."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.special_tokens = config.get('special_tokens', {})
        
        if 'char_to_id' in config:
            self.char_to_id = config['char_to_id']
            self.id_to_char = {int(idx): char for char, idx in self.char_to_id.items()}
        elif 'vocab' in config:
            self.build_from_vocab(config['vocab'])
    
    def __repr__(self):
        return f"CharacterTokenizer(vocab_size={self.vocab_size()})"