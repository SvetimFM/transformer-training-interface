from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import os


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""
    
    def __init__(self):
        self.special_tokens = {}
        
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        pass
    
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        pass
    
    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        pass
    
    def save(self, path: str):
        """Save tokenizer configuration and vocabulary."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config = {
            'type': self.__class__.__name__,
            'vocab_size': self.vocab_size(),
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @abstractmethod
    def load(self, path: str):
        """Load tokenizer configuration and vocabulary."""
        pass
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def batch_decode(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists."""
        return [self.decode(tokens) for tokens in token_lists]