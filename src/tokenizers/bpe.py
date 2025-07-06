from typing import List, Dict, Optional, Union
import json
import os
import re
from .base import BaseTokenizer

# Try to import tiktoken, fall back gracefully
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. BPE tokenizer will use character fallback.")


class BPETokenizer(BaseTokenizer):
    """BPE tokenizer wrapper supporting tiktoken and custom vocabularies."""
    
    def __init__(self, model_name: str = "gpt2", vocab_size: Optional[int] = None):
        super().__init__()
        self.model_name = model_name
        self.target_vocab_size = vocab_size
        self.encoding = None
        self.custom_vocab = None
        self.char_fallback = None
        
        if TIKTOKEN_AVAILABLE and model_name in ["gpt2", "gpt-3.5-turbo", "gpt-4"]:
            self._init_tiktoken(model_name)
        else:
            # Fall back to character-level tokenization
            print(f"Using character-level fallback for BPE tokenizer")
            from .character import CharacterTokenizer
            self.char_fallback = CharacterTokenizer()
    
    def _init_tiktoken(self, model_name: str):
        """Initialize tiktoken encoder."""
        encoding_map = {
            "gpt2": "gpt2",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base"
        }
        
        encoding_name = encoding_map.get(model_name, "gpt2")
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Store special tokens
        self.special_tokens = {}
        try:
            # Get the endoftext token ID directly from the encoding
            if hasattr(self.encoding, 'eot_token'):
                self.special_tokens['<|endoftext|>'] = self.encoding.eot_token
        except:
            pass
    
    def build_from_text(self, text: str, vocab_size: int = 10000):
        """Build custom BPE vocabulary from text (simplified version)."""
        # For now, we'll use character-level as fallback for custom text
        # A full BPE implementation would require training pairs
        print(f"Building custom vocabulary with size {vocab_size} (using character-level for now)")
        
        from .character import CharacterTokenizer
        self.char_fallback = CharacterTokenizer()
        self.char_fallback.build_from_text(text)
        self.custom_vocab = True
        self.target_vocab_size = self.char_fallback.vocab_size()
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        if self.char_fallback is not None:
            return self.char_fallback.encode(text)
        
        if self.encoding is not None:
            tokens = self.encoding.encode(text)
            # Limit vocab size if specified
            if self.target_vocab_size and self.target_vocab_size < self.vocab_size():
                tokens = [t if t < self.target_vocab_size else 0 for t in tokens]
            return tokens
        
        raise ValueError("No tokenizer initialized")
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        if self.char_fallback is not None:
            return self.char_fallback.decode(tokens)
        
        if self.encoding is not None:
            # Filter out any invalid tokens
            valid_tokens = [t for t in tokens if t < self.vocab_size()]
            return self.encoding.decode(valid_tokens)
        
        raise ValueError("No tokenizer initialized")
    
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        if self.char_fallback is not None:
            return self.char_fallback.vocab_size()
        
        if self.encoding is not None:
            if self.target_vocab_size:
                return min(self.target_vocab_size, self.encoding.n_vocab)
            return self.encoding.n_vocab
        
        return 0
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        if self.char_fallback is not None:
            return self.char_fallback.get_vocab()
        
        if self.encoding is not None:
            # For tiktoken, we can't easily get the full vocab
            # Return a subset or indicator
            return {"<vocab_size>": self.vocab_size()}
        
        return {}
    
    def save(self, path: str):
        """Save tokenizer configuration."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config = {
            'type': self.__class__.__name__,
            'model_name': self.model_name,
            'vocab_size': self.vocab_size(),
            'target_vocab_size': self.target_vocab_size,
            'custom_vocab': self.custom_vocab,
            'special_tokens': self.special_tokens
        }
        
        # If using character fallback, save its config too
        if self.char_fallback is not None:
            char_path = path.replace('.json', '_char.json')
            self.char_fallback.save(char_path)
            config['char_fallback_path'] = os.path.basename(char_path)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer configuration."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.model_name = config.get('model_name', 'gpt2')
        self.target_vocab_size = config.get('target_vocab_size')
        self.custom_vocab = config.get('custom_vocab', False)
        self.special_tokens = config.get('special_tokens', {})
        
        # Load character fallback if it was saved
        if 'char_fallback_path' in config:
            from .character import CharacterTokenizer
            self.char_fallback = CharacterTokenizer()
            char_path = os.path.join(os.path.dirname(path), config['char_fallback_path'])
            self.char_fallback.load(char_path)
        elif TIKTOKEN_AVAILABLE:
            self._init_tiktoken(self.model_name)
    
    def __repr__(self):
        if self.char_fallback is not None:
            return f"BPETokenizer(model={self.model_name}, vocab_size={self.vocab_size()}, mode=character_fallback)"
        return f"BPETokenizer(model={self.model_name}, vocab_size={self.vocab_size()})"