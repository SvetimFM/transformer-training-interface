import torch
from typing import Optional
from .dataset_manager import DatasetManager
from config import app_config

# Global dataset manager instance
_dataset_manager: Optional[DatasetManager] = None
_current_tokenizer = None
_current_text = None


def get_dataset():
    """Get dataset text. Legacy function for compatibility."""
    global _current_text
    if _current_text is None:
        manager = get_dataset_manager()
        _current_text = manager.load_dataset()
    return _current_text


def prepare_vocab():
    """Prepare vocabulary. Legacy function for compatibility."""
    global _current_tokenizer
    
    manager = get_dataset_manager()
    text = manager.load_dataset()
    
    if app_config.dataset.tokenizer_type == "character":
        # For backward compatibility, return character list
        vocab = sorted(list(set(text)))
        print("Number of unique characters:", len(vocab))
        print(vocab)
        return vocab
    else:
        # For BPE, create tokenizer and return None (will need to update calling code)
        tokenizer = manager.create_tokenizer(text)
        _current_tokenizer = tokenizer
        print(f"Using {app_config.dataset.tokenizer_type} tokenizer with vocab size: {tokenizer.vocab_size()}")
        return None


def get_dataset_manager() -> DatasetManager:
    """Get or create the global dataset manager."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager(app_config.dataset)
    return _dataset_manager


def get_tokenizer():
    """Get the current tokenizer."""
    global _current_tokenizer
    if _current_tokenizer is None:
        manager = get_dataset_manager()
        text = manager.load_dataset()
        _current_tokenizer = manager.create_tokenizer(text)
    return _current_tokenizer
