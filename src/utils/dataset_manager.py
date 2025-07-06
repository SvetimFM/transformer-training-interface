import os
import requests
from typing import Optional, Tuple
from ..tokenizers import CharacterTokenizer, BPETokenizer, BaseTokenizer
from ..config import DatasetConfig


class DatasetManager:
    """Manages dataset loading and tokenization."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer: Optional[BaseTokenizer] = None
        self.text: Optional[str] = None
        self.dataset_path = ".datasets"
        os.makedirs(self.dataset_path, exist_ok=True)
    
    def load_dataset(self) -> str:
        """Load dataset based on configuration."""
        if self.config.dataset_type == "shakespeare":
            return self._load_shakespeare()
        elif self.config.dataset_type == "custom":
            return self._load_custom_file()
        elif self.config.dataset_type == "url":
            return self._load_from_url()
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
    
    def _load_shakespeare(self) -> str:
        """Load the default Shakespeare dataset."""
        shakespeare_path = "./.dependencies/pretraining_dataset.txt"
        if not os.path.exists(shakespeare_path):
            # Download if not exists
            print("Downloading Tiny Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            response = requests.get(url)
            os.makedirs(os.path.dirname(shakespeare_path), exist_ok=True)
            with open(shakespeare_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_custom_file(self) -> str:
        """Load a custom text file."""
        if not self.config.dataset_path:
            raise ValueError("No dataset path provided for custom dataset")
        
        # Check file size
        file_size_mb = os.path.getsize(self.config.dataset_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB limit")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_from_url(self) -> str:
        """Download and cache a dataset from URL."""
        if not self.config.dataset_url:
            raise ValueError("No URL provided for URL dataset")
        
        # Create cache filename from URL
        cache_name = self.config.dataset_url.split('/')[-1]
        if not cache_name.endswith('.txt'):
            cache_name += '.txt'
        cache_path = os.path.join(self.dataset_path, cache_name)
        
        # Download if not cached
        if not os.path.exists(cache_path):
            print(f"Downloading dataset from {self.config.dataset_url}...")
            response = requests.get(self.config.dataset_url, stream=True)
            
            # Check size while downloading
            size = 0
            chunks = []
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                size += len(chunk)
                if size > self.config.max_file_size_mb * 1024 * 1024:
                    raise ValueError(f"Download exceeds {self.config.max_file_size_mb}MB limit")
                chunks.append(chunk)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                for chunk in chunks:
                    f.write(chunk)
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_tokenizer(self, text: Optional[str] = None) -> BaseTokenizer:
        """Create and initialize tokenizer based on configuration."""
        if text is None:
            text = self.text or self.load_dataset()
        
        if self.config.tokenizer_type == "character":
            tokenizer = CharacterTokenizer()
            tokenizer.build_from_text(text)
        elif self.config.tokenizer_type == "bpe":
            tokenizer = BPETokenizer(model_name=self.config.tokenizer_model)
            if self.config.tokenizer_model == "custom":
                tokenizer.build_from_text(text, vocab_size=self.config.vocab_size or 10000)
        else:
            raise ValueError(f"Unknown tokenizer type: {self.config.tokenizer_type}")
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def get_dataset_info(self, text: Optional[str] = None) -> dict:
        """Get information about the dataset."""
        if text is None:
            text = self.text or self.load_dataset()
        
        # Basic stats
        info = {
            'size_bytes': len(text.encode('utf-8')),
            'size_mb': len(text.encode('utf-8')) / (1024 * 1024),
            'character_count': len(text),
            'line_count': text.count('\n'),
            'unique_chars': len(set(text))
        }
        
        # Tokenizer stats
        if self.tokenizer:
            sample = text[:1000]  # Sample for token count estimation
            tokens = self.tokenizer.encode(sample)
            info['vocab_size'] = self.tokenizer.vocab_size()
            info['tokens_per_char'] = len(tokens) / len(sample) if sample else 0
            info['estimated_tokens'] = int(len(text) * info['tokens_per_char'])
        
        return info
    
    def prepare_data(self) -> Tuple[str, BaseTokenizer, dict]:
        """Load dataset, create tokenizer, and return everything needed for training."""
        # Load dataset
        self.text = self.load_dataset()
        
        # Create tokenizer
        self.tokenizer = self.create_tokenizer(self.text)
        
        # Get info
        info = self.get_dataset_info(self.text)
        
        # Note: The calling code should update the model vocab size in config
        
        return self.text, self.tokenizer, info