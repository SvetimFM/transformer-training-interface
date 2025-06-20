"""
Dataset utilities for LoRA fine-tuning.
Handles custom dataset loading, preprocessing, and management.
"""

import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from collections import Counter


class LoRADatasetManager:
    """Manages datasets for LoRA fine-tuning."""
    
    def __init__(self, base_vocab: List[str], device: str = 'cuda'):
        self.base_vocab = base_vocab
        self.vocab_size = len(base_vocab)
        self.device = device
        
        # Create mappings
        self.char_to_idx = {c: i for i, c in enumerate(base_vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(base_vocab)}
        
        # Encode/decode functions
        self.encode = lambda s: [self.char_to_idx.get(c, 0) for c in s]  # Use idx 0 for unknown chars
        self.decode = lambda l: ''.join([self.idx_to_char.get(i, '') for i in l])
        
    def load_dataset(self, file_path: str, file_type: str = 'auto') -> str:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to dataset file
            file_type: Type of file ('txt', 'json', 'csv', or 'auto')
            
        Returns:
            Raw text content
        """
        if file_type == 'auto':
            file_type = Path(file_path).suffix[1:]  # Remove the dot
            
        if file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Assume JSON has a 'text' field or is a list of texts
                if isinstance(data, list):
                    return '\n'.join(str(item) for item in data)
                elif isinstance(data, dict) and 'text' in data:
                    return data['text']
                else:
                    # Convert entire JSON to string
                    return json.dumps(data)
                    
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
            # Concatenate all string columns
            text_columns = df.select_dtypes(include=['object']).columns
            texts = []
            for col in text_columns:
                texts.extend(df[col].dropna().astype(str).tolist())
            return '\n'.join(texts)
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def prepare_dataset(
        self,
        text: str,
        train_split: float = 0.9,
        max_length: Optional[int] = None,
        augment: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Prepare dataset for training.
        
        Args:
            text: Raw text content
            train_split: Fraction of data for training
            max_length: Maximum sequence length (truncate if longer)
            augment: Whether to apply data augmentation
            
        Returns:
            train_data, val_data, dataset_info
        """
        # Clean text
        text = self._clean_text(text)
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
            
        # Augment if requested
        if augment:
            text = self._augment_text(text)
            
        # Get dataset statistics
        dataset_info = self._analyze_dataset(text)
        
        # Encode text
        encoded = self.encode(text)
        data = torch.tensor(encoded, dtype=torch.long)
        
        # Split into train/val
        train_size = int(train_split * len(data))
        train_data = data[:train_size].to(self.device)
        val_data = data[train_size:].to(self.device)
        
        dataset_info.update({
            'train_size': len(train_data),
            'val_size': len(val_data),
            'total_size': len(data)
        })
        
        return train_data, val_data, dataset_info
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excess whitespace
        text = ' '.join(text.split())
        
        # Replace unknown characters with spaces
        cleaned_chars = []
        for char in text:
            if char in self.char_to_idx:
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(' ')  # Replace unknown chars with space
                
        return ''.join(cleaned_chars)
        
    def _augment_text(self, text: str) -> str:
        """Apply simple text augmentation."""
        # Add some variations
        augmented = [text]
        
        # Add lowercase version
        augmented.append(text.lower())
        
        # Add uppercase version (partial)
        words = text.split()
        if len(words) > 0:
            # Capitalize first word of each sentence
            augmented.append('. '.join(s.capitalize() for s in text.split('. ')))
            
        return '\n'.join(augmented)
        
    def _analyze_dataset(self, text: str) -> Dict:
        """Analyze dataset characteristics."""
        # Character frequency
        char_freq = Counter(text)
        
        # Coverage of vocabulary
        unique_chars = set(text)
        vocab_coverage = len(unique_chars) / self.vocab_size
        
        # Text statistics
        lines = text.split('\n')
        words = text.split()
        
        return {
            'num_characters': len(text),
            'num_lines': len(lines),
            'num_words': len(words),
            'unique_characters': len(unique_chars),
            'vocab_coverage': vocab_coverage,
            'avg_line_length': len(text) / len(lines) if lines else 0,
            'most_common_chars': char_freq.most_common(10),
            'missing_vocab_chars': [c for c in unique_chars if c not in self.char_to_idx]
        }
        
    def create_custom_dataset(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        dataset_type: str = 'generation'
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Create a custom dataset from a list of texts.
        
        Args:
            texts: List of text samples
            labels: Optional labels for classification tasks
            dataset_type: Type of dataset ('generation', 'classification')
            
        Returns:
            train_data, val_data, dataset_info
        """
        if dataset_type == 'generation':
            # Concatenate all texts with delimiters
            combined_text = '\n\n'.join(texts)
            return self.prepare_dataset(combined_text)
            
        elif dataset_type == 'classification':
            # TODO: Implement classification dataset preparation
            raise NotImplementedError("Classification datasets not yet supported")
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    def save_dataset_info(self, info: Dict, save_path: str):
        """Save dataset information to JSON file."""
        # Convert non-serializable items
        if 'most_common_chars' in info:
            info['most_common_chars'] = [
                {'char': char, 'count': count} 
                for char, count in info['most_common_chars']
            ]
            
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
            
    def load_preprocessed_dataset(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Load a preprocessed dataset."""
        data = torch.load(path)
        return data['train'], data['val'], data['info']
        
    def save_preprocessed_dataset(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        info: Dict,
        path: str
    ):
        """Save preprocessed dataset."""
        torch.save({
            'train': train_data,
            'val': val_data,
            'info': info
        }, path)