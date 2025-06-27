"""
PCN Language Model - Predictive Coding Networks for Text Generation

A novel application of biologically-inspired predictive coding to language modeling.
"""

from .pcn_lm import PCNLanguageModel
from .tokenizer import CharacterTokenizer
from .dataset import TextDataset

__version__ = "0.1.0"
__all__ = ["PCNLanguageModel", "CharacterTokenizer", "TextDataset"]