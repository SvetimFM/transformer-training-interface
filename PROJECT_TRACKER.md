# Project Tracker: Dataset & Tokenizer Flexibility

## Main Goals
1. **Custom Dataset Support**
   - Allow users to upload/use any text file beyond Tiny Shakespeare
   - Support URL downloads and local file uploads
   - Add dataset validation and size limits (50MB default)
   
2. **Tokenizer Flexibility**
   - Easy switching between character-level and BPE tokenizers
   - Support common tokenizers (GPT-2, tiktoken)
   - Allow custom BPE training on user datasets

## Current Status (Started: 2024-01-07)

### Completed ✅
- [x] UI reorganization with experiments dropdown
- [x] Added scientific explainer for transformers
- [x] Added comprehensive tooltips
- [x] Added Shakespeare training recipes
- [x] Fixed parameter calculation accuracy
- [x] Committed initial UI improvements
- [x] Created tokenizer abstraction interface (BaseTokenizer)
- [x] Implemented CharacterTokenizer class
- [x] Implemented BPETokenizer wrapper (with tiktoken support)
- [x] Added DatasetConfig to configuration system
- [x] Created DatasetManager for flexible dataset loading
- [x] Updated dataset_preparation.py for backward compatibility

### In Progress 🚧
- [ ] Adding dataset selection UI panel
- [ ] Update model to handle variable vocab sizes
- [ ] Testing tokenizer integration

### Todo 📋
- [ ] Create dataset upload endpoints in app.py
- [ ] Add file upload with progress tracking
- [ ] Add dataset preview functionality
- [ ] Test with various text formats
- [ ] Add tokenizer caching for performance
- [ ] Handle Unicode properly in all tokenizers
- [ ] Add sample datasets (code, chat logs, etc.)

## Technical Decisions
- Keep character-level as default (simplest, works with any text)
- Use tiktoken for BPE implementation (well-tested, efficient)
- Store uploaded datasets in `.datasets/` folder
- Limit file uploads to 50MB by default
- Support only .txt files initially

## UI Flow
1. User selects "Dataset & Tokenization" panel
2. Choose dataset source:
   - Tiny Shakespeare (default)
   - Upload custom text file
   - Load from URL
3. Choose tokenizer:
   - Character-level (default)
   - BPE (GPT-2)
   - BPE (Custom trained)
4. Display dataset stats:
   - File size
   - Character count
   - Vocab size
   - Estimated tokens

## Notes
- Character tokenizer must handle any Unicode text
- BPE tokenizer needs vocab size limits (e.g., 50k tokens)
- Consider adding dataset preview (first 1000 chars)
- Need to update model architecture when vocab size changes
- Consider caching tokenized datasets for faster loading

## Recent Discoveries
- Original MultiHeadAttention implementation was non-standard (separate heads)
- Created StandardMultiHeadAttention that matches GPT-2/3 implementation
- Fixed parameter calculation:
  - Attention: Always 4 × n_embed × n_embed (Q,K,V,O projections)
  - LayerNorm: 2 × n_embed per norm (gamma and beta)
  - Each transformer block has 2 LayerNorms (before attention and FFN)
- The number of heads doesn't affect parameter count in standard implementation