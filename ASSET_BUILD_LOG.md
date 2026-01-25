# Asset Build Log

## 1. Input Corpus
- **Path**: `data/tokenizer_full_input_cleaned`
- **Total Lines**: 3.8 Million
- **Size**: ~1.6 GB
- **Normalization**: `PTBRNormalizer` (NFC, Mojibake fixed, Punctuation norm, Case preserved, Accents preserved).

## 2. Tokenizer Training
- **Engine**: **Native Rust** (`BPETrainer`).
- **Mode**: "Fire Mode" (Full Corpus | No Sampling | Rayon Parallelism).
- **Vocab Size**: 65,536 (u16).
- **Min Frequency**: 5.
- **Special Tokens**: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[SEP]`.
- **Merged Pairs**: 65,275.
- **Training Time**: ~9 hours (32476.2s).
- **Status**: ✅ **COMPLETE**. Saved to `data/tokenizer.json`.

## 3. Dataset Tokenization
- **Engine**: **Native Rust** (`Tokenize` command).
- **Format**: `train.bin` (u16 tokens, 8-byte u64 header).
- **Documents Processed**: 164 files.
- **Total Tokens**: **367,100,000** (367.1M).
- **Status**: ✅ **COMPLETE**.

## 4. Verification
- **Tool**: `scripts/verify_dataset.py`.
- **Status**: ✅ **VERIFIED**. Checks passed (Header, Tokens, Vocab).

## Next Steps
Run validation to confirm binary integrity:

```bash
python scripts/verify_dataset.py
```
