# 🇧🇷 PTBR-LLM

**Portuguese Brazilian Language Model — trained from scratch in Rust.**

<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.75+-orange.svg" /></a>
  <a href="https://burn.dev/"><img src="https://img.shields.io/badge/burn-0.14-red.svg" /></a>
  <a href="https://arxiv.org/abs/2305.13048"><img src="https://img.shields.io/badge/arch-RWKV--v6-green.svg" /></a>
  <img src="https://img.shields.io/badge/inference-zero%20python-blueviolet.svg" />
</p>

---

## 🎯 Overview

**PTBR-LLM** is a family of language models based on the **RWKV-v6 architecture**, optimized for the Rust ecosystem. Unlike traditional Transformers, RWKV combines efficient parallel training with RNN-like inference — enabling **constant memory usage** regardless of sequence length.

### Why RWKV?

- ⚡ **O(1) inference memory** — no KV-cache explosion
- 🚀 **Linear complexity** — scales efficiently with context
- 🦀 **Pure Rust** — training and inference with [Burn](https://burn.dev/), no Python runtime
- 🔒 **Privacy-first** — runs locally, data never leaves your infrastructure

---

## 📊 Models

| Size | Parameters | VRAM (Train) | VRAM (Inference) | Status |
|------|------------|--------------|------------------|--------|
| **85M** | 85 million | ~2GB | ~200MB | ⚙️ Ready |
| **170M** | 170 million | ~4GB | ~400MB | ⚙️ Ready |
| **400M** | 418 million | ~7GB | ~1GB | 🔄 Training |
| **1B** | 1 billion | ~16GB | ~2GB | ⚙️ Ready |
| **1.5B** | 1.5 billion | ~24GB | ~3GB | ⚙️ Ready |

> **Note:** RWKV inference memory is dramatically lower than training. 1B+ models require A100 or multi-GPU setup.

---

## 📚 Dataset (V17)

| Metric | Value |
|--------|-------|
| **Total Tokens** | 378M |
| **Vocabulary** | 32,000 (BPE) |
| **Documents** | 2.4M |
| **Sources** | Wikipedia, Wikisource, Wikibooks, Brazilian Laws |
| **Cleaning** | V14 pipeline (code removal, dedup, encoding fixes) |
| **License** | CC BY-SA / Public Domain |

---

## 🛠️ Full Pipeline

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# For GPU training (optional)
# Ensure CUDA toolkit is installed
```

### 1. Train Tokenizer

```bash
cargo run --release -- train-tokenizer \
  --corpus data/corpus.txt \
  --output data/tokenizer \
  --vocab-size 32000
```

### 2. Build Dataset

```bash
cargo run --release -- build-dataset \
  --tokenizer data/tokenizer/tokenizer.json \
  --output data/train.bin \
  --source data/tokenizer_full_input_cleaned \
  --clean
```

### 3. Train Model

**Local (85M):**

```bash
cargo run --release -- train \
  --data data/train.bin \
  --tokenizer data/tokenizer/tokenizer.json \
  --output checkpoints \
  --model-size 85m \
  --max-steps 10000 \
  --batch-size 2 \
  --grad-accum 8 \
  --seq-len 256
```

**Kaggle GPU (400M):**

```bash
./target/release/ptbr-llm train \
  --data /kaggle/input/dataset/train.bin \
  --tokenizer /kaggle/input/dataset/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size 400m \
  --max-steps 5000 \
  --batch-size 4 \
  --grad-accum 4 \
  --seq-len 256 \
  --eval-every 99999
```

### 4. Generate Text

```bash
cargo run --release -- generate \
  --model checkpoints/step_5000.mpk \
  --tokenizer data/tokenizer/tokenizer.json \
  --prompt "O Brasil é um país" \
  --max-tokens 50 \
  --temperature 0.7 \
  --model-size 400m
```

---

## 🚀 Kaggle Quick Start

```python
# Install Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
import os
os.environ['PATH'] += ":/root/.cargo/bin"

# Clone & Build
%cd /kaggle/working/
!git clone https://github.com/machadobatistacosta/ptbr-llm.git
%cd ptbr-llm
!/root/.cargo/bin/cargo build --release --features cuda
```

---

## 🗺️ Roadmap

- [x] RWKV-v6 implementation in Burn
- [x] BPE tokenizer (32K vocab)
- [x] Dataset pipeline (378M tokens)
- [ ] Train 400M model to convergence
- [ ] Publish trained weights
- [ ] Instruction fine-tuning (Chat)
- [ ] 4-bit quantization
- [ ] ONNX export
- [ ] WASM inference (browser)

---

## 🏗️ Project Structure

```
ptbr-llm/
├── src/
│   ├── main.rs              # CLI
│   ├── model/               # RWKV implementation
│   ├── data/                # Dataset & DataLoader
│   └── tokenizer/           # BPE Tokenizer
├── scripts/                 # Data processing
│   └── clean_tokenizer_input.py
└── data/
    ├── tokenizer/           # tokenizer.json & train.bin
    └── tokenizer_full_input_cleaned/
```

---

## ⚙️ Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-size` | 85m, 170m, 400m | 85m |
| `--max-steps` | Training steps | 50000 |
| `--batch-size` | Batch size | 4 |
| `--grad-accum` | Gradient accumulation | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--warmup-steps` | Warmup steps | 300 |
| `--seq-len` | Sequence length | 256 |
| `--save-every` | Checkpoint interval | 1000 |

---

##  License

[Apache License 2.0](LICENSE)

---

## 👤 Author

**Caike Machado Batista Costa**

<p align="left">
  <a href="https://www.linkedin.com/in/caike-machado-batista-costa/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" height="24" />
  </a>
  <a href="https://github.com/machadobatistacosta" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-111111?style=for-the-badge&logo=github&logoColor=white" height="24" />
  </a>
</p>

---

## 🔗 References

- [Burn Framework](https://burn.dev/)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
