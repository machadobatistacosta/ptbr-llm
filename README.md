# 🇧🇷 PTBR-LLM

**PT-BR RWKV Large Language Model trained from scratch in Rust (Burn). Tokenizer + streaming data pipeline + checkpoints.**

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
| **140M** | 140 million | ~3GB | ~300MB | ⚙️ Ready |
| **400M** | 418 million | ~7GB | ~1GB | 🔄 Training |
| **800M** | 800 million | ~12GB | ~1.8GB | ⚙️ Ready |
| **1B** | 1 billion | ~16GB | ~2GB | ⚙️ Ready |
| **1.5B** | 1.5 billion | ~24GB | ~3GB | ⚙️ Ready |

> **Note:** RWKV inference memory is dramatically lower than training.

---

## 📚 Dataset (V17 "Balanced Soberania")

| Metric | Value |
|--------|-------|
| **Total Tokens** | ~545M |
| **Vocabulary** | 32,000 (BPE) |
| **Documents** | ~2.4M |
| **Sources** | Wikipedia, Wikisource, Wikibooks, Brazilian Laws, Literature |
| **Cleaning** | Custom V14 pipeline (dedup, encoding fixes, quality filters) |
| **License** | CC BY-SA / Public Domain |

---

## 🛠️ Usage

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 1. Train Tokenizer

```bash
cargo run --release -- train-tokenizer \
  --corpus data/corpus_clean \
  --output data/tokenizer \
  --vocab-size 32000
```

### 2. Build Binary Dataset

```bash
cargo run --release -- build-dataset \
  --tokenizer data/tokenizer/tokenizer.json \
  --output data/tokenizer/train.bin \
  --source data/corpus_clean \
  --clean
```

### 3. Train Model

**Local (CPU/CUDA):**

```bash
cargo run --release --features cuda -- train \
  --data data/tokenizer/train.bin \
  --val-data data/tokenizer/val.bin \
  --tokenizer data/tokenizer/tokenizer.json \
  --output checkpoints \
  --model-size 400m \
  --max-steps 50000 \
  --batch-size 4 \
  --grad-accum 4 \
  --seq-len 256
```

### 4. Inference

```bash
cargo run --release --features cuda -- generate \
  --model checkpoints/step_50000 \
  --tokenizer data/tokenizer/tokenizer.json \
  --prompt "O futuro da inteligência artificial no Brasil é" \
  --max-tokens 100 \
  --temperature 0.7 \
  --model-size 400m
```

---

## 🚀 Technical Details

### Architecture
- **RWKV-v6** (Receptance Weighted Key Value)
- **Token Shift**: Time-mixing token shift mechanism
- **Channel Mixing**: Position-wise feed-forward network
- **Head**: Standard linear projection (optional weight tying)

### Optimization
- **WKV Kernel**: Custom optimized parallel scan for WKV computation
- **Mixed Precision**: FP16 training support (via Burn backend)
- **Gradient Accumulation**: Support for large effective batch sizes on limited hardware

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
