# 🇧🇷 PTBR-LLM

**Small Language Model para Português Brasileiro, treinado do zero em Rust.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Burn](https://img.shields.io/badge/burn-0.14-red.svg)](https://burn.dev/)

---

## 🎯 Sobre

Modelo de linguagem baseado na arquitetura **RWKV**, otimizado para:

- ✅ Rodar em máquinas com **8GB RAM** (CPU)
- ✅ Treinar do zero com pipeline completo
- ✅ Processar texto em **Português Brasileiro**
- ✅ Fontes 100% públicas e licenciadas

### Modelos

| Config | Parâmetros | RAM | Status |
|--------|------------|-----|--------|
| **micro** | 10M | ~2GB | ✅ Treinado |
| **mini** | 20M | ~4GB | ✅ Treinado |
| **85m** | 85M | ~8GB | 🔄 Em treino |

---

## 🚀 Quick Start

```bash
# Compilar
cargo build --release

# Treinar (exemplo mini, 20k steps)
./target/release/ptbr-llm train \
  --data data/tokenized \
  --tokenizer data/tokenizer/tokenizer.json \
  --output checkpoints \
  --model-size mini \
  --max-steps 20000 \
  --batch-size 4 \
  --save-every 5000

# Gerar texto
./target/release/ptbr-llm generate \
  --model checkpoints/checkpoint_20000.bin \
  --tokenizer data/tokenizer/tokenizer.json \
  --prompt "O Brasil é" \
  --max-tokens 50 \
  --model-size mini

---

## 📁 Estrutura

```
ptbr-llm/
├── src/
│   ├── main.rs           # CLI
│   ├── model/            # RWKV + Trainer
│   ├── data/             # Dataset + Parser
│   └── tokenizer/        # BPE
├── scripts/              # Pipelines de dados
├── Cargo.toml
└── ARQUITETURA.md        # Documentação técnica completa
```

---

## 🔧 Flags de Treinamento

| Flag | Descrição | Default |
|------|-----------|---------|
| `--model-size` | micro, mini, 85m | mini |
| `--max-steps` | Total de steps | 50000 |
| `--batch-size` | Batch size | 2 |
| `--grad-accum` | Gradient accumulation | 16 |
| `--learning-rate` | Learning rate | 3e-4 |
| `--warmup-steps` | Warmup steps | 500 |
| `--save-every` | Checkpoint interval | 2500 |
| `--seq-len` | Sequence length | 256 |

---

## 📚 Fontes de Dados

Todas públicas e com licença compatível:

- **Wikipedia PT-BR** (CC BY-SA)
- **Wikisource** (domínio público)
- **Wikibooks** (CC BY-SA)
- **Legislação brasileira** (domínio público)

> ⚠️ Dados não incluídos no repositório. Veja `scripts/` para pipelines.

---

## 🏋️ Treinar no Kaggle

1. Suba `tokenizer.json` + `train.bin` como Dataset
2. Clone este repo no notebook
3. Execute:

```bash
./target/release/ptbr-llm train \
  --data /kaggle/input/seu-dataset \
  --tokenizer /kaggle/input/seu-dataset/tokenizer.json \
  --output /kaggle/working/checkpoints \
  --model-size mini \
  --max-steps 20000
```

---

## 📖 Documentação

- **[ARQUITETURA.md](ARQUITETURA.md)** - Documentação técnica completa
- **[scripts/](scripts/)** - Pipelines de processamento

---

## 📄 Licença

[Apache License 2.0](LICENSE)

---

## 👤 Autor

**Caike Machado Batista Costa**

---

## 🔗 Links

- [Burn Framework](https://burn.dev/)
- [RWKV Paper](https://arxiv.org/abs/2305.13048)
```

---
