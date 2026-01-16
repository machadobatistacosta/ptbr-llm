# PTBR-SLM - Arquitetura v2 (Tokenizer Dinâmico)

**Data: 2026-01-15**  
**Atualização: Tokenizer Dinâmico ChatML-Ready**  
**Framework: Burn 0.14 + Rust 2021**  

---

## 🎯 Mudanças Principais (v2.0)

### ✨ Tokenizer Dinâmico

O `BPETrainer` agora suporta **special tokens personalizados** ao invés de hardcoded:

```rust
// v1: Hardcoded
pub struct BPETrainer { vocab_size, min_frequency }

// v2: Dinâmico
pub struct BPETrainer { 
    vocab_size, 
    min_frequency, 
    special_tokens: Vec<String>  ← NOVO!
}
```

#### Uso:
```bash
# Padrão (backward compatible)
./target/release/ptbr-slm train-tokenizer --corpus data/ --output out/

# ChatML
./target/release/ptbr-slm train-tokenizer --corpus data/ --output out/ \
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[SEP],<|im_start|>,<|im_end|>"

# Jurídico/RAG
./target/release/ptbr-slm train-tokenizer --corpus data/ --output out/ \
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[SEP],<|DOC|>,<|ARTIGO|>,<|LEI|>"
```

---

## 📦 Que Existe Agora

### Código Rust (24 arquivos = 5,064 linhas)

| Arquivo | Linhas | Função |
|---------|--------|--------|
| main.rs | 1,732 | CLI + dispatcher |
| bpe.rs | 387 | Tokenizer BPE |
| adapters.rs | 342 | LoRA adapters |
| rwkv.rs | 337 | Modelo RWKV |
| validation.rs | 282 | Validation callbacks |
| cleaner.rs | 281 | Data cleaning (regex) |
| metrics.rs | 264 | Training metrics |
| trainer.rs | 213 | Training loop |
| dataset.rs | 191 | Data loading |
| normalize.rs | 163 | PT-BR text normalization |
| config.rs | 137 | Model config |
| wiki_parser.rs | 133 | Wikipedia parser |
| wkv_optimized.rs | 132 | WKV optimization |
| checkpoint.rs | 93 | Checkpointing |
| evaluator.rs | 93 | Evaluation metrics |
| lr_finder.rs | 84 | Learning rate finder |
| format.rs | 81 | Formatting utils |
| precision.rs | 57 | Mixed precision |
| lib.rs | 37 | Library exports |
| 5 × mod.rs | 44 | Module organization |

### Dados em `data/` (7.3 GB total)

**Diretórios:**
- `tokenizer_v16_full/` - Tokenizer treinado (JSON + vocab)
- `tokenized_v16_full/` - Dados tokenizados (1 arquivo: train.bin)
- `wiki_clean/` - Wikipedia limpa (2.3 GB)
- `tokenizer_full_input/` - Input original (2.6 GB)
- `wikibooks_clean/` - WikiBooks (31 MB)
- `wikisource_clean/` - WikiSource (187 MB)
- `wikinews_clean/` - WikiNews (60 MB)
- `planalto_clean/` - Legislação brasileira (4 MB)

**Arquivos:**
- **328 .txt** (corpus) = 5.3 GB
- **1 .bin** (train.bin) = 1.2 GB
- **1 .json** (tokenizer.json) = 3.88 MB
- **1 .zip** (787 MB) - não especificado

### Build Status

```
cargo build --release
→ Exit Code: 1 (FALHA)

cargo build --release --features cuda
→ Exit Code: 0 (SUCESSO)
```

**Problema:** Falha com feature default vazio

---

## 📊 Estrutura Real do Código

### main.rs (1,732 linhas)
CLI com `clap` + dispatcher para 8 subcomandos

### Tokenizer (bpe.rs 387 + normalize.rs 163 = 550 linhas)

```
BPETokenizer
  └─ vocab: BPEVocab
  └─ encoder: HashMap
  └─ decoder: HashMap
  └─ cache: LRUCache (100k)

PTBRNormalizer
  └─ NFD decomposition
  └─ Lowercase
  └─ Whitespace normalization
```

### Model (rwkv.rs 337 linhas)

```
RWKV
  └─ embedding: Embedding
  └─ ln_pre: LayerNorm
  └─ blocks: Vec<RWKVBlock>
  └─ ln_out: LayerNorm
  └─ head: Linear

RWKVBlock
  ├─ time_mixing: TimeMixing (O(n))
  └─ channel_mixing: ChannelMixing (FFN)

RWKVState (para geração incremental)
  ├─ time_state
  └─ channel_state
```

### Training (trainer.rs 213 linhas)

```
Trainer
  ├─ model: RWKV
  ├─ optimizer: AdamW
  ├─ config: TrainingConfig
  ├─ step: usize
  ├─ micro_step: usize
  ├─ accumulated_loss: f32
  └─ ema_loss: f32

Features:
  ✓ Gradient accumulation
  ✓ LR schedule (cosine + warmup)
  ✓ NaN/Inf detection
```

### Data (dataset.rs 191 + wiki_parser.rs 133 + cleaner.rs 281 = 605)

```
MmapDataset (memory-mapped)
  └─ carrega .bin sem descomprimir

WikiStreamParser
  └─ parse BZ2 streaming

WikiCleaner
  ├─ Remove templates {{ }}
  ├─ Remove HTML tags
  ├─ Remove wiki links [[]]
  ├─ Remove categories
  ├─ Remove images
  └─ + 10 regex patterns
```

---

## 🔧 Dependências Reais

```toml
[dependencies]
burn = "0.14"
bincode = "2.0.0-rc.3"
serde = "1.0"
serde_json = "1.0"
memmap2 = "0.9"
quick-xml = "0.31"
bzip2 = "0.4"
regex = "1.10"
unicode-normalization = "0.1"
rayon = "1.10"
clap = "4.5"
tracing = "0.1"
```

---

## 📋 Status Real

✅ **Compilado:** Rust code compila (com --features cuda)  
✅ **Dados:** 328 .txt + 1 tokenizer + 1 train.bin  
❌ **Build padrão:** Falha (exit code 1)  
❌ **Checkpoints:** 0 salvos  
❌ **Modelos treinados:** Nenhum  

---

## 📁 Localização de Tudo

```
c:\Users\caike\Desktop\ptbr-slm\
├── src/
│   ├── main.rs (1,732 linhas)
│   ├── model/
│   │   ├── rwkv.rs
│   │   ├── trainer.rs
│   │   ├── config.rs
│   │   ├── checkpoint.rs
│   │   ├── adapters.rs
│   │   ├── evaluator.rs
│   │   ├── validation.rs
│   │   ├── lr_finder.rs
│   │   ├── precision.rs
│   │   ├── wkv_optimized.rs
│   │   └── mod.rs
│   ├── data/
│   │   ├── dataset.rs
│   │   ├── wiki_parser.rs
│   │   ├── cleaner.rs
│   │   └── mod.rs
│   ├── tokenizer/
│   │   ├── bpe.rs
│   │   ├── normalize.rs
│   │   └── mod.rs
│   ├── logger/
│   │   ├── metrics.rs
│   │   └── mod.rs
│   ├── utils/
│   │   ├── format.rs
│   │   └── mod.rs
│   └── lib.rs
├── data/
│   ├── tokenizer_v16_full/ → tokenizer.json (3.88 MB)
│   ├── tokenized_v16_full/ → train.bin (1.2 GB)
│   ├── wiki_clean/ (2.3 GB)
│   ├── tokenizer_full_input/ (2.6 GB)
│   ├── wikibooks_clean/ (31 MB)
│   ├── wikisource_clean/ (187 MB)
│   ├── wikinews_clean/ (60 MB)
│   └── planalto_clean/ (4 MB)
├── Cargo.toml (52 linhas)
├── README.md
├── ARQUITETURA.md (este arquivo)
└── corpus.txt
```

---

## 🔴 Problemas Conhecidos

1. **Build falha:** `cargo build --release` retorna exit code 1
   - MAS: `cargo build --release --features cuda` funciona
   - Indica: Problema com feature "default" ou conflito de dependência

2. **Nenhum checkpoint:** 0 modelos treinados salvos
   - diretório `checkpoints/` não existe

3. **Encoding README:** UTF-8 corrompido em alguns caracteres

---

## ✅ Que Funciona

- ✅ Parser de Wikipedia
- ✅ Limpeza de dados (15+ regex patterns)
- ✅ Tokenizer BPE
- ✅ Normalização PT-BR
- ✅ Arquitetura RWKV (pronta para treinar)
- ✅ CLI com 8 subcomandos (estrutura)
- ✅ Memory-mapped dataset loading
- ✅ Compilação com CUDA features

---

**Total verificado:** 24 arquivos .rs, 7.3 GB dados, 1 tokenizer, 0 checkpoints**
