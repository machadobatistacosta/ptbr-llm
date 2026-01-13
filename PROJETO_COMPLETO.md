# 🇧🇷 PTBR-SLM - Contexto Completo do Projeto

**Data:** 2026-01-11  
**Linguagem:** Rust 2021  
**Framework:** Burn 0.14 (Deep Learning)  
**Arquitetura:** RWKV (Recurrent Weight Key Value)  
**Status:** Em desenvolvimento

---

## 📋 Sumário Executivo

Modelo de linguagem **Small Language Model (SLM)** para Português Brasileiro treinado do zero em Rust. Arquitetura RWKV otimizada para:
- Eficiência de memória (8GB RAM em CPU)
- Treinamento completo com dados públicos
- Inference rápida e portável

**Tamanhos disponíveis:** 85M, 400M, 800M, 1B, 1.5B parâmetros

---

## 🏗️ ARQUITETURA DO PROJETO

```
ptbr-slm/
├── src/                         # Código Rust principal
│   ├── main.rs                 # CLI (1,732 linhas, 15 subcomandos)
│   ├── lib.rs                  # Exports públicos
│   ├── logger/                 # Sistema de logging
│   ├── utils/                  # Utilitários (format, etc)
│   │
│   ├── model/                  # RWKV + Treinamento
│   │   ├── rwkv.rs            # RWKV modelo (405 linhas)
│   │   ├── trainer.rs         # Training loop (213 linhas)
│   │   ├── config.rs          # Configurações (137 linhas)
│   │   ├── checkpoint.rs      # Gradient checkpointing (93 linhas)
│   │   ├── adapters.rs        # LoRA adapters (342 linhas)
│   │   ├── evaluator.rs       # Eval metrics (93 linhas)
│   │   ├── validation.rs      # Validation callbacks (282 linhas)
│   │   ├── lr_finder.rs       # Learning rate finder (84 linhas)
│   │   ├── precision.rs       # Mixed precision (57 linhas)
│   │   ├── wkv_optimized.rs   # WKV otimizado (132 linhas)
│   │   └── mod.rs             # Exports
│   │
│   ├── tokenizer/              # BPE Tokenizer
│   │   ├── bpe.rs             # BPE tokenizer (454 linhas)
│   │   ├── normalize.rs       # PT-BR normalization (163 linhas)
│   │   └── mod.rs             # Exports
│   │
│   ├── data/                    # Data loading e processamento
│   │   ├── dataset.rs         # MmapDataset (191 linhas)
│   │   ├── wiki_parser.rs     # Wikipedia XML.BZ2 parser (133 linhas)
│   │   ├── cleaner.rs         # Data cleaning/regex (281 linhas)
│   │   └── mod.rs             # Exports
│   │
│   └── [5 módulos com mod.rs]
│
├── data/                        # Dados (7.3 GB)
│   ├── tokenizer_v16_full/     # Vocab + merges (JSON)
│   ├── tokenized_v16_full/     # Dados tokenizados (train.bin 1.2GB)
│   ├── wiki_clean/             # Wikipedia limpa (2.3 GB)
│   ├── tokenizer_full_input/   # Input original (2.6 GB, 328 arquivos)
│   ├── wikibooks_clean/        # WikiBooks (31 MB)
│   ├── wikisource_clean/       # WikiSource (187 MB)
│   ├── wikinews_clean/         # WikiNews (60 MB)
│   └── planalto_clean/         # Legislação PT-BR (4 MB, 15 arquivos)
│
├── configs/                     # Configurações modelo
│   └── model_85m.toml
│
├── scripts/                     # Pipelines de dados (Python)
│   ├── build_corpus_v15_stream.py
│   ├── build_v16.py
│   ├── clean_sources_v15.py
│   └── [12+ outros scripts]
│
├── Cargo.toml                   # Manifest Rust
├── ARQUITETURA.md               # Documentação técnica
├── README.md                    # Getting started
└── LICENSE                      # Apache 2.0
```

---

## 🏛️ ESTRUTURAS/CLASSES PRINCIPAIS (Rust)

### 1️⃣ MODELO RWKV (`src/model/rwkv.rs`)

#### `struct RWKV<B: Backend>`
Modelo principal RWKV com N layers.

**Campos:**
```rust
pub struct RWKV<B: Backend> {
    embedding: Embedding<B>,           // Token embeddings
    ln_pre: LayerNorm<B>,              // Pre-norm
    blocks: Vec<RWKVBlock<B>>,         // N RWKVBlocks
    ln_out: LayerNorm<B>,              // Output norm
    head: Linear<B>,                   // Vocab projection
    vocab_size: usize,
    d_model: usize,
    n_layers: usize,
}
```

**Métodos principais:**
- `new(config, device)` - Cria modelo novo
- `forward(input_ids) -> Tensor[batch, seq_len, vocab]` - Forward pass
- `forward_inference(input_ids) -> Tensor[batch, vocab]` - Retorna logits último token

#### `struct RWKVBlock<B: Backend>`
Um bloco do modelo (pre-norm + residual).

**Campos:**
```rust
pub struct RWKVBlock<B: Backend> {
    ln1: LayerNorm<B>,              // Pre-norm time mixing
    time_mixing: TimeMixing<B>,     // O(n) attention
    ln2: LayerNorm<B>,              // Pre-norm channel
    channel_mixing: ChannelMixing<B>, // FFN
    dropout: Dropout,
}
```

**Forward:**
```
x -> LN -> TimeMixing + x (residual)
  -> LN -> ChannelMixing + x (residual)
```

#### `struct TimeMixing<B: Backend>`
RNN-like attention com complexidade O(n).

**Campos:**
```rust
pub struct TimeMixing<B: Backend> {
    receptance: Linear<B>,      // Gate
    key: Linear<B>,             // Key projection
    value: Linear<B>,           // Value projection
    output: Linear<B>,          // Output projection
    
    time_decay: Param<Tensor<B, 1>>,    // Decay rate (layer-dependent)
    time_first: Param<Tensor<B, 1>>,    // Bonus primeiro token
    time_mix_k: Param<Tensor<B, 1>>,    // Mix ratio para K
    time_mix_v: Param<Tensor<B, 1>>,    // Mix ratio para V
    time_mix_r: Param<Tensor<B, 1>>,    // Mix ratio para R (gate)
    d_model: usize,
}
```

**Algoritmo WKV (Weighted Key-Value):**
- Usa WKV kernel otimizado com log-sum-exp trick
- Processa em chunks para estabilidade
- Estado: $(a, b, p)$ mantido por layer e batch

#### `struct ChannelMixing<B: Backend>`
FFN com squared ReLU.

**Campos:**
```rust
pub struct ChannelMixing<B: Backend> {
    receptance: Linear<B>,          // Gate
    key: Linear<B>,                 // Projeção FFN
    value: Linear<B>,               // Projeção saída
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    d_model: usize,
}
```

#### `struct RWKVState<B: Backend>`
Estado para inference incremental (cache).

```rust
pub struct RWKVState<B: Backend> {
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    pub channel_state: Vec<Tensor<B, 2>>,
}
```

---

### 2️⃣ CONFIGURAÇÃO (`src/model/config.rs`)

#### `struct RWKVConfig`
Configuração estática do modelo.

```rust
pub struct RWKVConfig {
    pub vocab_size: usize,          // 32k
    pub d_model: usize,             // Hidden dim (768-2304)
    pub n_layers: usize,            // 12-28
    pub d_ffn: usize,               // FFN dim
    pub max_seq_len: usize,         // Max context (512-1024)
    pub dropout: f64,
    pub layer_norm_eps: f64,
}
```

**Presets:**
- `ptbr_85m()` - d_model=768, 12 layers → ~85M params
- `ptbr_400m()` - d_model=1024, 24 layers → ~400M params
- `ptbr_800m()` - d_model=1536, 24 layers → ~800M params
- `ptbr_1b()` - d_model=2048, 24 layers → ~1B params
- `ptbr_1_5b()` - d_model=2304, 28 layers → ~1.5B params

**Métodos:**
- `num_parameters()` - Calcula total de parâmetros
- `estimated_vram(batch_size, seq_len)` - VRAM em bytes

#### `struct TrainingConfig`
Configuração de treinamento.

```rust
pub struct TrainingConfig {
    pub learning_rate: f64,                 // 3e-4
    pub batch_size: usize,                  // Batch size efetivo
    pub gradient_accumulation_steps: usize, // 16
    pub warmup_steps: usize,                // 500
    pub max_steps: usize,                   // Total steps
    pub weight_decay: f64,                  // L2 regularization
    pub gradient_clip: f64,                 // Gradient clipping
    pub save_every: usize,                  // Checkpoint interval
    pub log_every: usize,                   // Log interval
    pub min_lr_ratio: f64,                  // Min LR em cosine decay
}
```

---

### 3️⃣ TREINAMENTO (`src/model/trainer.rs`)

#### `struct Trainer<B: AutodiffBackend>`
Training loop com gradient accumulation e LR scheduling.

**Campos:**
```rust
pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,
    config: TrainingConfig,
    model_config: RWKVConfig,
    
    // Estado
    step: usize,
    micro_step: usize,
    accumulated_loss: f32,
    
    // Métricas
    last_grad_norm: f32,
    ema_loss: f32,
    best_loss: f32,
    device: B::Device,
}
```

#### `struct TrainStats`
Estatísticas de um step.

```rust
pub struct TrainStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f64,
    pub tokens_per_sec: f32,
}
```

**Métodos principais:**
- `new(model_config, train_config, device)` - Cria trainer
- `train_step(input_ids, target_ids) -> Option<TrainStats>` - Micro-step (com grad accumulation)
- `get_learning_rate() -> f64` - Warmup linear + cosine decay
- `save_checkpoint(path)` - Salva modelo + metadados
- `load_checkpoint(path)` - Carrega checkpoint

**Loss:** Cross-entropy com log-softmax para estabilidade

---

### 4️⃣ DATASET (`src/data/dataset.rs`)

#### `struct MmapDataset`
Memory-mapped dataset de tokens tokenizados (u16 little-endian).

**Campos:**
```rust
pub struct MmapDataset {
    data: Mmap,                 // File memory-mapped
    indices: Vec<usize>,        // Sequência offsets
    seq_len: usize,             // Context length
    epoch: usize,               // Epoch counter
    num_tokens: usize,          // Total tokens
}
```

**Métodos:**
- `from_file(path, seq_len)` - Carrega .bin
- `get(idx)` -> (Vec<u16>, Vec<u16>) - Retorna (input, target)
- `shuffle(base_seed)` - Shuffla com ChaCha8RNG
- `next_epoch()` - Incrementa epoch

#### `struct DataLoader<'a>`
Iterator para batches.

```rust
pub struct DataLoader<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_idx: usize,
}
```

**Iterator:** Retorna `(Vec<Vec<u16>>, Vec<Vec<u16>>)` - (batch inputs, batch targets)

#### `struct TokenizedDatasetWriter`
Escreve tokens em .bin.

```rust
pub struct TokenizedDatasetWriter {
    writer: BufWriter<File>,
    tokens_written: usize,
}
```

**Métodos:**
- `new(path)` - Cria arquivo
- `write_tokens(&[u16])` - Escreve tokens
- `finish()` -> usize - Retorna total escrito

---

### 5️⃣ TOKENIZER (`src/tokenizer/bpe.rs`)

#### `struct BPETokenizer`
Tokenizer BPE thread-safe com LRU cache.

**Campos:**
```rust
pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    special_tokens: HashMap<String, u16>,
    cache: Arc<RwLock<LRUCache>>,  // 100k cache
}
```

**Tokens especiais:**
```rust
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
```

**Métodos:**
- `from_file(path)` - Carrega tokenizer JSON
- `from_vocab(vocab)` - Cria de BPEVocab
- `encode(&str)` -> Vec<u16> - Tokeniza texto
- `encode_batch(&[String])` -> Vec<Vec<u16>> - Paralelo com Rayon
- `decode(&[u16])` -> String - Detokeniza
- `vocab_size()` -> usize

#### `struct BPEVocab`
Vocab serializado (Serde JSON).

```rust
pub struct BPEVocab {
    pub id_to_token: Vec<Vec<u8>>,
    pub merges: Vec<(u16, u16)>,
    pub special_tokens: HashMap<String, u16>,
}
```

#### `struct PTBRNormalizer`
Normalizador para PT-BR (`src/tokenizer/normalize.rs`).

**Features:**
- NFD decomposição Unicode
- Lowercase
- Whitespace normalization
- Remoção diacríticos opcionais

---

### 6️⃣ DATA PROCESSING

#### `struct WikiCleaner` (`src/data/cleaner.rs`)
Limpeza de corpus com regex.

**Operações:**
- Remove templates `{{ }}`
- Remove HTML tags `< >`
- Remove wiki links `[[ ]]`
- Remove categorias
- Remove imagens
- 10+ regex patterns

#### `struct WikiStreamParser` (`src/data/wiki_parser.rs`)
Parser de Wikipedia XML.BZ2 streaming.

**Features:**
- Suporta streaming (não carrega tudo em memória)
- BZ2 decompression
- XML parsing com quick-xml
- Extrai texto puro

---

### 7️⃣ CHECKPOINTING (`src/model/checkpoint.rs`)

#### `struct CheckpointedActivation<B: Backend>`
Cache de activations para gradient checkpointing.

```rust
pub struct CheckpointedActivation<B: Backend> {
    recompute_fn: Box<dyn Fn() -> Tensor<B, 3>>,
    cached: RefCell<Option<Tensor<B, 3>>>,
}
```

#### `struct CheckpointConfig`
Configuração de checkpointing (economizar memória).

```rust
pub struct CheckpointConfig {
    pub enabled: bool,
    pub every_n_layers: usize,
    pub checkpoint_layers: Option<Vec<usize>>,
}
```

**Presets:**
- `aggressive()` - Checkpoint toda layer
- `balanced()` - Checkpoint a cada 2 layers

---

## 🔗 FLUXO DE DADOS

### Treinamento

```
Arquivo .bin (tokens u16)
    ↓
MmapDataset.get(idx) → (input[seq_len], target[seq_len])
    ↓
DataLoader → Batches[(batch_size, seq_len), (batch_size, seq_len)]
    ↓
Trainer.train_step():
    ├─ Forward: input_ids → RWKV → logits[B, S, V]
    ├─ Loss: CrossEntropy(logits, targets)
    ├─ Backward: compute_gradients()
    ├─ Micro-step: acumula gradientes
    └─ Step (a cada grad_accum steps): optimizer.step()
    ↓
Salva checkpoint a cada save_every steps
```

### Inference

```
Texto (String)
    ↓
BPETokenizer.encode() → input_ids[S]
    ↓
RWKV.forward_inference() → logits[V]
    ↓
Sample from distribution → next_token
    ↓
Repeat (token autoregressivo)
```

---

## 📊 DADOS DISPONÍVEIS

### Corpus (7.3 GB total)

| Fonte | Localização | Tamanho | Arquivos | Licença |
|-------|-------------|---------|----------|---------|
| **Wikipedia PT-BR** | `data/wiki_clean/` | 2.3 GB | 328 .txt | CC BY-SA |
| **Tokenizer Input** | `data/tokenizer_full_input/` | 2.6 GB | 328 .txt | Variado |
| **WikiBooks** | `data/wikibooks_clean/` | 31 MB | ? | CC BY-SA |
| **WikiSource** | `data/wikisource_clean/` | 187 MB | ? | Domínio público |
| **WikiNews** | `data/wikinews_clean/` | 60 MB | ? | CC BY-SA |
| **Legislação Planalto** | `data/planalto_clean/` | 4 MB | 15 .txt | Domínio público |
| **TOKENIZED** | `data/tokenized_v16_full/` | 1.2 GB | 1 .bin | - |

### Tokenizer

| Arquivo | Local | Tamanho | Tipo |
|---------|-------|---------|------|
| tokenizer.json | `data/tokenizer_v16_full/` | 3.88 MB | BPE (JSON) |
| model_85m.toml | `configs/` | ~1 KB | Config |

### Legislação Brasileira (Clean)

```
data/planalto_clean/
├── CDC.txt                        # Código de Defesa do Consumidor
├── CLT.txt                        # Consolidação Leis Trabalho
├── CODIGO_CIVIL.txt              # Código Civil
├── CODIGO_PENAL.txt              # Código Penal
├── CONSTITUICAO_FEDERAL.txt      # Constituição Federal
├── CPC.txt                        # Código Processo Civil
├── CPP.txt                        # Código Processo Penal
├── CTN.txt                        # Código Tributário Nacional
├── LEI_ANTICORRUPCAO.txt         # Lei Anticorrupção
├── LEI_FALENCIAS.txt             # Lei de Falências
├── LEI_INQUILINATO.txt           # Lei do Inquilinato
├── LEI_LICITACOES_1993.txt       # Lei de Licitações 1993
├── LGPD.txt                       # Lei Geral Proteção Dados
├── MARCO_CIVIL_INTERNET.txt      # Marco Civil da Internet
└── NOVA_LEI_LICITACOES.txt       # Nova Lei Licitações
```

---

## 🛠️ DEPENDÊNCIAS (Cargo.toml)

### Principais

| Crate | Versão | Função |
|-------|--------|--------|
| **burn** | 0.14 | Deep learning framework |
| **ndarray** | (via burn) | CPU backend |
| **cuda-jit** | (via burn) | GPU backend (NVIDIA) |
| **wgpu** | (via burn) | GPU backend (vulkan, metal, dx12) |
| **serde/serde_json** | 1.0 | Serialização JSON |
| **memmap2** | 0.9 | Memory-mapped files |
| **bzip2** | 0.4 | Decompressão BZ2 |
| **regex** | 1.10 | Pattern matching |
| **unicode-normalization** | 0.1 | Normalizador Unicode |
| **rayon** | 1.10 | Paralelização |
| **clap** | 4.5 | CLI parser |
| **quick-xml** | 0.31 | XML parser |
| **tracing** | 0.1 | Logging estruturado |

### Features

```toml
[features]
default = []           # Vazio para evitar conflitos
cpu = ["burn/ndarray"]
gpu = ["burn/wgpu"]
cuda = ["burn/cuda-jit"]
```

**Build:** `cargo build --release --features cuda`

---

## 🎯 CLI COMMANDS (15 subcomandos em main.rs)

```
ptbr-slm <COMMAND>

COMMANDS:
  process-wiki     Processa dump Wikipedia XML.BZ2
  train-tokenizer  Treina tokenizer BPE
  tokenize         Tokeniza corpus para binário
  train            Treina modelo RWKV
  resume           Retoma treino de checkpoint
  test-model       Testa modelo com prompts
  generate         Gera texto a partir de prompt
  clean-corpus     Limpa corpus de texto
  audit-corpus     Audita qualidade do corpus
  info             Mostra informações do modelo
  build-dataset    Constrói dataset tokenizado
  test-gpu         Testa se GPU suporta o modelo
  find-lr          Encontra learning rate ótimo
  benchmark        Benchmark de performance
```

### Exemplos de uso:

```bash
# Treinar modelo
./target/release/ptbr-slm train \
  --data ./data/tokenized_v16_full/train.bin \
  --tokenizer ./data/tokenizer_v16_full/tokenizer.json \
  --output ./runs/smoke_800m \
  --model-size 800m \
  --max-steps 2 \
  --batch-size 1 \
  --seq-len 64 \
  --learning-rate 3e-4

# Gerar texto
./target/release/ptbr-slm generate \
  --model ./runs/checkpoint.bin \
  --tokenizer ./tokenizer.json \
  --prompt "O Brasil é" \
  --max-tokens 100
```

---

## 📈 STATUS ATUAL

### ✅ Implementado
- [x] Arquitetura RWKV completa
- [x] BPE tokenizer with LRU cache
- [x] Memory-mapped dataset loading
- [x] Training loop com gradient accumulation
- [x] Checkpoint save/load
- [x] Learning rate scheduling (warmup + cosine)
- [x] Wikipedia parser
- [x] Text cleaner com regex
- [x] CLI com 15 comandos
- [x] Multi-backend suporte (CPU/GPU/CUDA)
- [x] Batch encode/decode (Rayon)

### ❌ Não implementado / TODO
- [ ] Gradient clipping (aguardando Burn API)
- [ ] Quantization (INT8, FP8)
- [ ] Flash Attention optimization
- [ ] Distributed training (multi-GPU)
- [ ] Inference optimization (KV cache, batching)
- [ ] Deployment (ONNX, TorchScript)

### 🔧 Build Status
- `cargo build --release` → ❌ Falha (feature default vazio)
- `cargo build --release --features cuda` → ✅ Sucesso

---

## 📁 ESTRUTURA DE ARQUIVOS IMPORTANTES

### Código (5,064 linhas total Rust)

**Por módulo:**
- `model/` - 1,734 linhas (rwkv, trainer, config, etc)
- `tokenizer/` - 617 linhas (bpe, normalize)
- `data/` - 605 linhas (dataset, cleaner, parser)
- `main.rs` - 1,732 linhas (CLI dispatcher)
- `logger/`, `utils/` - 376 linhas

**Por arquivo (top 10):**
1. main.rs - 1,732
2. bpe.rs - 454
3. adapters.rs - 342
4. rwkv.rs - 405
5. validation.rs - 282
6. cleaner.rs - 281
7. metrics.rs - 264
8. trainer.rs - 213
9. dataset.rs - 191
10. normalize.rs - 163

### Dados (7.3 GB)

**Por tipo:**
- Texto limpo: 5.3 GB (328 arquivos)
- Tokenizado: 1.2 GB (1 arquivo .bin)
- Vocab + Merges: 3.88 MB (JSON)

**Por linguagem:**
- Wikipedia PT-BR: 2.3 GB
- Legislação: 4 MB
- Diverso: 5 GB

---

## 🚀 PRÓXIMOS PASSOS SUGERIDOS

1. **Treinar modelo 85M** completo em dados reais
2. **Avaliar perplexidade** em teste set
3. **Otimizar inference** (cache, batching)
4. **Quantização** (reduzir tamanho modelo)
5. **Deploy** em container/web API
6. **Fine-tuning** em domínios específicos (jurídico, etc)

---

## 📝 NOTAS

- **Backend:** Burn 0.14 (desenvolvimento ativo)
- **Rust:** Edition 2021
- **CUDA:** Suportado via `--features cuda`
- **CPU:** NdArray backend como fallback
- **Dados:** 100% públicos, licenças abertas
- **VRAM:** ~8GB para 85M, ~16GB para 1B

---

## 🔐 Licença

Apache 2.0 - Veja LICENSE

---

**Criado por:** Caike Costa  
**Última atualização:** 2026-01-11
