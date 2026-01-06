# 📚 PTBR-SLM - Arquitetura Completa

**Small Language Model em Rust focado em Português Brasileiro**

- **Data**: Janeiro 2026 (Atualizado)
- **Versão**: 0.1.0
- **Linguagem**: Rust 2021 Edition
- **Framework ML**: Burn (backend NdArray + Autodiff)
- **Status**: ✅ Múltiplas versões em treinamento - v3 em desenvolvimento

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Modelo](#arquitetura-do-modelo)
3. [Módulos do Projeto](#módulos-do-projeto)
4. [Fluxo de Dados](#fluxo-de-dados)
5. [Estrutura de Diretórios](#estrutura-de-diretórios)
6. [Dependências](#dependências)
7. [Comandos CLI](#comandos-cli)

---

## 🎯 Visão Geral

O **PTBR-SLM** é um modelo de linguagem pequeno baseado na arquitetura **RWKV** (RNN com Mixing Temporal), otimizado para:

- ✅ Funcionar em máquinas com **8GB RAM**
- ✅ Suportar **múltiplos tamanhos** de modelo (micro, mini, 85M)
- ✅ Processar texto em **Português Brasileiro**
- ✅ Treinamento iterativo com **checkpoints**
- ✅ Executar em **CPU** (backend NdArray)

### Configurações Disponíveis

| Config | Params | d_model | Layers | d_ffn | RAM |
|--------|--------|---------|--------|-------|-----|
| **micro** | 10M | 256 | 4 | 1024 | 2GB |
| **mini** | 20M | 384 | 6 | 1344 | 4GB |
| **85m** | 85M | 768 | 12 | 2688 | 8GB |

---

## 🧠 Arquitetura do Modelo

### RWKVConfig (Configuração)

```rust
pub struct RWKVConfig {
    pub vocab_size: usize,       // 32.000 tokens
    pub d_model: usize,           // Dimensão oculta
    pub n_layers: usize,          // Número de blocos
    pub d_ffn: usize,             // Dimensão FFN
    pub max_seq_len: usize,       // Comprimento máximo
    pub dropout: f64,             // Taxa de dropout
    pub layer_norm_eps: f64,      // Epsilon do LayerNorm
}
```

**Métodos Factory:**
- `RWKVConfig::ptbr_85m()` - 85M parâmetros
- `RWKVConfig::ptbr_mini()` - 20M parâmetros
- `RWKVConfig::ptbr_micro()` - 10M parâmetros
- `num_parameters()` - Cálculo total de parâmetros

---

### TrainingConfig (Configuração de Treinamento)

```rust
pub struct TrainingConfig {
    pub learning_rate: f64,              // 3e-4
    pub batch_size: usize,               // 2
    pub gradient_accumulation_steps: usize, // 16
    pub warmup_steps: usize,             // 500
    pub max_steps: usize,                // 50.000
    pub weight_decay: f64,               // 0.01
    pub max_grad_norm: f64,              // 1.0
}
```

---

### RWKV (Modelo Principal)

```rust
pub struct RWKV<B: Backend> {
    embedding: Embedding<B>,
    blocks: Vec<RWKVBlock<B>>,
    ln_out: LayerNorm<B>,
    lm_head: Linear<B>,
}
```

**Pipeline:**
1. Embedding → `[batch, seq_len, d_model]`
2. N blocos RWKV sequenciais
3. LayerNorm final
4. Projeção linear → logits `[batch, seq_len, vocab_size]`

---

### RWKVBlock (Bloco Fundamental)

Combina dois componentes principais:

```rust
pub struct RWKVBlock<B: Backend> {
    ln1: LayerNorm<B>,
    time_mixing: TimeMixing<B>,
    ln2: LayerNorm<B>,
    channel_mixing: ChannelMixing<B>,
    dropout: Dropout,
}
```

**Fluxo:**
```
x → LayerNorm → TimeMixing → Dropout → +
x → LayerNorm → ChannelMixing → Dropout → +
```

---

### TimeMixing (Análogo a Atenção)

Implementa uma forma eficiente de atenção temporal sem operações quadráticas:

```rust
pub struct TimeMixing<B: Backend> {
    key: Linear<B>,
    value: Linear<B>,
    receptance: Linear<B>,
    output: Linear<B>,
}
```

**Operações:**
1. Shift temporal: `x_prev = concat([zeros, x[:-1]])`
2. Mix: `mixed = (x + x_prev) / 2`
3. Projeções: `k = key(mixed)`, `v = value(mixed)`, `r = sigmoid(receptance(mixed))`
4. Atenção: `out = softmax(k) @ v`
5. Output: `output(r * out)`

**Vantagem:** Complexidade $O(n)$ em vez de $O(n^2)$

---

### ChannelMixing (FFN Eficiente)

Rede neural feedforward com ativação **ReLU Quadrado**:

```rust
pub struct ChannelMixing<B: Backend> {
    key: Linear<B>,        // d_model → d_ffn
    value: Linear<B>,      // d_ffn → d_model
    receptance: Linear<B>, // d_model → d_model
}
```

**Operações:**
1. Shift temporal igual a TimeMixing
2. FFN: `k = key(mixed)`
3. Ativação: `k_squared = ReLU(k)²`
4. Gating: `r = sigmoid(receptance(mixed))`
5. Output: `r * value(k_squared)`

**Ativação ReLU²:** $\max(0, x)^2$ - mais suave que ReLU padrão

---

### Trainer (Gerenciador de Treinamento)

```rust
pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,
    config: TrainingConfig,
    step: usize,
    device: B::Device,
}
```

**Métodos Principais:**

- `new()` - Inicializa modelo, otimizador e config
- `train_step(input_ids, target_ids)` → `f32`
  - Forward pass
  - Cálculo de cross-entropy loss
  - Backward pass (autograd)
  - Atualização de pesos com AdamW
  
- `cross_entropy_loss()` - Loss combinado de tokens
- `get_learning_rate()` - Warmup + linear decay
- `save_checkpoint(path)` - Persistência em `.mpk`
- `load_checkpoint(path)` - Carregamento de pesos

---

## 📦 Módulos do Projeto

### `src/model/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `config.rs` | RWKVConfig, TrainingConfig | Configurações do modelo |
| `rwkv.rs` | RWKV, RWKVBlock, TimeMixing, ChannelMixing | Arquitetura do modelo |
| `trainer.rs` | Trainer | Otimização e treinamento |
| `adapters.rs` | (Auxiliar) | Adaptadores do Burn framework |
| `mod.rs` | Exports | Exporta estruturas públicas |

---

### `src/data/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `wiki_parser.rs` | WikiStreamParser, WikiArticle | Parse streaming de dumps Wikipedia BZ2 |
| `cleaner.rs` | WikiCleaner | Limpeza de markup Wikipedia |
| `dataset.rs` | MmapDataset, DataLoader, TokenizedDatasetWriter | Dataset otimizado e I/O |
| `mod.rs` | Exports | Exporta estruturas públicas |

#### WikiStreamParser

Parse eficiente de dumps Wikipedia em formato BZ2:

```rust
pub struct WikiStreamParser {
    config: WikiParserConfig,
}
```

- **Descompactação on-the-fly** com `bzip2`
- **Parser XML** com `quick_xml`
- **Iterator lazy** - processa sob demanda
- **Eficiente em memória** - não carrega tudo na RAM

#### WikiCleaner

Remove markup residual de Wikipedia:

```rust
pub struct WikiCleaner {
    min_sentence_len: usize,
}
```

**Remoções:**
- `<ref>` tags e referências
- `[[Ficheiro:...]]`, `[[Arquivo:...]]` - links de arquivo
- `[[Categoria:...]]` - categorias
- `{{...}}` - templates
- `<...>` - tags HTML
- `[[...|...]]` - links Wikipedia
- Espaços múltiplos e quebras de linha excessivas
- Caracteres de controle

#### MmapDataset

Dataset otimizado com **memory-mapping**:

```rust
pub struct MmapDataset {
    data: Mmap,           // Arquivo mapeado em memória
    indices: Vec<usize>,  // Índices de sequências
    seq_len: usize,       // Comprimento de sequência
}
```

**Características:**
- Lê arquivo tokenizado via `memmap2`
- Tokens armazenados como `u16` (2 bytes cada)
- Não carrega tudo na RAM
- Suporta shuffling com seed

**Métodos:**
- `from_file(path, seq_len)` - Carrega dataset
- `get(idx)` → `(input: Vec<u16>, target: Vec<u16>)`
- `shuffle(seed)` - Embaralha índices
- `len()` - Quantidade de sequências

#### DataLoader

Iterator para batches do dataset:

```rust
pub struct DataLoader<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_idx: usize,
}
```

- Agrupa sequências em batches
- Mantém estado de iteração
- Iterator trait implementado

#### TokenizedDatasetWriter

Escritor eficiente de dados tokenizados:

```rust
pub struct TokenizedDatasetWriter {
    // Buffer para I/O otimizado
}
```

- Escreve tokens como `u16` em arquivo binário
- BufWriter para performance

---

### `src/tokenizer/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `bpe.rs` | BPETokenizer, BPETrainer, BPEVocab | Tokenização BPE |
| `normalize.rs` | PTBRNormalizer | Normalização PT-BR |
| `mod.rs` | Exports | Exporta estruturas públicas |

#### BPEVocab (Vocabulário)

Estrutura serializável para JSON:

```rust
pub struct BPEVocab {
    pub id_to_token: Vec<Vec<u8>>,              // ID → bytes
    pub merges: Vec<(u16, u16)>,                // Histórico de fusões
    pub special_tokens: HashMap<String, u16>,   // Special tokens
}
```

#### BPETokenizer (Tokenizador)

```rust
pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    special_tokens: HashMap<String, u16>,
    cache: HashMap<String, Vec<u16>>,  // Cache 10k entradas
}
```

**Tokens Especiais:**
- `[PAD]` (ID 0) - Padding
- `[UNK]` (ID 1) - Desconhecido
- `[BOS]` (ID 2) - Início de sequência
- `[EOS]` (ID 3) - Fim de sequência
- `[SEP]` (ID 4) - Separador

**Métodos:**
- `encode(text)` → `Vec<u16>` (com cache)
- `decode(ids)` → `String`
- `from_file(path)` - Carrega JSON
- `save(path)` - Salva JSON
- `vocab_size()` - Tamanho do vocabulário

**Caching:** 10.000 sequências tokenizadas em cache LRU

#### BPETrainer (Treinador)

```rust
pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: usize,
}
```

- Constrói vocabulário iterativamente
- **Paralelização com Rayon** para performance
- Suporta até 32.000 tokens

#### PTBRNormalizer (Normalização PT-BR)

```rust
pub struct PTBRNormalizer {
    // Configurações
}
```

- Tratamento de acentos
- Normalização de pontuação
- Conversão de casos

---

## 🔄 Fluxo de Dados

```
┌─────────────────────┐
│ Wikipedia PT-BR BZ2 │
└──────────┬──────────┘
           │
           │ ProcessWiki (WikiStreamParser + WikiCleaner)
           │
           ▼
┌────────────────────────┐
│  Texto Limpo (TXT)     │
│ data/wiki_clean/       │
└──────────┬─────────────┘
           │
           │ TrainTokenizer (BPETrainer)
           │
           ▼
┌──────────────────────────┐
│  Vocabulário BPE (JSON)  │
│ data/tokenizer_v2/       │
└──────────┬───────────────┘
           │
           │ Tokenize (BPETokenizer)
           │
           ▼
┌──────────────────────────────┐
│  Dataset Tokenizado (BIN)    │
│ data/tokenized_v2/           │
│ (tokens como u16 + mmap)     │
└──────────┬───────────────────┘
           │
           │ DataLoader + Train (RWKV + Trainer)
           │
           ▼
┌──────────────────────────┐
│  Modelo Treinado (MPK)   │
│ checkpoints_v2/          │
│ - checkpoint_XXXXX.bin   │
│ - model_final.mpk        │
└──────────┬───────────────┘
           │
           │ Generate (Inferência)
           │
           ▼
┌──────────────────┐
│  Texto Gerado    │
│  PT-BR          │
└──────────────────┘
```

---

## 📁 Estrutura de Diretórios

```
ptbr-slm/
├── Cargo.toml                    # Dependências do projeto
├── Cargo.lock                    # Lock de versões
├── ARQUITETURA.md               # Este documento
├── corpus.txt                   # Corpus de treinamento
│
├── src/
│   ├── main.rs                  # CLI e funções principais
│   ├── model/
│   │   ├── mod.rs
│   │   ├── config.rs           # RWKVConfig, TrainingConfig
│   │   ├── rwkv.rs             # RWKV, RWKVBlock, TimeMixing, ChannelMixing
│   │   ├── trainer.rs          # Trainer
│   │   └── adapters.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── wiki_parser.rs      # WikiStreamParser, WikiArticle
│   │   ├── cleaner.rs          # WikiCleaner
│   │   └── dataset.rs          # MmapDataset, DataLoader
│   └── tokenizer/
│       ├── mod.rs
│       ├── bpe.rs              # BPETokenizer, BPETrainer, BPEVocab
│       └── normalize.rs        # PTBRNormalizer
│
├── configs/
│   └── model_85m.toml           # Configuração modelo 85M
│
├── data/
│   ├── raw/                     # Dados brutos
│   ├── dumps/                   # Wikipedia BZ2 comprimidos
│   ├── wiki_clean/              # Wikipedia limpa (TXT)
│   ├── wiki_processed_v2/       # Wikipedia processada v2
│   ├── planalto_clean/          # Corpus Planalto limpo
│   ├── v15_clean/               # Corpus v15 limpo
│   ├── wikibooks_clean/         # Wikibooks limpo
│   ├── wikinews_clean/          # Wikinews limpo
│   ├── wikisource_clean/        # Wikisource limpo
│   ├── sovereign/               # Corpus adicional
│   ├── processed/
│   │   ├── corpus/              # Wikipedia parseada
│   │   └── tokenized/           # Dataset antigo
│   ├── tokenized_v2/            # Dataset tokenizado v2 (BIN)
│   ├── tokenized_v3/            # Dataset tokenizado v3 (BIN)
│   ├── tokenized_v12/           # Dataset tokenizado v12 (BIN)
│   ├── tokenized_v15/           # Dataset tokenizado v15 (BIN)
│   ├── tokenizer/
│   │   └── tokenizer_v2/        # Vocabulário v2 (JSON)
│   ├── tokenizer_v2/            # Tokenizer v2 (JSON)
│   ├── tokenizer_v3/            # Tokenizer v3 (JSON)
│   ├── tokenizer_v12/           # Tokenizer v12 (JSON)
│   └── tokenizer_v15/           # Tokenizer v15 (JSON)
│
├── checkpoints/                 # Checkpoints v1 (legacy)
│   ├── checkpoint_5000.mpk
│   ├── checkpoint_10000.mpk
│   ├── checkpoint_60000.mpk
│   └── model_final.mpk          # Modelo final v1
│
├── checkpoints_v2/              # Checkpoints v2 
│   ├── checkpoint_2500.mpk
│   ├── checkpoint_5000.mpk
│   ├── checkpoint_30000.mpk
│   └── ... (até checkpoint_27500.mpk)
│
├── checkpoints_v3/              # Checkpoints v3 (atual)
│   ├── checkpoint_2500.mpk
│   ├── checkpoint_5000.mpk
│   ├── checkpoint_45000.mpk     # ← Último checkpoint
│   └── ... (até checkpoint_45000.mpk)
│
├── checkpoints_v12/             # Checkpoints v12 (archive)
├── checkpoints_v12_micro/       # Checkpoints v12 micro (archive)
│
├── logs/                        # Logs de treinamento (vazio)
├── scripts/                     # Scripts de processamento e limpeza
│   ├── audit_sources_v15.py
│   ├── build_corpus_v15_stream.py
│   ├── clean_sources_v15.py
│   └── ... (diversos scripts de processamento)
│
└── target/
    └── release/
        └── ptbr-slm.exe         # Executável compilado
```

---

## 📚 Dependências

### Framework ML
- **burn** `0.14` - Framework deep learning
  - Features: `ndarray` (CPU), `autodiff`, `train`

### Processamento de Dados
- **quick-xml** `0.31` - Parser XML
- **bzip2** `0.4` - Descompactação BZ2
- **memmap2** `0.9` - Memory-mapping de arquivos
- **serde** + **serde_json** - Serialização JSON

### Texto
- **regex** `1.10` - Expressões regulares
- **once_cell** `1.19` - Lazy initialization
- **unicode-normalization** `0.1` - Normalização Unicode

### Concorrência e Performance
- **crossbeam-channel** `0.5` - Channels thread-safe
- **rayon** `1.10` - Data parallelism

### Utilidades
- **rand** + **rand_chacha** `0.8/0.3` - Randomização
- **clap** `4.5` - CLI parser
- **tracing** + **tracing-subscriber** - Logging

### Binários
- **bincode** `2.0-rc3` - Serialização binária

---

## ⚙️ Comandos CLI

### 1. Processar Wikipedia

```bash
./ptbr-slm.exe process-wiki \
  --input wiki-dump.xml.bz2 \
  --output data/wiki_clean
```

**Entrada:** Dump Wikipedia BZ2  
**Saída:** 
- `wiki_000.txt` (10k artigos cada)
- `wiki_001.txt`
- ...

---

### 2. Treinar Tokenizer BPE

```bash
./ptbr-slm.exe train-tokenizer \
  --corpus data/wiki_clean \
  --output data/tokenizer_v2 \
  --vocab-size 32000
```

**Entrada:** Diretório com TXTs  
**Saída:** `tokenizer_v2/tokenizer.json`

---

### 3. Tokenizar Corpus

```bash
./ptbr-slm.exe tokenize \
  --input data/wiki_clean \
  --output data/tokenized_v2 \
  --tokenizer data/tokenizer_v2/tokenizer.json
```

**Entrada:** 
- TXTs (texto limpo)
- JSON (vocabulário BPE)

**Saída:** Arquivo binário com tokens u16

---

### 4. Treinar Modelo

```bash
./ptbr-slm.exe train \
  --data data/tokenized_v2 \
  --tokenizer data/tokenizer_v2/tokenizer.json \
  --output checkpoints_v2 \
  --model-size micro
```

**Parâmetros:**
- `--model-size`: `micro` (10M), `mini` (20M), ou `85m` (85M)

**Saída:**
- `checkpoint_2500.bin`
- `checkpoint_5000.bin`
- ... (a cada 2500 steps)

---

### 5. Continuar Treinamento

```bash
./ptbr-slm.exe resume \
  --checkpoint checkpoints_v2/checkpoint_27500.bin \
  --data data/tokenized_v2 \
  --output checkpoints_v2 \
  --additional-steps 50000 \
  --model-size micro
```

**Carrega checkpoint e treina mais 50k steps**

---

### 6. Testar Modelo

```bash
./ptbr-slm.exe test-model \
  --model checkpoints_v2/checkpoint_27500.bin \
  --tokenizer data/tokenizer_v2/tokenizer.json
```

**Mostra top-5 predições para tokens aleatórios**

---

### 7. Gerar Texto

```bash
./ptbr-slm.exe generate \
  --model checkpoints_v2/checkpoint_27500.bin \
  --tokenizer data/tokenizer_v2/tokenizer.json \
  --prompt "O Brasil é um país que" \
  --max-tokens 100
```

**Saída:** Continuação de texto em Português Brasileiro

**Exemplo:**
```
Prompt: "O Brasil é um país que"
Generated: "O Brasil é um país que possui uma rica história e cultura. Com mais de 200 milhões de habitantes, é o maior país da América do Sul..."
```

---

### 8. Limpar Corpus

```bash
./ptbr-slm.exe clean-corpus \
  --input data/raw \
  --output data/wiki_clean \
  --verbose true
```

**Remove markup residual de Wikipedia**

---

## 🔧 Build e Execução

### Compilar Release

```bash
cargo build --release
```

**Otimizações:**
- LTO (Link Time Optimization)
- Single codegen unit
- Optimization level 3

### Executar

```bash
.\target\release\ptbr-slm.exe <COMANDO>
```

---

## 📊 Estatísticas do Projeto

| Métrica | Valor |
|---------|-------|
| **Linhas de Código** | ~2000+ |
| **Arquivos Rust** | 10+ |
| **Módulos Principais** | 3 (model, data, tokenizer) |
| **Estruturas Principais** | 12+ |
| **Comandos CLI** | 8 |
| **Tamanho Vocab** | 32.000 tokens |
| **Modelos Suportados** | 3 (micro, mini, 85m) |
| **Max Seq Length** | 256-2048 |
| **Versões Treinadas** | 5 (v1, v2, v3, v12, v15) |
| **Total de Checkpoints** | 100+ |
| **Corpora Processados** | 6+ |
| **Scripts de Processamento** | 20+

---

## 🎓 Conceitos-Chave

### Arquitetura RWKV

RWKV (RNN with Gated Linear Recurrence) oferece:
- ✅ Complexidade linear $O(n)$ vs transformers $O(n^2)$
- ✅ Long-range dependencies sem atenção quadrática
- ✅ Eficiência em memória
- ✅ Treinamento paralelo

### Time Mixing vs Channel Mixing

- **Time Mixing**: Processa ao longo da sequência (temporal)
- **Channel Mixing**: Processa ao longo das dimensões (feedforward)

Combinados formam um bloco eficiente equivalente a Transformer.

### BPE Tokenization

Byte Pair Encoding:
1. Começa com bytes individuais
2. Iterativamente funde pares mais frequentes
3. Até atingir tamanho de vocabulário desejado

**Vantagens:**
- Suporta qualquer caractere
- Vocabulário compacto
- Subword units

---

## 📈 Status de Treinamento

**Status Atual:** Janeiro 2026 - Múltiplas versões em desenvolvimento

### Checkpoints Disponíveis

| Versão | Última | Steps | Status | Dados |
|--------|--------|-------|--------|-------|
| **v1** | checkpoint_60000 | 60k | ✅ Finalizado | Wikipedia v1 |
| **v2** | checkpoint_27500 | 27.5k | ⏸️ Pausado | Wikipedia v2 |
| **v3** | checkpoint_45000 | 45k | 🔄 Ativo | Wiki + Planalto + Wikisource |
| **v12** | checkpoint_30000 | 30k | 📦 Archive | Corpus diversificado |
| **v12_micro** | checkpoint_5000 | 5k | 📦 Archive | Modelo micro experimental |

### Dados e Tokenizers Treinados

| Versão | Tokenizer | Dataset | Corpora |
|--------|-----------|---------|---------|
| **v2** | ✅ 32k vocab | tokenized_v2 | Wikipedia pt-br |
| **v3** | ✅ 32k vocab | tokenized_v3 | Wiki + Planalto + Wikisource |
| **v12** | ✅ 32k vocab | tokenized_v12 | Multi-corpora |
| **v15** | ✅ 32k vocab | tokenized_v15 | Ultra-clean |

### Corpora Disponíveis

- ✅ Wikipedia (wiki_clean, wiki_processed_v2)
- ✅ Planalto (legislação e atos do governo)
- ✅ Wikisource (obras literárias)
- ✅ Wikibooks, Wikinews (conteúdo adicional)
- ✅ Corpus Soberano (v15 ultra-limpo)

### Próximos Passos

- [ ] Finalizar v3 training (45k → 100k steps)
- [ ] Avaliar qualidade v3 vs v2
- [ ] Fine-tuning em domínios específicos
- [ ] Benchmark de performance
- [ ] Otimizações de quantização

---

## 🔐 Notas Técnicas

### Backend Burn

Usando **NdArray** (CPU) em vez de GPU:
- Compatibilidade universal
- Sem dependências de drivers
- Ideal para desenvolvimento
- Pode ser trocado para GPU posteriormente

### Memory-Mapping

Dataset usa `memmap2` para:
- Não carregar arquivo inteiro na RAM
- Acesso rápido a qualquer parte
- Escalabilidade

### Autodiff

Burn + Autodiff providencia:
- Automatic differentiation
- Backward pass automático
- Gradientes computados eficientemente

---

## 📝 Autores e Licença

- **Autor:** Caike Machado Batista Costa
- **Versão:** 0.1.0
- **Edition:** Rust 2021
- **Data Atualização:** Janeiro 2026

---

## 🚀 Timeline do Projeto

### Fase 1: Foundation (✅ Concluída)
- [x] Arquitetura RWKV implementada em Rust
- [x] Framework Burn integrado (NdArray backend)
- [x] CLI básico com 8 comandos
- [x] Dataset v1 com Wikipedia

### Fase 2: Escalabilidade (✅ Concluída)
- [x] Memory-mapping para datasets grandes
- [x] Múltiplas versões de tokenizers
- [x] Processamento paralelo com Rayon
- [x] Checkpoints em v2 e v3

### Fase 3: Expansão de Dados (✅ Concluída)
- [x] Integração Planalto (legislação)
- [x] Wikisource (obras literárias)
- [x] Wikibooks e Wikinews
- [x] Ultra-clean corpus v15

### Fase 4: Otimização (🔄 Em Progresso)
- [ ] Métricas de avaliação
- [ ] Fine-tuning especializado
- [ ] Quantização e compressão
- [ ] Backend GPU (Wgpu)

### Fase 5: Produção (⏳ Planejado)
- [ ] API REST
- [ ] Containerização (Docker)
- [ ] Deployment em cloud
- [ ] Benchmarks públicos
