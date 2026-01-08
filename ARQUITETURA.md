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
#[derive(Config, Debug)]
pub struct RWKVConfig {
    #[config(default = "32000")]
    pub vocab_size: usize,         // Tamanho do vocabulário
    
    #[config(default = "768")]
    pub d_model: usize,             // Dimensão oculta (embedding + outputs)
    
    #[config(default = "12")]
    pub n_layers: usize,            // Número de blocos RWKV sequenciais
    
    #[config(default = "2688")]
    pub d_ffn: usize,               // Dimensão da rede feedforward
    
    #[config(default = "2048")]
    pub max_seq_len: usize,         // Comprimento máximo de sequência
    
    #[config(default = "0.1")]
    pub dropout: f64,               // Taxa de dropout para regularização
    
    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,        // Epsilon de estabilidade do LayerNorm
}
```

**Métodos Factory (Pré-configurações):**

| Método | Params | d_model | Layers | d_ffn | max_seq | RAM | Uso |
|--------|--------|---------|--------|-------|---------|-----|-----|
| `ptbr_85m()` | 85M | 768 | 12 | 2688 | 2048 | 8GB | Produção, benchmark |
| `ptbr_mini()` | 20M | 384 | 6 | 1344 | 512 | 4GB | Desenvolvimento, testes rápidos |
| `ptbr_micro()` | 10M | 256 | 4 | 1024 | 256 | 2GB | Testes, prototipagem |

**Cálculo de Parâmetros:**

```rust
pub fn num_parameters(&self) -> usize {
    let embedding = vocab_size * d_model;                    // Embedding layer
    let time_mixing = 5 * d_model * d_model * n_layers;      // TimeMixing layers
    let channel_mixing = 2 * d_model * d_ffn * n_layers;     // ChannelMixing layers
    let layer_norms = 4 * d_model * n_layers;                // LayerNorm weights
    embedding + time_mixing + channel_mixing + layer_norms
}
```

**Exemplo de cálculo para 85M:**
- Embedding: 32k × 768 = 24.6M
- TimeMixing: 5 × 768 × 768 × 12 = 35.8M
- ChannelMixing: 2 × 768 × 2688 × 12 = 50.3M
- LayerNorm: 4 × 768 × 12 = 36.9K
- **Total ≈ 85M parâmetros**

---

### TrainingConfig (Configuração de Treinamento)

```rust
#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = "3e-4")]
    pub learning_rate: f64,              // Taxa de aprendizado inicial
    
    #[config(default = "2")]
    pub batch_size: usize,               // Tamanho do batch
    
    #[config(default = "16")]
    pub gradient_accumulation_steps: usize, // Acumulação de gradientes
    
    #[config(default = "500")]
    pub warmup_steps: usize,             // Steps de warmup linear
    
    #[config(default = "50000")]
    pub max_steps: usize,                // Total de steps de treinamento
    
    #[config(default = "0.01")]
    pub weight_decay: f64,               // Regularização L2 (AdamW)
    
    #[config(default = "1.0")]
    pub gradient_clip: f64,              // Valor máximo para gradient clipping
    
    #[config(default = "2500")]
    pub save_every: usize,               // Salvar checkpoint a cada N steps
    
    #[config(default = "500")]
    pub eval_every: usize,               // Avaliar a cada N steps
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 2,
            gradient_accumulation_steps: 16,
            warmup_steps: 500,
            max_steps: 50_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 2500,
            eval_every: 500,
        }
    }
}
```

**Pipeline de Treinamento:**

1. **Inicialização** (início de cada epoch/run)
   - Cria modelo RWKV com config
   - Inicializa otimizador AdamW com weight_decay
   - Carrega dataset tokenizado em memória mapeada

2. **Loop de Treinamento**
   - Para cada batch: inputs `[batch_size, seq_len]` → targets `[batch_size, seq_len]`
   - Forward pass → logits `[batch_size, seq_len, vocab_size]`
   - Calcula cross-entropy loss
   - Backward pass com autograd
   - Update de pesos com learning rate dinâmico

3. **Checkpointing**
   - A cada `save_every` steps: salva modelo em `.mpk` com CompactRecorder
   - Nomeia como `checkpoint_{step}.bin`
   - Permite retomar treinamento via `resume` command

4. **Learning Rate Schedule**
   - **Fase Warmup** (steps 0 a `warmup_steps`):  
     $lr = learning\_rate \times \frac{step}{warmup\_steps}$
   - **Fase Decay** (steps `warmup_steps` a `max_steps`):  
     $lr = learning\_rate \times (1 - 0.9 \times progress)$ com mínimo de 0.1  
     onde $progress = \frac{step - warmup\_steps}{max\_steps - warmup\_steps}$

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

**Pipeline de Treinamento:**

1. **Inicialização** (`new`)
   - Cria modelo RWKV com config
   - Inicializa otimizador AdamW uma única vez
   - Armazena config de treinamento

2. **Forward Pass** (`train_step`)
   - Embedding → Sequência de blocos RWKV → LayerNorm → LM Head
   - Entrada: `[batch_size, seq_len, vocab_size]`
   - Saída: logits

3. **Loss Calculation** (`cross_entropy_loss`)
   - Flattens logits e targets
   - Log-softmax nos logits
   - Negative log-likelihood (NLL) dos targets
   - Mean reduction

4. **Backward Pass & Weight Update** 
   - Automatic differentiation (autograd)
   - Compute gradients com `.backward()`
   - Update pesos com `optimizer.step()`
   - Learning rate scheduling (warmup + decay)

**Métodos Principais:**

- `new(model_config, train_config, device)` → Trainer
  - Cria modelo RWKV
  - Inicializa AdamW com weight decay
  - Retorna instância do trainer

- `train_step(input_ids, target_ids)` → `f32`
  - Forward pass com modelo
  - Calcula cross-entropy loss
  - Backward pass (autograd) 
  - Atualiza pesos com learning rate dinamicamente
  - Retorna valor escalar do loss
  
- `cross_entropy_loss(logits, targets)` → `Tensor<B, 1>`
  - Flatten logits e targets
  - Log-softmax + NLL
  - Mean reduction
  - Negativo para loss

- `get_learning_rate()` → `f64`
  - Warmup linear: $lr_{initial} \times \frac{step}{warmup\_steps}$ se $step < warmup\_steps$
  - Decay linear: $lr_{initial} \times (1 - 0.9 \times progress)$ com mínimo 0.1
  - Onde $progress = \frac{step - warmup\_steps}{max\_steps - warmup\_steps}$

- `save_checkpoint(path)` → `std::io::Result<()>`
  - Serializa modelo com CompactRecorder
  - Salva em formato `.mpk`
  - Remove extensão do path automaticamente

- `load_checkpoint(&mut self, path)` → `std::io::Result<()>`
  - Desserializa modelo do arquivo `.mpk`
  - Restaura pesos para device
  - Printa confirmação do carregamento

- `step()` → `usize` - Retorna contador de steps
- `config()` → `&TrainingConfig` - Retorna config imutável

**Tipos de Backend:**

- `B: AutodiffBackend` - Backend com suporte a autograd
  - `B::InnerBackend` - Backend subjacente (ex: NdArray)
  - `B::Device` - Device (CPU/GPU)

**Otimizador:**

```rust
AdamW {
    learning_rate: f64,      // Taxa de aprendizado
    beta_1: f64,             // 0.9 (momentum)
    beta_2: f64,             // 0.999 (adaptive rate)
    epsilon: f64,            // Estabilidade numérica
    weight_decay: f64,       // Regularização L2
}
```

Com adaptador para integração com Burn's gradient system.

---

## 📦 Módulos do Projeto

### `src/model/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `config.rs` | RWKVConfig, TrainingConfig | Configurações do modelo (85M/mini/micro) |
| `rwkv.rs` | RWKV, RWKVBlock, TimeMixing, ChannelMixing | Arquitetura RWKV completa |
| `trainer.rs` | Trainer | Loop de treinamento, loss, backward, checkpoints |
| `adapters.rs` | OptimizerAdaptor | Adaptador de otimizador para Burn |
| `mod.rs` | Exports | Re-exporta structs públicas |

**Linhas de Código por Arquivo:**
- `config.rs`: ~124 linhas
- `rwkv.rs`: ~400+ linhas (modelo + blocos)
- `trainer.rs`: ~131 linhas (treinamento)
- `adapters.rs`: ~50+ linhas (adaptadores)
- `mod.rs`: ~10 linhas

---

### `src/data/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `wiki_parser.rs` | WikiStreamParser, WikiArticle, WikiParserConfig | Parse streaming BZ2, Iterator lazy |
| `cleaner.rs` | WikiCleaner | Remove markup, normaliza qualidade |
| `dataset.rs` | MmapDataset, DataLoader, TokenizedDatasetWriter | Memory-mapping, batching, I/O |
| `mod.rs` | Exports | Re-exporta estruturas públicas |

**Linhas de Código por Arquivo:**
- `wiki_parser.rs`: ~200+ linhas (parser + iterador)
- `cleaner.rs`: ~150+ linhas (regex + limpeza)
- `dataset.rs`: ~155 linhas (mmap + dataloader)
- `mod.rs`: ~10 linhas

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

#### MmapDataset (Dataset Otimizado com Memory-Mapping)

```rust
pub struct MmapDataset {
    data: Mmap,           // Arquivo mapeado em memória (read-only)
    indices: Vec<usize>,  // Índices de início de cada sequência
    seq_len: usize,       // Comprimento esperado de sequência
}
```

**Características:**
- Lê arquivo tokenizado via `memmap2::Mmap` (não carrega tudo na RAM)
- Tokens armazenados como `u16` little-endian (2 bytes cada)
- Suporta shuffling determinístico com seed (ChaCha8)
- Índices pré-calculados para acesso O(1)

**Métodos:**
- `from_file(path, seq_len)` - Carrega dataset com inferência de tamanho
  - Calcula: `num_tokens = file_size / 2`
  - Calcula: `num_sequences = (num_tokens - seq_len) / seq_len`
  
- `get(idx)` → `Option<(Vec<u16>, Vec<u16>)>` - Retorna (input, target)
  - input: tokens 0..seq_len
  - target: tokens 1..(seq_len+1) - shift de 1 para predição de próximo token
  
- `shuffle(seed)` - Embaralha índices com ChaCha8Rng (determinístico)
  
- `len()` - Número total de sequências disponíveis
  
- `is_empty()` - Verifica se dataset está vazio

**Exemplo de uso:**
```rust
let dataset = MmapDataset::from_file("data/tokenized_v3/train.bin", 256)?;
println!("Sequences: {}", dataset.len());  // ex: 1,000,000

dataset.shuffle(42);  // Determinístico

if let Some((input, target)) = dataset.get(0) {
    println!("Input: {:?}", input);    // Vec<u16> com 256 tokens
    println!("Target: {:?}", target);  // Vec<u16> com 256 tokens
}
```

#### DataLoader (Iterator com Batching)

```rust
pub struct DataLoader<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a MmapDataset, batch_size: usize) -> Self { ... }
    pub fn reset(&mut self) { ... }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Vec<Vec<u16>>, Vec<Vec<u16>>);  // (inputs, targets)
    fn next(&mut self) -> Option<Self::Item> { ... }
}
```

**Funcionalidade:**
- Agrupa sequências do dataset em batches
- Mantém estado interno de iteração
- Implementa padrão Iterator de Rust
- Cada chamada a `next()` retorna `(batch_inputs, batch_targets)`
  - `batch_inputs`: Vec com `batch_size` sequências de input
  - `batch_targets`: Vec com `batch_size` sequências de target

**Exemplo:**
```rust
let dataset = MmapDataset::from_file("train.bin", 256)?;
let loader = DataLoader::new(&dataset, 32);  // batch_size=32

for (batch_inputs, batch_targets) in loader {
    // batch_inputs: Vec<Vec<u16>> com 32 sequências de 256 tokens
    // batch_targets: Vec<Vec<u16>> com 32 sequências de 256 tokens
    let loss = trainer.train_step(inputs_tensor, targets_tensor);
}
```

#### TokenizedDatasetWriter (Serialização de Tokens)

```rust
pub struct TokenizedDatasetWriter {
    writer: BufWriter<File>,
    tokens_written: usize,
}

impl TokenizedDatasetWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> { ... }
    pub fn write_tokens(&mut self, tokens: &[u16]) -> std::io::Result<()> { ... }
    pub fn finish(mut self) -> std::io::Result<usize> { ... }
}
```

**Funcionalidade:**
- Escreve tokens em arquivo binário com buffer otimizado (1MB)
- Cada token `u16` é serializado em little-endian (2 bytes)
- `finish()` faz flush e retorna total de tokens escritos

**Exemplo:**
```rust
let mut writer = TokenizedDatasetWriter::new("data/tokenized_v3/train.bin")?;

let tokens = tokenizer.encode("O Brasil é um país..."); // Vec<u16>
writer.write_tokens(&tokens)?;

let total = writer.finish()?;  // Flush e retorna count
println!("Total tokens: {}", total);
```

**Fluxo Completo:**
```
Texto em PT-BR
    ↓
BPETokenizer::encode()  → Vec<u16> (tokens)
    ↓
TokenizedDatasetWriter::write_tokens()  → Arquivo binário
    ↓
MmapDataset::from_file()  → Carrega para treinamento
```

---

### `src/tokenizer/`

| Arquivo | Componente | Responsabilidade |
|---------|-----------|------------------|
| `bpe.rs` | BPETokenizer, BPETrainer, BPEVocab | Tokenização BPE, treinamento vocab |
| `normalize.rs` | PTBRNormalizer | Normalização PT-BR (acentos, espaço) |
| `mod.rs` | Exports | Re-exporta estruturas públicas |

**Linhas de Código por Arquivo:**
- `bpe.rs`: ~500+ linhas (tokenizador + trainer + vocab)
- `normalize.rs`: ~100+ linhas (normalizador)
- `mod.rs`: ~10 linhas

#### BPEVocab (Estrutura de Vocabulário)

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct BPEVocab {
    pub id_to_token: Vec<Vec<u8>>,      // Mapeamento ID → bytes do token
    pub merges: Vec<(u16, u16)>,        // Histórico de fusões em ordem
    pub special_tokens: HashMap<String, u16>,  // Tokens especiais
}

impl BPEVocab {
    pub fn new() -> Self { ... }
    pub fn build_token_to_id(&self) -> HashMap<Vec<u8>, u16> { ... }
}
```

**Armazenamento:**
- Serializada em JSON (`tokenizer.json`)
- Contém tudo necessário para reconstruir BPETokenizer
- Tamanho típico: 200-500KB para vocab de 32k

**Exemplo de JSON:**
```json
{
  "id_to_token": [
    [112, 97, 100],      // "pad"
    [117, 110, 107],     // "unk"
    ...
  ],
  "merges": [
    [0, 1],   // Funde tokens 0 e 1
    [2, 3],   // Funde tokens 2 e 3
    ...
  ],
  "special_tokens": {
    "[PAD]": 0,
    "[UNK]": 1,
    ...
  }
}
```

#### BPETokenizer (Tokenizador Principal)

```rust
pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    special_tokens: HashMap<String, u16>,
    cache: HashMap<String, Vec<u16>>,  // LRU cache com até 10k entradas
}
```

**Tokens Especiais (IDs 0-4):**
- `[PAD]` (ID 0) - Padding para sequências curtas
- `[UNK]` (ID 1) - Token desconhecido (fallback)
- `[BOS]` (ID 2) - Beginning of sequence (início)
- `[EOS]` (ID 3) - End of sequence (fim)
- `[SEP]` (ID 4) - Separador (para pares de textos)

**Métodos Principais:**

- `encode(text: &str)` → `Vec<u16>` (com cache LRU)
  - Normaliza texto com PTBRNormalizer
  - Converte em bytes individuais como tokens iniciais
  - Aplica merges do BPE iterativamente
  - Verifica cache: hit evita recomputação
  - Retorna IDs dos tokens finais

- `decode(ids: &[u16])` → `String`
  - Converte IDs de volta para bytes
  - Junta bytes em string UTF-8
  - Trata tokens especiais corretamente

- `from_file(path: &str)` → `Result<Self>`
  - Carrega `tokenizer.json` salvo com serde
  - Reconstrói token_to_id a partir de id_to_token
  - Retorna tokenizador pronto

- `save(path: &str)` → `Result<()>`
  - Serializa em JSON com serde_json
  - Formato: { id_to_token, merges, special_tokens }

- `vocab_size()` → `usize`
  - Retorna: `id_to_token.len()`

- `eos_id()` → `u16`
  - Retorna ID do token [EOS]

**Cache LRU:**
- Limita a 10.000 entradas para não crescer infinitamente
- Melhora performance em textos repetitivos
- Implementado com HashMap (não é LRU puro, mas aproximado)

#### BPETrainer (Treinador de Vocabulário)

```rust
pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: usize,
}

impl BPETrainer {
    pub fn new(vocab_size: usize, min_frequency: usize) -> Self { ... }
    pub fn train(self, texts: Box<dyn Iterator<Item = String>>) -> BPEVocab { ... }
}
```

**Algoritmo BPE (Byte Pair Encoding):**

1. **Inicialização:** Começa com todos os bytes únicos (0-255) + special tokens
   - Inicial: 259 tokens (256 bytes + 5 specials - 2 overlap)

2. **Iteração Merging:**
   - Conta frequência de pares adjacentes em todo o corpus
   - Funde o par mais frequente em um novo token
   - Incrementa vocab de 1
   - Repete até atingir `vocab_size`

3. **Resultado:**
   - 32.000 tokens = 256 bytes + 31.744 merges
   - Cada merge cria um novo token com IDs 256+

**Paralelização com Rayon:**
- Processa corpus em paralelo
- Cada thread mantém contagem local de frequências
- Sincroniza para encontrar melhor merge
- Muito mais rápido para corpus grande

**Exemplo:**
```rust
let trainer = BPETrainer::new(32_000, 2);  // Vocab 32k, min freq 2

let texts = Box::new(
    std::fs::read_dir("data/wiki_clean")?
        .filter_map(|e| e.ok())
        .flat_map(|e| std::fs::read_to_string(e.path()).ok().into_iter())
        .flat_map(|c| c.lines().map(String::from).collect::<Vec<_>>())
);

let vocab = trainer.train(texts);
let tokenizer = BPETokenizer::from_vocab(vocab);
tokenizer.save("data/tokenizer_v3/tokenizer.json")?;
```

**Tempo de Treinamento:**
- 1 GB de texto: ~5-10 minutos (com Rayon)
- 10 GB de texto: ~30-60 minutos
- Depende de: CPU cores, velocidade de I/O, tamanho final do vocab

#### PTBRNormalizer (Normalização Portuguesa)

```rust
pub struct PTBRNormalizer {
    // Configurações internas
}

impl PTBRNormalizer {
    pub fn new() -> Self { ... }
    pub fn normalize(text: &str) -> String { ... }
}
```

**Normalizações Aplicadas:**

1. **Acentuação:**
   - Unicode NFD decomposition (separa base de diacríticos)
   - Opcionalmente remove acentos (configurável)

2. **Pontuação:**
   - Padroniza tipos de aspas: " → " (Unicode)
   - Padroniza travessões: - → —
   - Espaços ao redor de pontuação

3. **Espaçamento:**
   - Remove espaços múltiplos → espaço único
   - Normaliza quebras de linha: \r\n, \r → \n
   - Remove espaços no início/fim de linhas

4. **Case:** 
   - Mantém case original (não força lowercase)
   - Preserva siglas: "PT" vs "Portugal"

5. **Caracteres de Controle:**
   - Remove: \x00-\x08, \x0b-\x0c, \x0e-\x1f
   - Mantém: \n (\x0a), \t (\x09)

**Exemplo:**
```rust
let text = "  O  Brasil   é  um  país...  ";
let normalized = PTBRNormalizer::normalize(text);
// Resultado: "O Brasil é um país..."

let text_acentos = "São Paulo - maçã";
// Com NFD: "Sa~o Paulo - mac~a" (base + diacríticos separados)
```

**Quando é Aplicado:**
- Durante `encode()` do BPETokenizer (antes de tokenizar)
- Garante consistência: "BRASIL" e "Brasil" são normalizados

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
├── scripts/                     # Scripts Python de processamento
│   ├── audit_sources_v15.py              # Auditoria de fontes v15
│   ├── build_corpus_v15_stream.py        # Build streaming v15
│   ├── build_corpus_v15_v2_clean.py      # Build alternativo v15
│   ├── clean_sources_v15.py              # Limpeza de markup v15
│   ├── clean_sovereign.py                # Processa corpus soberano
│   ├── extract_chroma.py                 # Extrai características
│   ├── filter_v11_brasil_only.py         # Filtro Brasil-only (legacy)
│   ├── filter_v12_ultra_clean.py         # Filtro ultra-clean v12 (legacy)
│   ├── filter_v13_keep_more.py           # Filtro mais permissivo v13
│   ├── filter_v14_quick_clean.py         # Filtro rápido v14
│   ├── fix_newlines_to_lf.py             # Normaliza quebras de linha
│   ├── planalto_codigo_civil_v3.py       # Processa legislação v3
│   ├── planalto_resolve_codigo_civil.py  # Processa CC v1 (legacy)
│   ├── planalto_resolve_codigo_civil_v2.py # Processa CC v2
│   ├── scrape_planalto_fix_frames.py     # Scraper Planalto com frames
│   ├── scrape_planalto_seeds.py          # Scraper Planalto v1 (legacy)
│   ├── scrape_planalto_seeds_v2.py       # Scraper Planalto v2
│   └── unify_corpus.py                   # Unifica múltiplas fontes
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
  - GPU Feature: `wgpu` (opcional, feature `gpu`)

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

### Serialização
- **bincode** `2.0-rc3` - Serialização binária (checkpoint save/load)

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
  --data data/tokenized_v3 \
  --tokenizer data/tokenizer_v3/tokenizer.json \
  --output checkpoints_v3 \
  --model-size micro \
  --max-steps 100000 \
  --save-every 2500 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --warmup-steps 1000 \
  --seq-len 256
```

**Entrada:**
- `--data`: Diretório com arquivo `train.bin` (tokens u16)
- `--tokenizer`: Caminho para `tokenizer.json`

**Parâmetros de Modelo:**
- `--model-size`: `micro` (10M), `mini` (20M), ou `85m` (85M)

**Parâmetros de Treinamento:**
- `--max-steps`: Total de steps (default: 50000)
- `--save-every`: Salvar checkpoint a cada N steps (default: 2500)
- `--batch-size`: Tamanho do batch (default: 2)
- `--learning-rate`: Taxa de aprendizado (default: 3e-4)
- `--warmup-steps`: Steps de warmup linear (default: 500)
- `--seq-len`: Comprimento de sequência (default: 256)
- `--grad-accum`: Acumulação de gradientes (default: 16)

**Saída:**
- `checkpoints_v3/checkpoint_2500.bin`
- `checkpoints_v3/checkpoint_5000.bin`
- ... (a cada `save_every` steps)
- `checkpoints_v3/model_step_100000.bin` (modelo final)

**Output no Terminal:**
```
Modelo criado com 10.5M parâmetros
  Iniciando do step 0...

Step    100 | Loss: 4.5234 | 2.45 steps/s | ETA: 11h23m
Step    200 | Loss: 3.8456 | 2.48 steps/s | ETA: 11h15m
  Checkpoint salvo: "checkpoints_v3/checkpoint_2500.bin"
...
```

---

### 5. Continuar Treinamento (Resume)

```bash
./ptbr-slm.exe resume \
  --checkpoint checkpoints_v3/checkpoint_45000.bin \
  --data data/tokenized_v3 \
  --output checkpoints_v3 \
  --additional-steps 55000 \
  --model-size micro \
  --save-every 2500 \
  --batch-size 8
```

**Entrada:**
- `--checkpoint`: Caminho do checkpoint a continuar (ex: `checkpoint_45000.bin`)
- `--data`: Dataset de treinamento
- `--additional-steps`: Quantos steps adicionais treinar (não total!)

**Funcionalidade:**
- Carrega pesos salvos do checkpoint
- Recupera step interno do nome do arquivo
- Continua treinamento a partir desse ponto
- Salva novos checkpoints com step correto

**Exemplo Prático:**
- Se checkpoint é `checkpoint_45000.bin` e `--additional-steps 55000`:
- Treinamento vai de step 45000 até 100000
- Vai salvar em: 47500, 50000, 52500, ..., 100000

**Vantagens:**
- Recuperação de falhas (crash, interrupção)
- Treinamento em fases (ajustar LR, batch size, etc)
- Eficiência: reutiliza pesos já aprendidos

---

### 6. Testar Modelo (Inference)

```bash
./ptbr-slm.exe test-model \
  --model checkpoints_v3/checkpoint_45000.bin \
  --tokenizer data/tokenizer_v3/tokenizer.json \
  --model-size micro
```

**Entrada:**
- `--model`: Caminho do checkpoint a testar
- `--tokenizer`: Vocabulário tokenizador
- `--model-size`: Tamanho do modelo (deve ser o mesmo do treinamento)

**Funcionalidade:**
- Carrega modelo e tokenizador
- Seleciona tokens aleatórios do vocabulário
- Mostra top-5 predições com probabilidades
- Permite avaliar qualidade de forma simples

**Exemplo de Output:**
```
Testando modelo...

Token: "Brasil"
  1. " é" (prob: 0.234)
  2. " um" (prob: 0.189)
  3. " país" (prob: 0.145)
  4. " de" (prob: 0.098)
  5. " estado" (prob: 0.076)

Token: "linguagem"
  1. " de" (prob: 0.312)
  2. " portuguesa" (prob: 0.256)
  3. " é" (prob: 0.148)
  4. " programa" (prob: 0.089)
  5. " artificial" (prob: 0.065)
```

**Utilitário Para:**
- Debug inicial: modelo carrega/roda?
- Qualidade informal: predições fazem sentido?
- Comparar versões: v3 vs v2

---

### 7. Gerar Texto (Generation)

```bash
./ptbr-slm.exe generate \
  --model checkpoints_v3/checkpoint_45000.bin \
  --tokenizer data/tokenizer_v3/tokenizer.json \
  --prompt "O Brasil é um país" \
  --max-tokens 100 \
  --model-size micro \
  --temperature 0.8
```

**Entrada:**
- `--prompt`: Texto inicial para continuar
- `--max-tokens`: Número máximo de tokens a gerar (default: 100)
- `--temperature`: Controla aleatoriedade (0.1=determinístico, 2.0=muito aleatório)
  - 0.5-0.8: Bom balanço
  - 0.3: Conservador, predições óbvias
  - 1.2: Mais criativo, mas pode sair do contexto

**Algoritmo:**
1. Tokeniza prompt → `Vec<u16>`
2. Para cada step até `max-tokens`:
   - Forward pass: últimos tokens → logits
   - Aplica temperature: divide logits por temperature
   - Computa softmax: logits → probabilidades
   - Amostra next token com WeightedIndex
   - Se token == [EOS]: para
   - Senão: adiciona token e continua

3. Decodifica tokens → string

**Exemplo de Output:**
```
Prompt: O Brasil é um país
Temperatura: 0.8
Gerado: O Brasil é um país que possui uma rica história cultural e natural. Com mais de 200 milhões de habitantes e uma vasta extensão territorial, o Brasil é conhecido mundialmente pela sua biodiversidade única, pelas suas florestas tropicais e pela sua população diversa. A economia brasileira é uma das maiores do mundo, com setores importantes como a agricultura...
```

**Notas:**
- Qualidade melhora com mais treinamento (steps)
- Temperature afeta coerência vs diversidade
- Modelo pode repetir ou divergir (limitações do modelo pequeno)

---

### 8. Limpar Corpus

```bash
./ptbr-slm.exe clean-corpus \
  --input data/raw \
  --output data/wiki_clean \
  --verbose true
```

**Entrada:**
- `--input`: Diretório com arquivos TXT (brutos de Wikipedia)

**Funcionalidade:**
- Remove markup residual de Wikipedia
- Filtra blocos de texto de baixa qualidade
- Detecta e remove "garbage" (markup deixado)
- Salva versão limpa

**Remoções Aplicadas:**

1. **Tags XML/HTML:**
   - `<ref>...</ref>` - Referências
   - `<!--...-->` - Comentários HTML
   - Tags genéricas: `<...>`

2. **Markup Wikipedia:**
   - `[[Ficheiro:...]]`, `[[Arquivo:...]]` - Links de arquivo
   - `[[Categoria:...]]` - Categorias
   - `[[...|...]]` - Links internos com pipe
   - `{{...}}` - Templates

3. **Caracteres Especiais:**
   - `{{`, `}}` - Brackets duplos
   - `|-`, `|` - Delimitadores de tabela
   - `align=`, `width=`, `colspan=` - Atributos HTML

4. **Formatação:**
   - Espaços múltiplos → espaço único
   - Quebras de linha excessivas → máx 2 quebras
   - Caracteres de controle removidos

**Filtro de Qualidade:**
- Remove blocos com < 100 caracteres
- Remove blocos sem linhas com > 50 caracteres
- Remove blocos que contêm garbage após limpeza

**Output:**
```
===================================================
  Limpando corpus
===================================================

  Encontrados 15 arquivos

  wiki_001.txt: 2540KB -> 1820KB
  wiki_002.txt: 2310KB -> 1650KB
  ...

===================================================
  Arquivos salvos: 15
  Tamanho: 45.2MB -> 32.1MB (28.9% removido)
===================================================
```

**Tempo:**
- 1 GB: ~5 minutos
- 10 GB: ~30-40 minutos

---

## 🔧 Build e Execução

### Compilar Release

```bash
cargo build --release
```

**Otimizações Aplicadas** (em `Cargo.toml`):
```toml
[profile.release]
lto = true              # Link Time Optimization
codegen-units = 1      # Single code gen unit (mais otimizado, mais lento)
opt-level = 3          # Máxima otimização
```

**Tempo de compilação:**
- Primeira vez: 3-5 minutos (download deps)
- Subsequentes: ~1-2 minutos

**Resultado:**
```
./target/release/ptbr-slm.exe  (~50-80 MB)
```

### Executar Comandos

```bash
# Windows
.\target\release\ptbr-slm.exe <COMANDO>

# Linux/Mac
./target/release/ptbr-slm.exe <COMANDO>
```

### Build de Desenvolvimento (Debug)

```bash
cargo build  # Sem --release
cargo run -- <COMANDO>
```

**Nota:** Build Debug é 10-100x mais lento. Use apenas para debugging!

---

## � Workflow Completo: Do Zero ao Modelo Treinado

### Cenário: Treinar Novo Modelo v4 do Zero

**Premissas:**
- Wikipedia PT-BR em `data/dumps/` (em formato BZ2)
- 8GB RAM disponível
- ~30 horas de tempo de treino

### Passo 1: Processar Wikipedia (1-2h)

```bash
.\target\release\ptbr-slm.exe process-wiki \
  --input data/dumps/pt-latest-pages-articles.xml.bz2 \
  --output data/wiki_raw_v4
```

**Resultado:** Arquivos TXT em `data/wiki_raw_v4/`
- Estrutura: `wiki_000.txt`, `wiki_001.txt`, ...
- Cada arquivo: ~10.000 artigos
- Total: ~500MB-1GB

### Passo 2: Limpar Corpus (1h)

```bash
.\target\release\ptbr-slm.exe clean-corpus \
  --input data/wiki_raw_v4 \
  --output data/wiki_clean_v4 \
  --verbose true
```

**Resultado:** Corpus limpo em `data/wiki_clean_v4/`
- Remoção de markup, limpeza de qualidade
- Redução: ~30-40% do tamanho original

### Passo 3: Treinar Tokenizer (10-20m)

```bash
.\target\release\ptbr-slm.exe train-tokenizer \
  --corpus data/wiki_clean_v4 \
  --output data/tokenizer_v4 \
  --vocab-size 32000
```

**Resultado:** `data/tokenizer_v4/tokenizer.json`
- Vocabulário BPE com 32.000 tokens
- Contém: base + merges + special tokens

### Passo 4: Tokenizar Dataset (30-60m)

```bash
.\target\release\ptbr-slm.exe tokenize \
  --input data/wiki_clean_v4 \
  --output data/tokenized_v4 \
  --tokenizer data/tokenizer_v4/tokenizer.json
```

**Resultado:** `data/tokenized_v4/train.bin`
- Arquivo binário com tokens u16
- Tamanho: ~200-400MB (dependendo do corpus)
- Pronto para treinamento

### Passo 5: Treinar Modelo (20-30h)

```bash
.\target\release\ptbr-slm.exe train \
  --data data/tokenized_v4 \
  --tokenizer data/tokenizer_v4/tokenizer.json \
  --output checkpoints_v4 \
  --model-size micro \
  --max-steps 100000 \
  --save-every 2500 \
  --batch-size 4 \
  --warmup-steps 1000 \
  --seq-len 256
```

**Durante o Treinamento:**
- Checkpoints salvos: 40x (a cada 2.500 steps)
- Terminal mostra: loss, steps/sec, ETA
- Prova de conceito em ~1 hora (10k steps)

**Resultado:** `checkpoints_v4/`
- `checkpoint_2500.bin`, `checkpoint_5000.bin`, ...
- `model_step_100000.bin` (modelo final)

### Passo 6: Testar Modelo

```bash
.\target\release\ptbr-slm.exe test-model \
  --model checkpoints_v4/checkpoint_100000.bin \
  --tokenizer data/tokenizer_v4/tokenizer.json \
  --model-size micro
```

### Passo 7: Gerar Texto

```bash
.\target\release\ptbr-slm.exe generate \
  --model checkpoints_v4/checkpoint_100000.bin \
  --tokenizer data/tokenizer_v4/tokenizer.json \
  --prompt "O Brasil é" \
  --max-tokens 150 \
  --temperature 0.8
```

### Timeline Resumido

| Etapa | Tempo | Output |
|-------|-------|--------|
| Process | 1-2h | wiki_raw_v4/ |
| Clean | 0.5-1h | wiki_clean_v4/ |
| Tokenize Training | 0.25-0.5h | tokenizer_v4/ |
| Tokenize Data | 0.5-1h | tokenized_v4/ |
| Train | 20-30h | checkpoints_v4/ |
| **TOTAL** | **~24-34h** | **Modelo pronto!** |

### Alternativa: Continuar Treinamento Existente

Se já existe `checkpoint_50000.bin` e quer continuar:

```bash
.\target\release\ptbr-slm.exe resume \
  --checkpoint checkpoints_v3/checkpoint_50000.bin \
  --data data/tokenized_v3 \
  --output checkpoints_v3 \
  --additional-steps 50000 \
  --model-size micro
```

- Vai treinar até step 100.000 (50k + 50k)
- Muito mais rápido: reutiliza pesos já aprendidos
- Tempo: ~10-15h (em vez de 25h)

| Métrica | Valor |
|---------|-------|
| **Linhas de Código Rust** | ~2500+ |
| **Arquivos Rust** | 11 |
| **Módulos Principais** | 3 (model, data, tokenizer) |
| **Estruturas Principais** | 14+ |
| **Comandos CLI** | 8 |
| **Tamanho Vocab** | 32.000 tokens |
| **Modelos Suportados** | 3 (micro, mini, 85m) |
| **Max Seq Length** | 256-2048 |
| **Versões Treinadas** | 5 (v1, v2, v3, v12, v15) |
| **Total de Checkpoints** | 100+ |
| **Corpora Processados** | 6+ (Wikipedia, Planalto, Wikisource, Wikibooks, Wikinews, Sovereign) |
| **Scripts de Processamento** | 18 |
| **Dependências Diretas** | 16 |

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

**Status Atual:** Janeiro 2026 - v3 em progresso (45k steps concluídos)

### Checkpoints Disponíveis

| Versão | Última | Steps | Status | Dados |
|--------|--------|-------|--------|-------|
| **v1** | checkpoint_60000 | 60k | ✅ Finalizado | Wikipedia v1 |
| **v2** | checkpoint_30000 | 30k | ⏸️ Pausado | Wikipedia v2 |
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

- [x] Dados preparados: Wikipedia (6 corpora) + Planalto + Wikisource
- [x] v3 em treinamento (45k steps)
- [ ] Continuar v3 training até 100k steps
- [ ] Avaliar qualidade v3 vs v2
- [ ] Fine-tuning em domínios específicos (jurídico, técnico)
- [ ] Benchmark de performance contra baselines
- [ ] Otimizações de quantização para deployment

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

### Trainer - Implementação Detalhada

**Inicialização do Otimizador:**
```rust
let optimizer = AdamWConfig::new()
    .with_weight_decay(train_config.weight_decay as f32)
    .init();
```
- Otimizador criado **UMA ÚNICA VEZ** no construtor
- Configurado com weight decay para regularização
- Não é recreado a cada step

**Update de Pesos no Train Step:**
```rust
let grad_params = GradientsParams::from_grads(grads, &self.model);
self.model = self.optimizer.step(lr, self.model.clone(), grad_params);
```
- Extrai gradientes com `loss.backward()`
- Converte em `GradientsParams` com referência ao modelo
- `optimizer.step()` retorna modelo atualizado
- Learning rate é dinâmico (warmup + decay)

**Learning Rate Schedule:**
- Fase Warmup: Aumenta linearmente de 0 até `learning_rate` durante `warmup_steps`
- Fase Decay: Reduz linearmente de `learning_rate` até 10% durante `max_steps`
- Implementado sem callback - calculado a cada step

**Loss Calculation:**
```rust
let log_probs = activation::log_softmax(logits_flat, 1);
selected.mean().neg()
```
- Cross-entropy = negativo da mean log-likelihood
- Aplicado ao vocabulário inteiro
- Reduzido via média (não soma)

---

## � Scripts Python (`scripts/`)

Complementam o projeto com utilitários de processamento de dados em Python.

### Scripts de Corpus v15 (Ultra-Clean)

#### `audit_sources_v15.py`
- **Propósito:** Auditoria de fontes de dados v15
- **Entrada:** Diretórios de corpus brutos
- **Saída:** Relatório de qualidade, estatísticas por fonte
- **Uso:** `python scripts/audit_sources_v15.py --input data/raw`

#### `build_corpus_v15_stream.py`
- **Propósito:** Build streaming de corpus v15 (sem carregar tudo na RAM)
- **Entrada:** Múltiplas fontes (Wikipedia, Planalto, Wikisource)
- **Saída:** Arquivo unificado em v15_clean/
- **Otimizações:** Iterator lazy, processamento em chunks
- **Uso:** `python scripts/build_corpus_v15_stream.py --output data/v15_clean`

#### `build_corpus_v15_v2_clean.py`
- **Propósito:** Variação alternativa do build v15
- **Diferença:** Diferentes critérios de limpeza, mais agressivo
- **Output:** v15_clean_v2/ (archive)

#### `clean_sources_v15.py`
- **Propósito:** Limpeza de markup para corpus v15
- **Entrada:** Arquivos brutos com markup Wikipedia/Planalto
- **Saída:** Texto limpo e normalizado
- **Remove:** HTML, templates, referências, caracteres especiais
- **Uso:** `python scripts/clean_sources_v15.py --input raw/ --output clean/`

### Scripts de Filtro (Evoluções v11-v14)

#### `filter_v11_brasil_only.py`
- **Propósito:** Filtro v11 - apenas conteúdo Brasil
- **Critério:** Remove conteúdo não-relacionado ao Brasil
- **Status:** ✅ Legacy/Archive
- **Output:** filtered_v11/ (archive)

#### `filter_v12_ultra_clean.py`
- **Propósito:** Filtro v12 - ultra limpeza agressiva
- **Critério:** Muito agressivo, remove muito markup e conteúdo curto
- **Status:** ✅ Usado em v12 (archive)
- **Output:** filtered_v12/ (archive)

#### `filter_v13_keep_more.py`
- **Propósito:** Filtro v13 - mantém mais conteúdo
- **Critério:** Menos agressivo que v12, balanço qualidade/quantidade
- **Status:** 🔄 Experimental
- **Output:** filtered_v13/ (archive)

#### `filter_v14_quick_clean.py`
- **Propósito:** Filtro v14 - rápido e pragmático
- **Critério:** Rápido de executar, qualidade razoável
- **Status:** 🔄 Experimental
- **Output:** filtered_v14/ (archive)

### Scripts de Normalização

#### `fix_newlines_to_lf.py`
- **Propósito:** Normalizar quebras de linha
- **Converte:** \r\n, \r → \n (LF Unix standard)
- **Entrada:** Arquivos com quebras inconsistentes
- **Saída:** Arquivos com LF padronizado
- **Uso:** `python scripts/fix_newlines_to_lf.py --input raw/ --output fixed/`

### Scripts do Planalto (Legislação Brasileira)

#### `scrape_planalto_seeds.py`
- **Propósito:** Scraper inicial de planalto.gov.br
- **Alvo:** Leis, decretos, atos administrativos, Código Civil
- **Output:** Seeds/URLs para processamento posterior
- **Status:** ✅ v1 (legacy, funcional)

#### `scrape_planalto_seeds_v2.py`
- **Propósito:** Scraper v2 com melhorias
- **Melhorias:** 
  - Tratamento de frames HTML
  - Sessões HTTP com retry
  - Parser robusto de elementos
- **Output:** planalto_raw/ (bruto com markup)
- **Status:** ✅ v2 (mais robusta)

#### `scrape_planalto_fix_frames.py`
- **Propósito:** Scraper com suporte especial para frames
- **Foco:** Wikipedia-style frame extraction
- **Output:** planalto_frames_raw/
- **Status:** 🔄 Variação experimental

#### `planalto_resolve_codigo_civil.py`
- **Propósito:** Processa especificamente Código Civil brasileiro
- **Entrada:** HTML bruto do planalto.gov.br
- **Saída:** Texto limpo com estrutura de artigos
- **Preserva:** Números de artigos, estrutura legal
- **Status:** ✅ v1 (funcional)

#### `planalto_resolve_codigo_civil_v2.py`
- **Propósito:** Versão v2 com parsing melhorado
- **Melhoria:** 
  - Mantém estrutura de artigos e incisos
  - Preserva formatação jurídica importante
  - Melhor tratamento de parágrafos
- **Output:** planalto_clean/ (como usado em v3)
- **Status:** ✅ v2 (mais preciso)

#### `planalto_codigo_civil_v3.py`
- **Propósito:** Versão v3 com suporte a múltiplas leis
- **Expansão:** Não só Código Civil, mas:
  - Lei Complementar
  - Código Penal
  - Leis Especiais
  - Decreto-Lei
- **Output:** planalto_clean/ (atualizado e expandido)
- **Status:** ✅ v3 (mais abrangente)

### Scripts de Extração de Features

#### `extract_chroma.py`
- **Propósito:** Extrair "chroma" (características/features salientes)
- **Função:** Identificar trechos mais informativos e únicos do corpus
- **Output:** corpus_highlights/ (para análise e curation)
- **Uso:** Seleção de exemplos para fine-tuning específico
- **Status:** 🔄 Experimental, not actively used

### Scripts de Unificação

#### `unify_corpus.py`
- **Propósito:** Unifica múltiplas fontes em corpus único coeso
- **Entrada:** 
  - Wikipedia (wiki_clean/)
  - Planalto (planalto_clean/)
  - Wikisource (wikisource_clean/)
  - Wikibooks (wikibooks_clean/)
  - Wikinews (wikinews_clean/)
- **Saída:** Arquivo único consolidated.txt com separadores
- **Estratégia:** 
  - Mantém proporções de cada fonte
  - Adiciona marcadores de seção
  - Embaralha entre fontes para melhor mix
- **Uso:** `python scripts/unify_corpus.py --output data/unified_corpus.txt`
- **Status:** ✅ Funcional, usado em v3

#### `clean_sovereign.py`
- **Propósito:** Processa corpus "Soberano" (proprietário/específico)
- **Entrada:** Corpus externo de domínio específico (ex: textos técnicos)
- **Saída:** sovereign_clean/ (integrado ao v15)
- **Função:** Adicionar conteúdo especializado ao treinamento
- **Uso:** Melhorar performance em domínios específicos
- **Status:** ✅ Funcional, usado em v15

---

## 📂 Estrutura de Dados Completa (`data/`)

### Diretórios de Dumps Originais

#### `dumps/`
- **Conteúdo:** Wikipedia XML BZ2 comprimidos
- **Arquivos:** 3 arquivos
- **Tamanho Total:** 130.06 MB
- **Formato:** BZ2 (comprimido)
- **Descomprimido:** ~600 MB-1 GB (estimado)
- **Uso:** Entrada para `process-wiki` command
- **Status:** ✅ Disponível

#### `raw/`
- **Conteúdo:** Dados brutos antes de processamento
- **Fontes:** Scrapes do Planalto, exports de Wikibooks, etc
- **Formato:** Texto/HTML misto
- **Tamanho:** 2.4 GB (1 arquivo)
- **Arquivos:** Corpus bruto não-processado
- **Status:** ⚠️ Intermediário (descartável após limpeza)

### Diretórios Limpos (Processados)

#### `wiki_clean/`
- **Conteúdo:** Wikipedia PT-BR após WikiCleaner
- **Arquivos:** 132 arquivos
- **Tamanho Total:** 2.38 GB
- **Formato:** TXT (um documento por linha ou por arquivo)
- **Estrutura:** `wiki_000.txt`, `wiki_001.txt`, etc (10k artigos cada)
- **Usado Por:** `train-tokenizer`, `tokenize` commands
- **Status:** ✅ Ativo, qualidade alta

#### `wiki_processed_v2/`
- **Conteúdo:** Wikipedia processada versão 2
- **Arquivos:** 0 (diretório vazio)
- **Tamanho Total:** 0 MB
- **Status:** 📦 Diretório vazio/descartado

#### `planalto_clean/`
- **Conteúdo:** Legislação brasileira (Código Civil, leis, decretos)
- **Arquivos:** 18 arquivos
- **Tamanho Total:** 4.74 MB
- **Fonte:** planalto.gov.br scraped via scripts Python
- **Conteúdo Detalhado:**
  - `CDC.txt` (86 KB) - Código de Defesa do Consumidor
  - `CLT.txt` (1.4 MB) - Consolidação das Leis do Trabalho
  - `CODIGO_CIVIL.txt` (27 KB) - Código Civil Brasileiro
  - `CODIGO_PENAL.txt` (265 KB) - Código Penal
  - `CONSTITUICAO_FEDERAL.txt` (63 KB) - Constituição Federal 1988
  - `CPC.txt` (608 KB) - Código de Processo Civil
  - `CPP.txt` (394 KB) - Código de Processo Penal
  - `CTN.txt` (105 KB) - Código Tributário Nacional
  - `LEI_ANTICORRUPCAO.txt` (35 KB) - Lei de Anticorrupção
  - `LEI_FALENCIAS.txt` (270 KB) - Lei de Falências e Recuperação
  - `LEI_INQUILINATO.txt` (63 KB) - Lei do Inquilinato
  - `LEI_LICITACOES_1993.txt` (209 KB) - Lei de Licitações 1993
  - `LGPD.txt` (110 KB) - Lei Geral de Proteção de Dados
  - `MARCO_CIVIL_INTERNET.txt` (45 KB) - Marco Civil da Internet
  - `NOVA_LEI_LICITACOES.txt` (279 KB) - Nova Lei de Licitações 2021
  - `CODIGO_CIVIL.html` (976 KB) - Backup em HTML
  - `CODIGO_CIVIL_best_attempt.txt` (387 B) - Debug/tentativa
  - `CODIGO_CIVIL_debug.txt` (387 B) - Debug/tentativa
- **Estrutura:** Preserva artigos, incisos, parágrafos numerados
- **Qualidade:** Alta, textos oficiais brasileiros
- **Usado Em:** v3, v15 training
- **Status:** ✅ Ativo, base jurídica completa

#### `wikisource_clean/`
- **Conteúdo:** Obras literárias de domínio público
- **Arquivos:** 11 arquivos
- **Tamanho Total:** 178.47 MB
- **Fonte:** Wikisource PT-BR
- **Exemplos:** Machado de Assis, Aluísio Azevedo, Cruz e Sousa, etc
- **Linguagem:** Português clássico + moderno
- **Usado Em:** v3, v15 training
- **Status:** ✅ Ativo, qualidade literária

#### `wikibooks_clean/`
- **Conteúdo:** Conteúdo educacional (manuais, tutoriais)
- **Arquivos:** 2 arquivos
- **Tamanho Total:** 29.68 MB
- **Fonte:** Wikibooks PT-BR
- **Tópicos:** Programação, matemática, história, idiomas, etc
- **Usado Em:** v3, v15 training
- **Status:** ✅ Ativo, conteúdo técnico

#### `wikinews_clean/`
- **Conteúdo:** Notícias de arquivo (2005-2020)
- **Arquivos:** 4 arquivos
- **Tamanho Total:** 57.36 MB
- **Fonte:** Wikinews PT-BR
- **Utilidade:** Linguagem natural contemporânea, eventos históricos, atualidades
- **Usado Em:** v3, v15 training
- **Status:** ✅ Ativo, linguagem atual

#### `v15_clean/`
- **Conteúdo:** Corpus ultra-limpo unificado v15
- **Arquivos:** 33 arquivos
- **Tamanho Total:** 1.2 GB
- **Fontes:** Wikipedia + Planalto + Wikisource + Wikibooks + Wikinews
- **Qualidade:** Altíssima (muito filtrado, agressivo)
- **Características:** 
  - Remoção agressiva de stubs
  - Filtro por comprimento mínimo
  - Deduplicação aplicada
  - Normalização intensa
- **Usado Em:** v15 training (mais recente)
- **Status:** ✅ Ativo, melhor qualidade conhecida

### Diretórios de Tokenização

#### `tokenizer/` (legacy)
- **Conteúdo:** Vocabulário antigo (v1/v2)
- **Arquivos:** 0 (diretório vazio)
- **Tamanho:** 0 MB
- **Status:** 📦 Descartado (diretório vazio)

#### `tokenizer_v2/`
- **Conteúdo:** Vocabulário BPE 32k para v2
- **Arquivo:** `tokenizer.json`
- **Tamanho:** 1.25 MB
- **Vocab Size:** 32.000 tokens BPE
- **Baseado Em:** Wikipedia v2 limpa
- **Treinado Com:** build_corpus_v2 dataset
- **Status:** ✅ Funcional, legacy

#### `tokenizer_v3/`
- **Conteúdo:** Vocabulário BPE 32k para v3
- **Arquivo:** `tokenizer.json`
- **Tamanho:** 1.31 MB
- **Vocab Size:** 32.000 tokens BPE
- **Baseado Em:** Wikipedia + Planalto + Wikisource + Wikibooks
- **Treinado Com:** build_corpus_v3 (multi-source)
- **Status:** ✅ Funcional, ativo

#### `tokenizer_v12/`
- **Conteúdo:** Vocabulário BPE 32k para v12
- **Arquivo:** `tokenizer.json`
- **Tamanho:** 1.33 MB
- **Vocab Size:** 32.000 tokens BPE
- **Baseado Em:** Multi-corpus experimental
- **Status:** 📦 Archive (experimental)

#### `tokenizer_v15/`
- **Conteúdo:** Vocabulário BPE 32k para v15 (ultra-clean)
- **Arquivo:** `tokenizer.json`
- **Tamanho:** 1.27 MB
- **Vocab Size:** 32.000 tokens BPE
- **Baseado Em:** v15_clean (ultra-clean, multi-source)
- **Treinado Com:** build_corpus_v15 (mais rigoroso)
- **Qualidade:** Mais específico, menos ruído
- **Status:** ✅ Funcional, mais recente

### Diretórios de Dataset Tokenizado

#### `tokenized_v2/`
- **Conteúdo:** Dataset tokenizado para v2
- **Arquivo:** `train.bin`
- **Formato:** Binário (tokens u16, little-endian)
- **Tamanho:** 1.10 GB
- **Tokens Aproximados:** ~550M tokens (estimado)
- **Seq Len:** 256-512
- **Status:** ✅ Funcional, legacy

#### `tokenized_v3/`
- **Conteúdo:** Dataset tokenizado para v3
- **Arquivo:** `train.bin`
- **Formato:** Binário (tokens u16, little-endian)
- **Tamanho:** 88.63 MB
- **Tokens Aproximados:** ~44M tokens (estimado)
- **Seq Len:** 256-2048
- **Status:** ✅ Funcional, em uso

#### `tokenized_v12/`
- **Conteúdo:** Dataset tokenizado para v12
- **Arquivo:** `train.bin`
- **Tamanho:** 39.04 MB
- **Status:** 📦 Archive (experimental)

#### `tokenized_v15/`
- **Conteúdo:** Dataset tokenizado para v15 (ultra-clean)
- **Arquivo:** `train.bin`
- **Formato:** Binário (tokens u16, little-endian)
- **Tamanho:** 555.39 MB
- **Tokens Aproximados:** ~277M tokens (estimado)
- **Qualidade:** Altíssima (muito filtrado, deduplicated)
- **Status:** ✅ Funcional, mais recente

### Diretórios Processados (Legacy)

#### `processed/`
- **Subdiretórios:**
  - `corpus/` - Wikipedia parseada antiga (132 arquivos, 3.88 GB)
  - `tokenized/` - Dataset antigo (legacy)
- **Tamanho Total:** 3.88 GB
- **Status:** 📦 Archive/Backup

### Corpus Proprietário

#### `sovereign/`
- **Conteúdo:** Corpus proprietário com múltiplas versões acumuladas
- **Tamanho Total:** 7.6 GB (10 arquivos)
- **Propósito:** Acumular versões experimentais para comparação e análise
- **Arquivos Detalhados:**
  - `corpus_v3.txt` (1.88 GB) - Versão v3 original, base histórica
  - `corpus_v15_base.txt` (1.17 GB) - Versão v15 preparada para treinamento
  - `corpus_v14_clean.txt` (968 MB) - Versão v14 limpa
  - `corpus_v13_generous.txt` (1.05 GB) - Versão v13 mais permissiva/menos agressiva
  - `corpus_v11_brasil.txt` (273 MB) - Versão v11, foco Brasil only
  - `wiki_brasil.txt` (1.92 GB) - Wikipedia Brasil limpo unificado
  - `corpus_v12_ultra.txt` (88 MB) - Versão v12 ultra-limpada (muito pequeno)
  - `corpus_sample.txt` (196 MB) - Amostra para testes e validações
  - `leis.txt` (2.5 MB) - Textos jurídicos especializados
  - `dedup_v15.sqlite` (3.6 MB) - Database SQLite de deduplicação v15
- **Estratégia:** Cada versão reflete diferentes limites de limpeza/filtro agressivo
- **Uso:** Experimentos v11-v15, análise de qualidade de corpus, comparação entre versões
- **Status:** ✅ Ativo, múltiplas versões para análise e treinamento

---

## 📊 Resumo Completo de Arquivos

### Arquivos Rust (`src/`)

| Arquivo | LOC | Componente | Status |
|---------|-----|-----------|--------|
| main.rs | 799 | CLI, funções principais | ✅ |
| model/config.rs | 124 | RWKVConfig, TrainingConfig | ✅ |
| model/rwkv.rs | 400+ | RWKV, RWKVBlock, TimeMixing, ChannelMixing | ✅ |
| model/trainer.rs | 131 | Trainer, train_step, loss | ✅ |
| model/adapters.rs | 50+ | OptimizerAdaptor | ✅ |
| model/mod.rs | 10 | Re-exports | ✅ |
| data/wiki_parser.rs | 200+ | WikiStreamParser, WikiArticle | ✅ |
| data/cleaner.rs | 150+ | WikiCleaner, regex patterns | ✅ |
| data/dataset.rs | 155 | MmapDataset, DataLoader, Writer | ✅ |
| data/mod.rs | 10 | Re-exports | ✅ |
| tokenizer/bpe.rs | 500+ | BPETokenizer, BPETrainer, BPEVocab | ✅ |
| tokenizer/normalize.rs | 100+ | PTBRNormalizer | ✅ |
| tokenizer/mod.rs | 10 | Re-exports | ✅ |
| **TOTAL** | **~2700** | **13 arquivos** | **✅ Completo** |

### Scripts Python (`scripts/`)

| Script | Propósito | Versão | Status |
|--------|-----------|--------|--------|
| audit_sources_v15.py | Auditoria de fontes | v15 | ✅ |
| build_corpus_v15_stream.py | Build corpus v15 | v15 | ✅ |
| build_corpus_v15_v2_clean.py | Build alternativo v15 | v15 | 🔄 |
| clean_sources_v15.py | Limpeza v15 | v15 | ✅ |
| clean_sovereign.py | Corpus soberano | - | ✅ |
| extract_chroma.py | Extração de features | - | 🔄 |
| filter_v11_brasil_only.py | Filtro Brasil | v11 | 📦 |
| filter_v12_ultra_clean.py | Filtro ultra-clean | v12 | 📦 |
| filter_v13_keep_more.py | Filtro menos agressivo | v13 | 🔄 |
| filter_v14_quick_clean.py | Filtro rápido | v14 | 🔄 |
| fix_newlines_to_lf.py | Normalizar quebras | - | ✅ |
| planalto_codigo_civil_v3.py | Legislação v3 | v3 | ✅ |
| planalto_resolve_codigo_civil.py | CC parsing v1 | v1 | 📦 |
| planalto_resolve_codigo_civil_v2.py | CC parsing v2 | v2 | ✅ |
| scrape_planalto_fix_frames.py | Scraper frames | - | 🔄 |
| scrape_planalto_seeds.py | Scraper v1 | v1 | 📦 |
| scrape_planalto_seeds_v2.py | Scraper v2 | v2 | ✅ |
| unify_corpus.py | Unificar fontes | - | ✅ |
| **TOTAL** | **18 scripts** | **Multi-versão** | **✅ Completo** |

### Estrutura de Dados - Resumo Completo

| Diretório | Arquivos | Tamanho | Tipo | Status |
|-----------|----------|--------|------|--------|
| dumps/ | 3 | 130.06 MB | Wikipedia BZ2 | ✅ Ativo |
| raw/ | 1 | 2.4 GB | Dados brutos | ⚠️ Intermediário |
| wiki_clean/ | 132 | 2.38 GB | Wikipedia TXT | ✅ Ativo |
| wiki_processed_v2/ | 0 | 0 MB | - | 📦 Vazio |
| planalto_clean/ | 18 | 4.74 MB | Legislação | ✅ Ativo |
| wikisource_clean/ | 11 | 178.47 MB | Literatura | ✅ Ativo |
| wikibooks_clean/ | 2 | 29.68 MB | Educacional | ✅ Ativo |
| wikinews_clean/ | 4 | 57.36 MB | Notícias | ✅ Ativo |
| v15_clean/ | 33 | 1.2 GB | Ultra-clean | ✅ Ativo |
| processed/ | 132 | 3.88 GB | Legacy | 📦 Archive |
| tokenizer/ | 0 | 0 MB | - | 📦 Vazio |
| tokenizer_v2/ | 1 | 1.25 MB | BPE Vocab | ✅ Funcional |
| tokenizer_v3/ | 1 | 1.31 MB | BPE Vocab | ✅ Ativo |
| tokenizer_v12/ | 1 | 1.33 MB | BPE Vocab | 📦 Archive |
| tokenizer_v15/ | 1 | 1.27 MB | BPE Vocab | ✅ Ativo |
| tokenized_v2/ | 1 | 1.1 GB | Dataset BIN | ✅ Funcional |
| tokenized_v3/ | 1 | 88.63 MB | Dataset BIN | ✅ Ativo |
| tokenized_v12/ | 1 | 39.04 MB | Dataset BIN | 📦 Archive |
| tokenized_v15/ | 1 | 555.39 MB | Dataset BIN | ✅ Ativo |
| sovereign/ | 10 | 7.6 GB | Multi-version | ✅ Ativo |
| **TOTAL** | **363** | **~23 GB** | **Multi-type** | **✅ Completo** |

---

## �📝 Autores e Licença

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
