# ptbr-llm: Diretrizes de Engenharia de Elite

> Sistema RWKV em Rust com Burn. Cada regra existe para resolver um problema real.
> Se uma regra não previne bug, não acelera execução ou não facilita manutenção — ela não pertence aqui.

---

## 1. Zero Placebo: Robustez Não-Negociável

### 1.1 Tratamento de Erros

- **Proibido `unwrap()` e `expect()` em código de produção.**
  Use `Result<T, E>` com `thiserror` para erros tipados do domínio e `anyhow` apenas em binários/scripts.
  Único caso tolerado: testes unitários e invariantes matemáticas comprovadas (documente com `// SAFETY:`).

- **Erros são dados, não strings.**
  Todo módulo expõe seu próprio `enum Error` derivando `thiserror::Error`.
  Mensagens de erro devem conter contexto suficiente para reproduzir o problema sem debugger.

```rust
// ❌ Errado — sem contexto, sem tipo
fn load_model(path: &str) -> Result<Model, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    // ...
}

// ✅ Correto — erro tipado com contexto
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Falha ao carregar modelo de '{path}': {source}")]
    LoadFailed {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Checkpoint corrompido: hash esperado {expected}, obtido {actual}")]
    CorruptedCheckpoint {
        expected: String,
        actual: String,
    },
    #[error("Dimensão incompatível: esperado {expected}, obtido {actual}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
}

fn load_model(path: &Path) -> Result<Model, ModelError> {
    let data = std::fs::read(path).map_err(|source| ModelError::LoadFailed {
        path: path.to_path_buf(),
        source,
    })?;
    // ...
}
```

### 1.2 Tipagem Forte via NewType Pattern

- Nunca represente IDs, tokens, dimensões ou índices como `usize` ou `String` soltos.
- Confusão de argumentos é bug em tempo de compilação, não em tempo de execução.

```rust
// ❌ Errado — qual usize é qual?
fn embed(token: usize, vocab_size: usize, dim: usize) -> Vec<f32>

// ✅ Correto — impossível confundir argumentos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

#[derive(Debug, Clone, Copy)]
pub struct VocabSize(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct EmbedDim(pub usize);

fn embed(token: TokenId, vocab_size: VocabSize, dim: EmbedDim) -> Vec<f32>
```

### 1.3 Assertions Estruturais

- Validações de shape, dimensão e configuração acontecem **na construção**, não no uso.
- Use o padrão "parse, don't validate".

```rust
// ❌ Errado — valida tarde demais
impl<B: Backend> RwkvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        assert_eq!(x.dims()[1], self.d_model); // panic em produção
        // ...
    }
}

// ✅ Correto — valida na construção
pub struct ValidatedInput<B: Backend> {
    tensor: Tensor<B, 2>,
    d_model: usize,
}

impl<B: Backend> ValidatedInput<B> {
    pub fn new(tensor: Tensor<B, 2>, expected_dim: usize) -> Result<Self, ModelError> {
        let actual = tensor.dims()[1];
        if actual != expected_dim {
            return Err(ModelError::DimensionMismatch {
                expected: expected_dim,
                actual,
            });
        }
        Ok(Self { tensor, d_model: expected_dim })
    }
}
```

---

## 2. Performance com Propósito

### 2.1 Async com Critério

- LLMs são I/O bound (carregar pesos) e CPU/GPU bound (inferência).
- Nunca bloqueie o runtime async com operações de tensor.

```rust
// ❌ Errado — bloqueia o runtime async
async fn generate(prompt: &str) -> Result<String> {
    let output = model.forward(tensor); // operação pesada no runtime async
    Ok(decode(output))
}

// ✅ Correto — compute pesado em thread dedicada
async fn generate(prompt: &str) -> Result<String> {
    let tensor = tokenize(prompt).await?;
    let output = tokio::task::spawn_blocking(move || {
        model.forward(tensor)
    }).await??;
    Ok(decode(output))
}
```

### 2.2 Zero-Copy por Padrão

```rust
// ❌ Alocação desnecessária
fn process(input: String) -> String {
    if input.is_ascii() {
        input.to_lowercase() // aloca nova String sempre
    } else {
        input
    }
}

// ✅ Zero-copy quando possível
fn process<'a>(input: &'a str) -> Cow<'a, str> {
    if input.chars().any(|c| c.is_uppercase()) {
        Cow::Owned(input.to_lowercase())
    } else {
        Cow::Borrowed(input)
    }
}
```

- Use `bytes::Bytes` para buffers de rede e modelo.
- Use `memmap2` para carregar pesos > 1GB — não carregue 7GB de pesos na RAM de uma vez.

```rust
// ✅ Memory-mapped file para pesos grandes
use memmap2::MmapOptions;

fn load_weights(path: &Path) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap)
}
```

### 2.3 Benchmarks são Obrigatórios

- Toda claim de performance deve ter benchmark com `criterion`.
- Sem número medido, não é otimização — é superstição.

```rust
// benches/inference_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_forward_pass(c: &mut Criterion) {
    let model = setup_model();
    let input = setup_input();

    c.bench_function("rwkv_forward_1024_tokens", |b| {
        b.iter(|| model.forward(input.clone()))
    });
}

fn bench_tokenization(c: &mut Criterion) {
    let tokenizer = setup_tokenizer();
    let text = "Um texto de exemplo para benchmark.";

    c.bench_function("tokenize_short_text", |b| {
        b.iter(|| tokenizer.encode(text))
    });
}

criterion_group!(benches, bench_forward_pass, bench_tokenization);
criterion_main!(benches);
```

### 2.4 Alocações Conscientes

```rust
// ❌ Errado — realoca a cada push
fn collect_tokens(text: &str) -> Vec<TokenId> {
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        tokens.push(tokenize_word(word));
    }
    tokens
}

// ✅ Correto — pré-aloca com estimativa
fn collect_tokens(text: &str) -> Vec<TokenId> {
    let estimated = text.len() / 4; // ~4 chars por token em média
    let mut tokens = Vec::with_capacity(estimated);
    for word in text.split_whitespace() {
        tokens.push(tokenize_word(word));
    }
    tokens
}
```

---

## 3. Arquitetura Antirrecuo

### 3.1 Trait-Based Design

- Abstraia via trait qualquer componente que possa mudar: modelo, tokenizer, backend, sampler.
- Regra: se existe chance > 10% de trocar a implementação em 6 meses, é trait.

```rust
/// Geração de texto a partir de prompt
pub trait TextGenerator: Send + Sync {
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<GenerationOutput>;
    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<TokenId>> + '_>>;
}

/// Tokenização bidirecional
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    fn vocab_size(&self) -> VocabSize;
}

/// Amostragem de próximo token a partir de logits
pub trait Sampler: Send + Sync {
    fn sample(&self, logits: &[f32], rng: &mut impl Rng) -> TokenId;
}

/// Configuração de geração
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop_tokens: Vec<TokenId>,
    pub seed: Option<u64>,
}
```

### 3.2 Testes como Contrato

- **Unitários:** Toda função pura tem teste. Sem exceção.
- **Integração:** Todo pipeline (tokenize → inference → decode) tem teste end-to-end.
- **Snapshot:** Para saídas determinísticas (com seed fixa), use `insta` para detectar regressões.
- **Property-based:** Para invariantes matemáticas, use `proptest`.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Teste unitário — função pura
    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // Teste unitário — estabilidade numérica
    #[test]
    fn test_softmax_handles_large_values() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        assert!(probs.iter().all(|p| p.is_finite()));
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    // Teste de integração — pipeline completo
    #[test]
    fn test_generate_produces_valid_output() {
        let model = TestModel::new(SEED);
        let tokenizer = TestTokenizer::new();
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        };

        let output = model.generate("Olá", &config).unwrap();
        assert!(!output.text.is_empty());
        assert!(output.tokens.len() <= 10);
    }

    // Snapshot test — detecta regressão
    #[test]
    fn test_deterministic_output() {
        let output = generate_with_seed("teste", 42);
        insta::assert_snapshot!(output);
    }
}
```

### 3.3 Módulos com Fronteira Clara

```
ptbr-llm/
├── crates/
│   ├── ptbr-core/          # Tipos fundamentais: TokenId, VocabSize, configs
│   │   └── src/
│   │       ├── types.rs     # NewTypes
│   │       ├── config.rs    # ModelConfig, TrainingConfig
│   │       └── error.rs     # Erros do domínio
│   │
│   ├── ptbr-tokenizer/     # Encoding/decoding de texto
│   │   └── src/
│   │       ├── lib.rs       # trait Tokenizer
│   │       ├── bpe.rs       # Implementação BPE
│   │       └── wordpiece.rs # Implementação WordPiece
│   │
│   ├── ptbr-model/         # RWKV forward pass, arquitetura
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── rwkv.rs      # Modelo RWKV principal
│   │       ├── blocks.rs    # Time-mixing, Channel-mixing
│   │       ├── state.rs     # RwkvState — cidadão de primeira classe
│   │       └── attention.rs # WKV kernel
│   │
│   ├── ptbr-sampling/      # Estratégias de amostragem
│   │   └── src/
│   │       ├── lib.rs       # trait Sampler
│   │       ├── temperature.rs
│   │       ├── top_k.rs
│   │       ├── top_p.rs
│   │       └── repetition.rs
│   │
│   ├── ptbr-training/      # Loop de treino, otimizadores
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── trainer.rs   # Training loop
│   │       ├── dataset.rs   # Data loading & batching
│   │       └── lr_schedule.rs
│   │
│   ├── ptbr-checkpoint/    # Serialização, conversão de formatos
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── save.rs
│   │       ├── load.rs
│   │       └── convert.rs   # PyTorch → Burn, safetensors
│   │
│   └── ptbr-serving/       # API, streaming, batching
│       └── src/
│           ├── lib.rs
│           ├── api.rs       # HTTP endpoints
│           ├── stream.rs    # SSE/WebSocket streaming
│           └── batch.rs     # Request batching
│
├── bins/
│   ├── train.rs             # Binary de treino
│   ├── infer.rs             # Binary de inferência
│   └── convert.rs           # Binary de conversão de checkpoints
│
├── benches/
│   ├── forward_pass.rs
│   ├── tokenization.rs
│   └── sampling.rs
│
└── tests/
    ├── integration/
    │   ├── full_pipeline.rs
    │   └── checkpoint_roundtrip.rs
    └── snapshots/           # Gerenciados por insta
```

### 3.4 Dependency Injection

```rust
// ❌ Errado — acoplamento direto
struct Pipeline {
    tokenizer: BpeTokenizer,        // tipo concreto
    model: RwkvModel<Wgpu>,         // backend hardcoded
    sampler: TemperatureSampler,    // implementação fixa
}

// ✅ Correto — injeção de dependência via generics
struct Pipeline<T, B, S>
where
    T: Tokenizer,
    B: Backend,
    S: Sampler,
{
    tokenizer: T,
    model: RwkvModel<B>,
    sampler: S,
}

// ✅ Alternativa — trait objects para flexibilidade em runtime
struct DynPipeline {
    tokenizer: Box<dyn Tokenizer>,
    sampler: Box<dyn Sampler>,
}
```

---

## 4. RWKV & Burn: Regras Específicas

### 4.1 Backend Agnosticism é Lei

- Toda função de modelo usa generics sobre `Backend`.
- O backend concreto (`Wgpu`, `NdArray`, `Cuda`) é escolhido **apenas** em `main.rs` ou no binário.

```rust
// ❌ Nunca no core
fn forward(x: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2>

// ✅ Sempre genérico para inferência
fn forward<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2>

// ✅ Quando precisa de gradientes (treino)
fn train_step<B: AutodiffBackend>(
    x: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> Tensor<B, 1>

// ✅ Ponto de entrada — único lugar que escolhe backend
fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type TrainBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let model = RwkvModel::<TrainBackend>::new(&config, &device);
    // ...
}
```

### 4.2 State do RWKV é Cidadão de Primeira Classe

- State **NUNCA** é `Clone` implícito — mutação acidental de estado é bug crítico que gera saída corrompida silenciosamente.
- State é sempre passado e retornado **explicitamente**: `fn forward(x, state) -> (output, new_state)`.
- State tem validação na construção e serialização com hash de integridade.

```rust
/// Estado do modelo RWKV — um por layer, contém time-mixing e channel-mixing state
pub struct RwkvState<B: Backend> {
    layers: Vec<LayerState<B>>,
    num_layers: usize,
    d_model: usize,
}

/// Estado de uma layer individual
pub struct LayerState<B: Backend> {
    /// Time-mixing state
    pub time_state: Tensor<B, 2>,
    /// Channel-mixing state  
    pub channel_state: Tensor<B, 2>,
}

// NÃO derive Clone — forçar explicitação
impl<B: Backend> RwkvState<B> {
    /// Cria estado inicial zerado com validação
    pub fn initial(num_layers: usize, d_model: usize, device: &B::Device) -> Result<Self> {
        if num_layers == 0 {
            return Err(StateError::InvalidLayers(num_layers));
        }
        if d_model == 0 {
            return Err(StateError::InvalidDimension(d_model));
        }

        let layers = (0..num_layers)
            .map(|_| LayerState {
                time_state: Tensor::zeros([1, d_model], device),
                channel_state: Tensor::zeros([1, d_model], device),
            })
            .collect();

        Ok(Self { layers, num_layers, d_model })
    }

    /// Validação de integridade
    pub fn validate(&self) -> Result<(), StateError> {
        if self.layers.len() != self.num_layers {
            return Err(StateError::LayerCountMismatch {
                expected: self.num_layers,
                actual: self.layers.len(),
            });
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let dims = layer.time_state.dims();
            if dims[1] != self.d_model {
                return Err(StateError::DimensionMismatch {
                    layer: i,
                    expected: self.d_model,
                    actual: dims[1],
                });
            }
        }
        Ok(())
    }

    /// Clone explícito com nome intencional
    pub fn deep_copy(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|l| LayerState {
                time_state: l.time_state.clone(),
                channel_state: l.channel_state.clone(),
            }).collect(),
            num_layers: self.num_layers,
            d_model: self.d_model,
        }
    }
}

// ✅ Forward SEMPRE retorna novo state
impl<B: Backend> RwkvModel<B> {
    pub fn forward(
        &self,
        token: TokenId,
        state: RwkvState<B>,
    ) -> Result<(Tensor<B, 1>, RwkvState<B>)> {
        state.validate()?;
        // ... forward pass ...
        Ok((logits, new_state))
    }
}
```

### 4.3 Reprodutibilidade Determinística

- **Toda** operação estocástica recebe `seed: u64` explícito.
- Essencial para debugar problemas de convergência (loss estagnado, NaN, etc.).

```rust
// ❌ Não-determinístico — impossível reproduzir
let weights = Tensor::<B, 2>::random([d_model, d_model], Distribution::Normal(0.0, 0.02), &device);

// ✅ Determinístico — reproduzível
fn init_weights<B: Backend>(
    shape: [usize; 2],
    seed: u64,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Usar seed para inicialização determinística
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..shape[0] * shape[1])
        .map(|_| {
            let normal = rand_distr::Normal::new(0.0, 0.02).unwrap();
            rng.sample(normal)
        })
        .collect();
    Tensor::from_floats(&data[..], device).reshape(shape)
}

// ✅ Config de treino sempre inclui seed
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub seed: u64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gradient_clip: Option<f32>,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    // ...
}
```

### 4.4 Estabilidade Numérica

```rust
/// Softmax numericamente estável — SEMPRE subtrair o máximo
fn stable_softmax<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1> {
    let max_val = logits.clone().max();
    let shifted = logits - max_val;
    let exp = shifted.exp();
    let sum = exp.clone().sum();
    exp / sum
}

/// Log-softmax numericamente estável
fn stable_log_softmax<B: Backend>(logits: Tensor<B, 1>) -> Tensor<B, 1> {
    let max_val = logits.clone().max();
    let shifted = logits - max_val.clone();
    let log_sum_exp = shifted.clone().exp().sum().log();
    shifted - log_sum_exp
}

/// LayerNorm com epsilon seguro
pub struct LayerNormConfig {
    pub d_model: usize,
    pub eps: f64, // mínimo 1e-6 para f32, 1e-8 para f64
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            d_model: 0,
            eps: 1e-5, // padrão seguro
        }
    }
}

/// Validação de loss antes de backprop
fn validate_loss<B: Backend>(loss: &Tensor<B, 1>) -> Result<(), TrainingError> {
    let loss_val: f32 = loss.clone().into_scalar().elem();
    if loss_val.is_nan() {
        return Err(TrainingError::NumericalInstability {
            kind: "NaN loss detected".into(),
            step: 0, // preenchido pelo caller
        });
    }
    if loss_val.is_infinite() {
        return Err(TrainingError::NumericalInstability {
            kind: "Infinite loss detected".into(),
            step: 0,
        });
    }
    if loss_val > 100.0 {
        tracing::warn!(loss = loss_val, "Loss anormalmente alto — possível instabilidade");
    }
    Ok(())
}

/// Gradient clipping — previne explosão de gradientes
fn clip_gradients<B: AutodiffBackend>(
    gradients: B::Gradients,
    max_norm: f32,
) -> B::Gradients {
    // Implementar gradient clipping por norma global
    // Log warning se clipping ativado > 50% dos steps
    gradients
}
```

### 4.5 Checkpointing & Serialização

```rust
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

/// Checkpoint completo — tudo necessário para retomar treino
#[derive(Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Step atual do treino
    pub step: u64,
    /// Configuração completa do modelo
    pub model_config: ModelConfig,
    /// Configuração completa do treino
    pub training_config: TrainingConfig,
    /// Learning rate atual
    pub current_lr: f64,
    /// Melhor loss observado
    pub best_loss: f32,
    /// Hash SHA256 do dataset para verificar consistência
    pub dataset_hash: String,
    /// Timestamp de criação
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Path dos pesos do modelo (salvo separadamente por tamanho)
    pub weights_path: PathBuf,
    /// Path do optimizer state
    pub optimizer_path: PathBuf,
}

impl TrainingCheckpoint {
    /// Salva checkpoint com verificação de integridade
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;

        let meta_path = dir.join("checkpoint.json");
        let meta_json = serde_json::to_string_pretty(self)?;
        std::fs::write(&meta_path, &meta_json)?;

        // Verificação de integridade
        let hash = Sha256::digest(meta_json.as_bytes());
        std::fs::write(dir.join("checkpoint.sha256"), format!("{:x}", hash))?;

        tracing::info!(step = self.step, path = ?dir, "Checkpoint salvo");
        Ok(())
    }

    /// Carrega checkpoint com verificação de integridade
    pub fn load(dir: &Path) -> Result<Self> {
        let meta_path = dir.join("checkpoint.json");
        let meta_json = std::fs::read_to_string(&meta_path)?;

        // Verificar integridade
        let expected_hash = std::fs::read_to_string(dir.join("checkpoint.sha256"))?;
        let actual_hash = format!("{:x}", Sha256::digest(meta_json.as_bytes()));
        if expected_hash.trim() != actual_hash {
            return Err(CheckpointError::CorruptedCheckpoint {
                expected: expected_hash,
                actual: actual_hash,
            }.into());
        }

        let checkpoint: Self = serde_json::from_str(&meta_json)?;
        tracing::info!(step = checkpoint.step, "Checkpoint carregado");
        Ok(checkpoint)
    }
}

/// Salvar pesos do modelo usando formato Burn nativo
fn save_model_weights<B: Backend>(
    model: &RwkvModel<B>,
    path: &Path,
) -> Result<()> {
    model.save_file(path, &burn::record::NamedMpkFileRecorder::default())?;
    Ok(())
}

/// Carregar pesos de formato PyTorch (.pth) e converter para Burn
fn load_from_pytorch<B: Backend>(
    path: &Path,
    config: &ModelConfig,
    device: &B::Device,
) -> Result<RwkvModel<B>> {
    // Converter pesos PyTorch → Burn record
    // Validar shapes contra config
    todo!("Implementar conversão PyTorch → Burn")
}
```

### 4.6 WKV Kernel — Coração do RWKV

```rust
/// WKV (Weighted Key-Value) — operação central do RWKV
/// Esta é a operação mais performance-critical do modelo.
///
/// Referência: https://github.com/BlinkDL/RWKV-LM
///
/// ARCH: O WKV substitui a atenção quadrática do Transformer por
/// uma recorrência linear O(T*D) em vez de O(T²*D).
pub fn wkv_forward<B: Backend>(
    receptance: Tensor<B, 2>,  // [batch, d_model]
    key: Tensor<B, 2>,         // [batch, d_model]
    value: Tensor<B, 2>,       // [batch, d_model]
    time_decay: Tensor<B, 1>,  // [d_model] — learned decay
    time_first: Tensor<B, 1>,  // [d_model] — learned bonus
    state: Tensor<B, 2>,       // [batch, d_model] — running state
) -> Result<(Tensor<B, 2>, Tensor<B, 2>)> {
    // Validação de shapes
    let [batch, d_model] = receptance.dims();
    debug_assert_eq!(key.dims(), [batch, d_model]);
    debug_assert_eq!(value.dims(), [batch, d_model]);

    // WKV computation
    // ...

    Ok((output, new_state))
}
```

---

## 5. Observabilidade e Debugging

### 5.1 Logging Estruturado com `tracing`

- **Nunca** use `println!` ou `eprintln!` em código de produção.
- **Nunca** use o crate `log` — use `tracing` para structured logging e spans.
- Todo span deve ter contexto suficiente para entender o que aconteceu sem debugger.

```rust
use tracing::{info, warn, error, instrument, debug};

/// Forward pass instrumentado
#[instrument(
    skip(self, token, state),
    fields(
        token_id = token.0,
        num_layers = self.config.num_layers,
        d_model = self.config.d_model,
    )
)]
pub fn forward<B: Backend>(
    &self,
    token: TokenId,
    state: RwkvState<B>,
) -> Result<(Tensor<B, 1>, RwkvState<B>)> {
    debug!("Iniciando forward pass");

    let (logits, new_state) = self.inner_forward(token, state)?;

    debug!(
        logits_shape = ?logits.dims(),
        "Forward pass concluído"
    );

    Ok((logits, new_state))
}

/// Training step instrumentado
#[instrument(skip(self, batch), fields(step = self.current_step))]
pub fn train_step<B: AutodiffBackend>(
    &mut self,
    batch: &TrainingBatch<B>,
) -> Result<f32> {
    let loss = self.compute_loss(batch)?;
    let loss_val: f32 = loss.clone().into_scalar().elem();

    if loss_val.is_nan() {
        error!("NaN loss detectado!");
        return Err(TrainingError::NanLoss { step: self.current_step });
    }

    info!(
        loss = loss_val,
        lr = self.current_lr,
        tokens_per_sec = self.throughput(),
        "Step concluído"
    );

    Ok(loss_val)
}

/// Setup do subscriber de tracing
fn setup_tracing() -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env()
            .add_directive("ptbr_llm=debug".parse()?)
            .add_directive("burn=warn".parse()?))
        .with(fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true))
        .init();

    Ok(())
}
```

### 5.2 Métricas de Treino

```rust
/// Métricas coletadas durante treino
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingMetrics {
    pub step: u64,
    pub loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f64,
    pub tokens_per_second: f64,
    pub memory_used_mb: f64,
    pub elapsed_secs: f64,
}

/// Logger de métricas — extensível para TensorBoard, W&B, etc.
pub trait MetricsLogger: Send + Sync {
    fn log(&self, metrics: &TrainingMetrics);
    fn flush(&self) -> Result<()>;
}

/// Implementação CSV simples
pub struct CsvMetricsLogger {
    writer: std::sync::Mutex<csv::Writer<std::fs::File>>,
}

impl MetricsLogger for CsvMetricsLogger {
    fn log(&self, metrics: &TrainingMetrics) {
        let mut writer = self.writer.lock().unwrap();
        writer.serialize(metrics).ok();
    }

    fn flush(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.flush()?;
        Ok(())
    }
}

/// Detector de anomalias no treino
pub struct TrainingMonitor {
    loss_history: Vec<f32>,
    grad_norm_history: Vec<f32>,
    window_size: usize,
}

impl TrainingMonitor {
    pub fn check(&mut self, metrics: &TrainingMetrics) {
        self.loss_history.push(metrics.loss);
        self.grad_norm_history.push(metrics.gradient_norm);

        // Detectar loss estagnado
        if self.loss_history.len() >= self.window_size {
            let recent = &self.loss_history[self.loss_history.len() - self.window_size..];
            let variance: f32 = statistical_variance(recent);
            if variance < 1e-6 {
                warn!(
                    loss = metrics.loss,
                    variance = variance,
                    "Loss estagnado há {} steps", self.window_size
                );
            }
        }

        // Detectar gradient explosion
        if metrics.gradient_norm > 100.0 {
            warn!(
                grad_norm = metrics.gradient_norm,
                step = metrics.step,
                "Gradient norm anormalmente alto — aplicar clipping"
            );
        }

        // Detectar loss spike
        if self.loss_history.len() > 1 {
            let prev = self.loss_history[self.loss_history.len() - 2];
            if metrics.loss > prev * 3.0 {
                warn!(
                    current_loss = metrics.loss,
                    previous_loss = prev,
                    "Loss spike detectado (3x aumento)"
                );
            }
        }
    }
}
```

### 5.3 Panics são Bugs

- Se o sistema panicou em produção, é um **bug de design**.
- Todo panic deve gerar issue com reprodução obrigatória.
- Use `std::panic::catch_unwind` em boundaries de API para converter panics em erros.

```rust
/// Boundary de API — converte panic em erro HTTP 500
async fn api_boundary<F, T>(f: F) -> Result<T, ApiError>
where
    F: FnOnce() -> Result<T> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => Err(ApiError::Internal(e.to_string())),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");

            error!(panic = msg, "PANIC em produção — isto é um BUG");
            Err(ApiError::InternalPanic(msg.to_string()))
        }
    }
}
```

---

## 6. Convenções de Código

### 6.1 Documentação

```rust
//! # ptbr-model
//!
//! Implementação do modelo RWKV para inferência e treino.
//!
//! ## Responsabilidade
//! - Forward pass do RWKV (v5/v6)
//! - Gerenciamento de state
//! - WKV kernel
//!
//! ## Fronteiras
//! - NÃO faz tokenização (use `ptbr-tokenizer`)
//! - NÃO faz sampling (use `ptbr-sampling`)
//! - NÃO faz I/O de rede (use `ptbr-serving`)

/// Executa o forward pass completo do modelo RWKV.
///
/// # Argumentos
/// - `token` - ID do token de entrada
/// - `state` - Estado recorrente das layers anteriores
///
/// # Retorno
/// Tupla `(logits, new_state)` onde:
/// - `logits` tem shape `[vocab_size]`
/// - `new_state` é o estado atualizado para o próximo token
///
/// # Erros
/// - `ModelError::DimensionMismatch` se o state tem dimensões incompatíveis
/// - `ModelError::NumericalInstability` se NaN detectado internamente
///
/// # Exemplo
/// ```rust
/// let model = RwkvModel::<MyBackend>::new(&config, &device)?;
/// let state = RwkvState::initial(config.num_layers, config.d_model, &device)?;
/// let (logits, new_state) = model.forward(TokenId(42), state)?;
/// ```
pub fn forward<B: Backend>(
    &self,
    token: TokenId,
    state: RwkvState<B>,
) -> Result<(Tensor<B, 1>, RwkvState<B>)> {
    // ...
}
```

### 6.2 Naming Conventions

```rust
// ✅ Tensors: nomes do paper, não genéricos
let receptance = linear_r.forward(x.clone());  // não "r" ou "tensor1"
let key = linear_k.forward(x.clone());          // não "k" ou "input2"
let value = linear_v.forward(x.clone());        // não "v" ou "data"
let time_decay = self.w.clone();                 // não "decay" ou "param1"
let logits = head.forward(hidden);               // não "output" ou "result"
let attn_weights = receptance * key;             // não "weights" ou "w"

// ✅ Dimensões: constantes nomeadas, NUNCA magic numbers
const D_MODEL: usize = 768;
const N_LAYERS: usize = 12;
const VOCAB_SIZE: usize = 50277;
const CTX_LEN: usize = 1024;

// ❌ Magic numbers
let tensor = Tensor::zeros([1, 768, 12]);

// ✅ Constantes nomeadas
let tensor = Tensor::zeros([batch_size, D_MODEL, N_LAYERS]);

// ✅ Ou melhor ainda, via config
let tensor = Tensor::zeros([
    config.batch_size,
    config.d_model,
    config.num_layers,
]);
```

### 6.3 Estrutura de Arquivos

```rust
// Cada arquivo de módulo segue esta estrutura:

// 1. Module-level docs
//! Descrição do módulo

// 2. Imports agrupados
use std::path::Path;

use burn::prelude::*;
use thiserror::Error;

use crate::types::{TokenId, VocabSize};

// 3. Errors do módulo
#[derive(Debug, Error)]
pub enum BlockError { /* ... */ }

// 4. Types e structs
pub struct TimeMixing<B: Backend> { /* ... */ }

// 5. Implementações
impl<B: Backend> TimeMixing<B> { /* ... */ }

// 6. Trait implementations
impl<B: Backend> Module<B> for TimeMixing<B> { /* ... */ }

// 7. Funções auxiliares privadas
fn compute_wkv<B: Backend>(/* ... */) -> Tensor<B, 2> { /* ... */ }

// 8. Testes
#[cfg(test)]
mod tests { /* ... */ }
```

### 6.4 Clippy & Formatting

```toml
# Cargo.toml do workspace
[workspace.lints.clippy]
all = "deny"
pedantic = "deny"
nursery = "warn"
# Exceções justificadas:
module_name_repetitions = "allow"  # ptbr_model::ModelConfig é ok
cast_possible_truncation = "allow" # controlado por NewTypes
cast_precision_loss = "allow"      # inevitável em f32 math

[workspace.lints.rust]
unsafe_code = "deny"               # exceto onde SAFETY doc existe
missing_docs = "warn"
```

```yaml
# CI — zero warnings
- name: Clippy
  run: cargo clippy --workspace --all-targets -- -D warnings

- name: Format
  run: cargo fmt --all -- --check

- name: Tests
  run: cargo test --workspace

- name: Doc tests
  run: cargo test --doc --workspace
```

---

## 7. Regras de Ouro (Resumo Executivo)

| # | Regra | Consequência de Violar |
|---|-------|----------------------|
| 1 | Sem `unwrap()` em prod | Crash em produção |
| 2 | NewTypes para IDs/dims | Bug silencioso de lógica |
| 3 | Backend sempre genérico | Lock-in de hardware |
| 4 | State explícito `(out, state)` | Corrupção silenciosa de state |
| 5 | Seed em toda operação estocástica | Bug irreproducível |
| 6 | Validar loss antes de backprop | Treino diverge silenciosamente |
| 7 | `tracing`, não `println!` | Zero visibilidade em prod |
| 8 | Benchmark toda otimização | Otimização placebo |
| 9 | Checkpoint com hash | Corrupção silenciosa de modelo |
| 10 | Testes unitário + integração + snapshot | Regressão não detectada |