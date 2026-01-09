//! LoRA Adapters para Fine-Tuning Eficiente

#![allow(dead_code)]

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

/// Domínios suportados para fine-tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    General,
    Legal,
    Financial,
    Tech,
    Medical,
    Academic,
    News,
}

impl Default for Domain {
    fn default() -> Self {
        Domain::General
    }
}

impl Domain {
    pub fn recommended_config(&self) -> DomainFineTuneConfig {
        match self {
            Domain::General => DomainFineTuneConfig::default(),
            Domain::Legal => DomainFineTuneConfig::legal(),
            Domain::Financial => DomainFineTuneConfig::financial(),
            Domain::Tech => DomainFineTuneConfig::tech(),
            Domain::Medical => DomainFineTuneConfig::medical(),
            Domain::Academic => DomainFineTuneConfig::academic(),
            Domain::News => DomainFineTuneConfig::news(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Domain::General => "general",
            Domain::Legal => "legal",
            Domain::Financial => "financial",
            Domain::Tech => "tech",
            Domain::Medical => "medical",
            Domain::Academic => "academic",
            Domain::News => "news",
        }
    }
}

/// LoRA adapter para fine-tuning eficiente
#[derive(Module, Debug)]
pub struct LoRAAdapter<B: Backend> {
    down: Linear<B>,
    up: Linear<B>,

    #[module(skip)]
    scale: f32,
    #[module(skip)]
    rank: usize,
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> LoRAAdapter<B> {
    pub fn new(d_model: usize, rank: usize, scale: f32, device: &B::Device) -> Self {
        let down = LinearConfig::new(d_model, rank)
            .with_bias(false)
            .init(device);
        let up = LinearConfig::new(rank, d_model)
            .with_bias(false)
            .init(device);

        Self {
            down,
            up,
            scale,
            rank,
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let down_out = self.down.forward(x);
        let up_out = self.up.forward(down_out);
        up_out * self.scale
    }

    pub fn apply(&self, x: Tensor<B, 3>, base_output: Tensor<B, 3>) -> Tensor<B, 3> {
        base_output + self.forward(x)
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn num_parameters(&self) -> usize {
        2 * self.d_model * self.rank
    }
}

/// Container de múltiplos adapters
/// NOTA: Domain tracking é feito externamente para evitar bugs do Burn macro
#[derive(Module, Debug)]
pub struct DomainAdapterBank<B: Backend> {
    adapters: Vec<LoRAAdapter<B>>,

    #[module(skip)]
    active_idx: Option<usize>,
    #[module(skip)]
    num_adapters: usize,
}

impl<B: Backend> DomainAdapterBank<B> {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active_idx: None,
            num_adapters: 0,
        }
    }

    /// Adiciona um adapter ao banco, retorna o índice
    pub fn add_adapter(&mut self, adapter: LoRAAdapter<B>) -> usize {
        let idx = self.num_adapters;
        self.adapters.push(adapter);
        self.num_adapters += 1;
        idx
    }

    /// Define adapter ativo por índice
    pub fn set_active(&mut self, idx: usize) {
        if idx < self.adapters.len() {
            self.active_idx = Some(idx);
        }
    }

    /// Desativa todos os adapters
    pub fn deactivate(&mut self) {
        self.active_idx = None;
    }

    /// Forward com adapter ativo (se houver)
    pub fn forward(&self, x: Tensor<B, 3>, base_output: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.active_idx {
            Some(idx) if idx < self.adapters.len() => base_output + self.adapters[idx].forward(x),
            _ => base_output,
        }
    }

    /// Retorna índice ativo
    pub fn active_index(&self) -> Option<usize> {
        self.active_idx
    }

    pub fn len(&self) -> usize {
        self.adapters.len()
    }
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    pub fn total_parameters(&self) -> usize {
        self.adapters.iter().map(|a| a.num_parameters()).sum()
    }
}

impl<B: Backend> Default for DomainAdapterBank<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Gerenciador de domínios (separado do Module para evitar bugs do Burn)
#[derive(Debug, Clone, Default)]
pub struct DomainRegistry {
    domains: Vec<Domain>,
}

impl DomainRegistry {
    pub fn new() -> Self {
        Self {
            domains: Vec::new(),
        }
    }

    pub fn register(&mut self, domain: Domain) -> usize {
        let idx = self.domains.len();
        self.domains.push(domain);
        idx
    }

    pub fn get(&self, idx: usize) -> Option<Domain> {
        self.domains.get(idx).copied()
    }

    pub fn find(&self, domain: Domain) -> Option<usize> {
        self.domains.iter().position(|&d| d == domain)
    }

    pub fn all(&self) -> &[Domain] {
        &self.domains
    }
}

/// Configuração para fine-tuning vertical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainFineTuneConfig {
    pub domain: Domain,
    pub lora_rank: usize,
    pub lora_scale: f32,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub gradient_accumulation: usize,
    pub freeze_base: bool,
    pub warmup_ratio: f32,
}

impl Default for DomainFineTuneConfig {
    fn default() -> Self {
        Self {
            domain: Domain::General,
            lora_rank: 8,
            lora_scale: 1.0,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            epochs: 3,
            batch_size: 4,
            gradient_accumulation: 4,
            freeze_base: true,
            warmup_ratio: 0.1,
        }
    }
}

impl DomainFineTuneConfig {
    pub fn legal() -> Self {
        Self {
            domain: Domain::Legal,
            lora_rank: 16,
            lora_scale: 0.5,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            epochs: 5,
            batch_size: 2,
            gradient_accumulation: 8,
            freeze_base: true,
            warmup_ratio: 0.1,
        }
    }

    pub fn medical() -> Self {
        Self {
            domain: Domain::Medical,
            lora_rank: 32,
            lora_scale: 0.5,
            learning_rate: 3e-5,
            weight_decay: 0.01,
            epochs: 8,
            batch_size: 2,
            gradient_accumulation: 8,
            freeze_base: true,
            warmup_ratio: 0.15,
        }
    }

    pub fn financial() -> Self {
        Self {
            domain: Domain::Financial,
            lora_rank: 16,
            lora_scale: 0.5,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            epochs: 4,
            batch_size: 4,
            gradient_accumulation: 4,
            freeze_base: true,
            warmup_ratio: 0.1,
        }
    }

    pub fn tech() -> Self {
        Self {
            domain: Domain::Tech,
            lora_rank: 8,
            lora_scale: 0.8,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            epochs: 3,
            batch_size: 4,
            gradient_accumulation: 4,
            freeze_base: true,
            warmup_ratio: 0.05,
        }
    }

    pub fn academic() -> Self {
        Self {
            domain: Domain::Academic,
            lora_rank: 16,
            lora_scale: 0.6,
            learning_rate: 5e-5,
            weight_decay: 0.01,
            epochs: 5,
            batch_size: 2,
            gradient_accumulation: 8,
            freeze_base: true,
            warmup_ratio: 0.1,
        }
    }

    pub fn news() -> Self {
        Self {
            domain: Domain::News,
            lora_rank: 8,
            lora_scale: 0.7,
            learning_rate: 8e-5,
            weight_decay: 0.01,
            epochs: 3,
            batch_size: 4,
            gradient_accumulation: 4,
            freeze_base: true,
            warmup_ratio: 0.05,
        }
    }

    pub fn trainable_params(&self, d_model: usize, n_layers: usize) -> usize {
        let params_per_adapter = 2 * d_model * self.lora_rank;
        let adapters_per_layer = 2;
        n_layers * adapters_per_layer * params_per_adapter
    }

    pub fn estimated_memory(&self, d_model: usize, n_layers: usize) -> usize {
        let params = self.trainable_params(d_model, n_layers);
        params * 4 * 4
    }
}

/// Builder para criar adapters
pub struct LoRABuilder {
    d_model: usize,
    rank: usize,
    scale: f32,
}

impl LoRABuilder {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            rank: 8,
            scale: 1.0,
        }
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn from_config(d_model: usize, config: &DomainFineTuneConfig) -> Self {
        Self {
            d_model,
            rank: config.lora_rank,
            scale: config.lora_scale,
        }
    }

    pub fn build<B: Backend>(self, device: &B::Device) -> LoRAAdapter<B> {
        LoRAAdapter::new(self.d_model, self.rank, self.scale, device)
    }
}
