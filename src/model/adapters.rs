// src/model/adapters.rs

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use serde::{Serialize, Deserialize};

/// Domínios suportados para fine-tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    General,
    Legal,
    Financial,
    Tech,
    Medical,
}

impl Default for Domain {
    fn default() -> Self {
        Domain::General
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
}

impl<B: Backend> LoRAAdapter<B> {
    pub fn new(d_model: usize, rank: usize, scale: f32, device: &B::Device) -> Self {
        Self {
            down: LinearConfig::new(d_model, rank).with_bias(false).init(device),
            up: LinearConfig::new(rank, d_model).with_bias(false).init(device),
            scale,
            rank,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let adapter_out = self.up.forward(self.down.forward(x));
        adapter_out * self.scale
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Container de múltiplos adapters
#[derive(Module, Debug)]
pub struct DomainAdapterBank<B: Backend> {
    adapters: Vec<LoRAAdapter<B>>,
    
    #[module(skip)]
    active_idx: Option<usize>,
}

impl<B: Backend> DomainAdapterBank<B> {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active_idx: None,
        }
    }

    pub fn add_adapter(&mut self, adapter: LoRAAdapter<B>) {
        self.adapters.push(adapter);
    }

    pub fn set_active(&mut self, idx: usize) {
        if idx < self.adapters.len() {
            self.active_idx = Some(idx);
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, base_output: Tensor<B, 3>) -> Tensor<B, 3> {
        match self.active_idx {
            Some(idx) if idx < self.adapters.len() => {
                base_output + self.adapters[idx].forward(x)
            }
            _ => base_output,
        }
    }
}

impl<B: Backend> Default for DomainAdapterBank<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuração para fine-tuning vertical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainFineTuneConfig {
    pub domain: Domain,
    pub lora_rank: usize,
    pub lora_scale: f32,
    pub learning_rate: f64,
    pub epochs: usize,
    pub freeze_base: bool,
}

impl DomainFineTuneConfig {
    pub fn legal() -> Self {
        Self {
            domain: Domain::Legal,
            lora_rank: 8,
            lora_scale: 0.5,
            learning_rate: 1e-4,
            epochs: 3,
            freeze_base: true,
        }
    }

    pub fn medical() -> Self {
        Self {
            domain: Domain::Medical,
            lora_rank: 16,
            lora_scale: 0.5,
            learning_rate: 5e-5,
            epochs: 5,
            freeze_base: true,
        }
    }
}