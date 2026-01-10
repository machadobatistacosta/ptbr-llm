// src/model/checkpoint.rs
//! Gradient Checkpointing para reduzir uso de memória

use burn::tensor::{backend::Backend, Tensor};
use std::cell::RefCell;

/// Wrapper para checkpointing de activations
pub struct CheckpointedActivation<B: Backend> {
    /// Função que recomputa a activation
    recompute_fn: Box<dyn Fn() -> Tensor<B, 3>>,
    /// Cache (Some = já computado, None = precisa recomputar)
    cached: RefCell<Option<Tensor<B, 3>>>,
}

impl<B: Backend> CheckpointedActivation<B> {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn() -> Tensor<B, 3> + 'static,
    {
        Self {
            recompute_fn: Box::new(f),
            cached: RefCell::new(None),
        }
    }

    pub fn get(&self) -> Tensor<B, 3> {
        let mut cache = self.cached.borrow_mut();
        if let Some(ref t) = *cache {
            t.clone()
        } else {
            let t = (self.recompute_fn)();
            *cache = Some(t.clone());
            t
        }
    }

    pub fn clear(&self) {
        *self.cached.borrow_mut() = None;
    }
}

/// Executa função com checkpointing
/// Durante forward: salva apenas input, não activations intermediárias
/// Durante backward: recomputa activations
pub fn checkpoint<B, F, T>(input: Tensor<B, 3>, f: F) -> Tensor<B, 3>
where
    B: Backend,
    F: Fn(Tensor<B, 3>) -> Tensor<B, 3>,
{
    // Em modo de treino, recomputa durante backward
    // Por enquanto, implementação simples que apenas executa
    // TODO: Integrar com Burn's autodiff quando suportado
    f(input)
}

/// Configuração de checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Ativa checkpointing
    pub enabled: bool,
    /// Checkpoint a cada N layers
    pub every_n_layers: usize,
    /// Layers específicas para checkpoint (override)
    pub checkpoint_layers: Option<Vec<usize>>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            every_n_layers: 2,
            checkpoint_layers: None,
        }
    }
}

impl CheckpointConfig {
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            every_n_layers: 1, // Checkpoint toda layer
            checkpoint_layers: None,
        }
    }

    pub fn balanced() -> Self {
        Self {
            enabled: true,
            every_n_layers: 2, // Checkpoint a cada 2 layers
            checkpoint_layers: None,
        }
    }

    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(ref layers) = self.checkpoint_layers {
            layers.contains(&layer_idx)
        } else {
            layer_idx % self.every_n_layers == 0
        }
    }
}