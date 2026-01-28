//! WKV Debug - Versão mínima para identificar segfault

use burn::tensor::{backend::Backend, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub detach_between_chunks: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self { chunk_size: 16, detach_between_chunks: true }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self { Self::default() }
    pub fn for_high_memory() -> Self { Self::default() }
}

/// DEBUG: Simplesmente retorna v (sem processamento)
/// Se ainda der segfault, o problema NÃO está no WKV
pub fn wkv_linear<B: Backend>(
    _k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    _w: Tensor<B, 1>,
    _u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    // Simplesmente retorna v - isso DEVE funcionar
    v
}

pub fn wkv_step<B: Backend>(
    _k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    _w: Tensor<B, 1>,
    _u: Tensor<B, 1>,
    _state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    v
}

pub fn init_state<B: Backend>(
    batch_size: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    (
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
    )
}

pub fn wkv_parallel_scan<B: Backend>(
    _k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    _w: Tensor<B, 1>,
    _u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    v
}