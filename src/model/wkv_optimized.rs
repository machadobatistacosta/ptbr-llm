//! WKV Linear Attention - DIAGNÓSTICO
//! 
//! VERSÃO SIMPLIFICADA: apenas retorna v diretamente
//! Se loss ficar ~11, o problema está no WKV
//! Se loss continuar ~39, o problema está em outro lugar

use burn::tensor::{backend::Backend, ElementConversion, Tensor, TensorData};
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;

static DECAY_CACHE: Lazy<RwLock<HashMap<(usize, i32), Vec<f32>>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_fp32_accumulator: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self { chunk_size: 32, use_fp32_accumulator: true }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self { Self::default() }
    pub fn for_high_memory() -> Self { Self { chunk_size: 64, use_fp32_accumulator: true } }
}

#[allow(dead_code)]
fn get_cached_decay_matrix(seq_len: usize, w_mean_x100: i32) -> Vec<f32> {
    let key = (seq_len, w_mean_x100);
    {
        let cache = DECAY_CACHE.read().unwrap();
        if let Some(data) = cache.get(&key) {
            return data.clone();
        }
    }
    
    let w_mean = w_mean_x100 as f32 / 100.0;
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..i {
            data[i * seq_len + j] = (w_mean * (i - j) as f32).max(-20.0).exp();
        }
    }
    
    let mut cache = DECAY_CACHE.write().unwrap();
    cache.insert(key, data.clone());
    data
}

/// WKV - DIAGNÓSTICO: apenas retorna v
/// 
/// Esta versão NÃO faz atenção, apenas passa v através.
/// Serve para diagnosticar se o problema está no WKV ou não.
pub fn wkv_linear<B: Backend>(
    _k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    _w: Tensor<B, 1>,
    _u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    // DIAGNÓSTICO: retorna v diretamente (sem atenção)
    // Se loss ficar ~11, problema está no WKV real
    // Se loss continuar ~39, problema está em outro lugar
    v
}

pub fn wkv_step<B: Backend>(
    _k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    _w: Tensor<B, 1>,
    _u: Tensor<B, 1>,
    _state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    // DIAGNÓSTICO: retorna v diretamente
    v
}

#[allow(dead_code)]
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
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}