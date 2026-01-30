//! WKV Linear Attention - Versão SIMPLIFICADA ESTÁVEL
//! 
//! Usa matmul com matriz triangular de decay.
//! Decay usa MÉDIA do parâmetro aprendido w (compromisso entre hardcoded e per-channel)
//!
//! Esta versão é mais simples e estável no CUDA.

use burn::tensor::{backend::Backend, ElementConversion, Tensor, TensorData};
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;

// Cache global para matrizes de decay (thread-safe)
static DECAY_CACHE: Lazy<RwLock<HashMap<(usize, i32), Vec<f32>>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_fp32_accumulator: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32,
            use_fp32_accumulator: true,
        }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self {
        Self::default()
    }
    pub fn for_high_memory() -> Self {
        Self {
            chunk_size: 64,
            use_fp32_accumulator: true,
        }
    }
}

/// Obtém ou cria matriz de decay CACHEADA
fn get_cached_decay_matrix(seq_len: usize, w_mean_x100: i32) -> Vec<f32> {
    let key = (seq_len, w_mean_x100);
    
    // Tenta ler do cache primeiro
    {
        let cache = DECAY_CACHE.read().unwrap();
        if let Some(data) = cache.get(&key) {
            return data.clone();
        }
    }
    
    // Cria nova matriz
    let w_mean = w_mean_x100 as f32 / 100.0;
    let mut decay_data = vec![0.0f32; seq_len * seq_len];
    
    for i in 0..seq_len {
        for j in 0..i {  // j < i (triangular inferior, sem diagonal)
            let dist = (i - j) as f32;
            // decay = exp(w * dist), clamped para estabilidade
            decay_data[i * seq_len + j] = (w_mean * dist).max(-20.0).exp();
        }
    }
    
    // Salva no cache
    {
        let mut cache = DECAY_CACHE.write().unwrap();
        cache.insert(key, decay_data.clone());
    }
    
    decay_data
}

/// WKV Linear Attention - Versão Matmul com Decay Aprendido
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    w: Tensor<B, 1>,      // [channels] - learned decay (USADO!)
    u: Tensor<B, 1>,      // [channels] - learned bonus
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // ========================================
    // ESTABILIDADE NUMÉRICA
    // ========================================
    let k_safe = k.clamp(-15.0, 15.0);
    let u_safe = u.clamp(-15.0, 15.0);
    let w_safe = w.clamp(-8.0, -0.01);
    
    // ========================================
    // USA MÉDIA DO DECAY APRENDIDO
    // ========================================
    // Compromisso: não é per-channel exato, mas usa o valor aprendido
    let w_mean: f32 = w_safe.mean().into_scalar().elem();
    let w_mean_x100 = (w_mean * 100.0) as i32;
    
    // LogSumExp trick: k_centered = k - k_max
    let k_max = k_safe.clone().max_dim(1);
    let k_centered = k_safe - k_max.clone();
    let k_exp = k_centered.exp();
    
    // v ponderado por exp(k - k_max)
    let weighted_v = k_exp.clone() * v.clone();
    
    // ========================================
    // CONTRIBUIÇÃO DO TOKEN ATUAL
    // ========================================
    let u_exp = u_safe.clamp(-10.0, 10.0).exp().reshape([1, 1, channels]);
    let current_contrib = k_exp.clone() * u_exp.clone() * v.clone();
    let current_weight = k_exp.clone() * u_exp;
    
    // ========================================
    // MATRIZ DE DECAY (com valor aprendido)
    // ========================================
    let decay_data = get_cached_decay_matrix(seq_len, w_mean_x100);
    
    let decay_tensor_data = TensorData::from(decay_data.as_slice());
    let sum_matrix: Tensor<B, 1> = Tensor::from_data(decay_tensor_data, &device);
    let sum_matrix = sum_matrix.reshape([seq_len, seq_len]);
    
    // ========================================
    // CUMSUM VIA MATMUL
    // ========================================
    let wv_t = weighted_v.swap_dims(1, 2);
    let wv_flat = wv_t.reshape([batch_size * channels, seq_len]);
    let wv_flat_t = wv_flat.transpose();
    let cum_wv_t = sum_matrix.clone().matmul(wv_flat_t);
    let cum_wv = cum_wv_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    let kexp_t = k_exp.clone().swap_dims(1, 2);
    let kexp_flat = kexp_t.reshape([batch_size * channels, seq_len]);
    let kexp_flat_t = kexp_flat.transpose();
    let cum_kexp_t = sum_matrix.matmul(kexp_flat_t);
    let cum_kexp = cum_kexp_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // ========================================
    // OUTPUT FINAL
    // ========================================
    let numerator = cum_wv + current_contrib;
    let denominator = cum_kexp + current_weight + 1e-6;
    
    (numerator / denominator).clamp(-100.0, 100.0)
}

/// WKV step para inferência (token por token)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels]
    v: Tensor<B, 2>,      // [batch, channels]
    w: Tensor<B, 1>,      // [channels]
    u: Tensor<B, 1>,      // [channels]
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_batch_size, channels] = k.dims();

    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;

    let w_exp = w.clone().clamp(-8.0, -0.01).reshape([1, channels]).exp();
    let u_exp = u.clamp(-10.0, 10.0).reshape([1, channels]).exp();
    let k_exp = k.clone().clamp(-15.0, 15.0).exp();
    
    let uk_exp = k_exp.clone() * u_exp;

    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp + 1e-6;
    let output = (num / den).clamp(-100.0, 100.0);

    *state_num = state_num.clone() * w_exp.clone() + k_exp.clone() * v;
    *state_den = state_den.clone() * w_exp + k_exp;

    output
}

/// Inicializa estado para inferência
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

/// Alias para compatibilidade
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}