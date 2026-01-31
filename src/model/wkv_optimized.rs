//! WKV Linear Attention - Versão CORRIGIDA FINAL
//! 
//! Implementação com:
//! - Decay aprendido (média de w)
//! - u bonus só para token atual
//! - Matmul para cumsum eficiente

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

/// WKV - Implementação Final Corrigida
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Clamp para estabilidade
    let k_safe = k.clamp(-10.0, 10.0);
    let w_safe = w.clamp(-5.0, -0.5);  // Sempre negativo
    let u_safe = u.clamp(-3.0, 3.0);   // Pequeno
    
    // Média do decay aprendido
    let w_mean: f32 = w_safe.mean().into_scalar().elem();
    let w_x100 = (w_mean * 100.0) as i32;
    
    // exp(k) para todos os tokens
    let k_exp = k_safe.exp();
    
    // Matriz de decay triangular
    let decay_data = get_cached_decay_matrix(seq_len, w_x100);
    let decay_td = TensorData::from(decay_data.as_slice());
    let decay: Tensor<B, 1> = Tensor::from_data(decay_td, &device);
    let decay = decay.reshape([seq_len, seq_len]);
    
    // Cumsum do passado: sum_{j<i} decay^(i-j) * exp(k_j) * v_j
    let wv = (k_exp.clone() * v.clone()).swap_dims(1, 2).reshape([batch_size * channels, seq_len]);
    let cum_wv = decay.clone().matmul(wv.transpose()).transpose()
        .reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // Cumsum dos pesos
    let kexp_flat = k_exp.clone().swap_dims(1, 2).reshape([batch_size * channels, seq_len]);
    let cum_kexp = decay.matmul(kexp_flat.transpose()).transpose()
        .reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // Token atual: exp(u + k_t) = exp(u) * exp(k_t)
    let u_exp = u_safe.exp().reshape([1, 1, channels]);
    let current_weight = k_exp * u_exp;
    let current_num = current_weight.clone() * v;
    
    // Output final
    let num = cum_wv + current_num;
    let den = cum_kexp + current_weight + 1e-6;
    (num / den).clamp(-20.0, 20.0)
}

pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_, channels] = k.dims();
    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;

    let k_safe = k.clamp(-10.0, 10.0);
    let w_exp = w.clamp(-5.0, -0.5).reshape([1, channels]).exp();
    let u_exp = u.clamp(-3.0, 3.0).reshape([1, channels]).exp();
    let k_exp = k_safe.exp();
    
    let current_weight = k_exp.clone() * u_exp;
    let current_num = current_weight.clone() * v.clone();
    
    let num = state_num.clone() + current_num;
    let den = state_den.clone() + current_weight + 1e-6;
    let output = (num / den).clamp(-20.0, 20.0);

    *state_num = state_num.clone() * w_exp.clone() + k_exp.clone() * v;
    *state_den = state_den.clone() * w_exp + k_exp;
    output
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