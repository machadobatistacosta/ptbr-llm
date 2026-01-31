//! WKV Linear Attention - Versão CORRIGIDA
//! 
//! CORREÇÃO: u (bonus) só aplica ao token atual, não ao cumsum do passado
//! 
//! Fórmula RWKV correta:
//! wkv_t = (Σ_{i<t} exp(-(t-1-i)*w + k_i) * v_i + exp(u + k_t) * v_t) /
//!         (Σ_{i<t} exp(-(t-1-i)*w + k_i) + exp(u + k_t))

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

/// WKV - Versão CORRIGIDA
/// 
/// IMPORTANTE: u (bonus) só aplica ao token ATUAL (posição t=t),
/// NÃO aos tokens passados no cumsum!
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
    let w_safe = w.clamp(-5.0, -0.1);
    let u_safe = u.clamp(-5.0, 5.0);
    
    // Média do decay aprendido
    let w_mean: f32 = w_safe.mean().into_scalar().elem();
    let w_x100 = (w_mean * 100.0) as i32;
    
    // ========================================
    // CONTRIBUIÇÃO DOS TOKENS PASSADOS (cumsum com decay)
    // ========================================
    // Para i < t: weight = exp(-(t-1-i)*w + k_i)
    // Aproximação: exp(decay) * exp(k)
    
    let k_exp = k_safe.clone().exp();  // exp(k) para todos os tokens
    
    // Matriz de decay triangular
    let decay_data = get_cached_decay_matrix(seq_len, w_x100);
    let decay_td = TensorData::from(decay_data.as_slice());
    let decay: Tensor<B, 1> = Tensor::from_data(decay_td, &device);
    let decay = decay.reshape([seq_len, seq_len]);
    
    // weighted_v = exp(k) * v para todos os tokens
    let wv = (k_exp.clone() * v.clone()).swap_dims(1, 2).reshape([batch_size * channels, seq_len]);
    
    // cumsum com decay: soma dos tokens ANTERIORES (j < i)
    let cum_wv = decay.clone().matmul(wv.transpose()).transpose()
        .reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // soma dos pesos (denominador)
    let kexp_flat = k_exp.clone().swap_dims(1, 2).reshape([batch_size * channels, seq_len]);
    let cum_kexp = decay.matmul(kexp_flat.transpose()).transpose()
        .reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // ========================================
    // CONTRIBUIÇÃO DO TOKEN ATUAL (com bonus u)
    // ========================================
    // weight_current = exp(u + k_t) = exp(u) * exp(k_t)
    // Isso só aplica ao token na posição t, não aos anteriores!
    
    let u_exp = u_safe.exp().reshape([1, 1, channels]);  // exp(u)
    let current_weight = k_exp.clone() * u_exp.clone();  // exp(u + k_t) para cada t
    let current_num = current_weight.clone() * v;
    
    // ========================================
    // OUTPUT FINAL
    // ========================================
    // numerator = sum_past + current
    // denominator = sum_past_weights + current_weight
    
    let num = cum_wv + current_num;
    let den = cum_kexp + current_weight + 1e-6;
    
    // Clamp mais apertado no output
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
    let w_exp = w.clamp(-5.0, -0.1).reshape([1, channels]).exp();
    let u_exp = u.clamp(-5.0, 5.0).reshape([1, channels]).exp();
    let k_exp = k_safe.exp();
    
    // Token atual: exp(u + k)
    let current_weight = k_exp.clone() * u_exp;
    let current_num = current_weight.clone() * v.clone();
    
    // Output
    let num = state_num.clone() + current_num;
    let den = state_den.clone() + current_weight + 1e-6;
    let output = (num / den).clamp(-20.0, 20.0);

    // Atualiza estado para próximo token (só exp(k), sem u!)
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