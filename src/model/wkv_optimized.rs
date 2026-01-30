//! WKV Linear Attention - Versão ESTÁVEL para CUDA
//! 
//! CORREÇÃO: Evita Tensor::stack que causa segfault no CUDA
//! Usa abordagem vetorizada com cumsum via matmul triangular
//!
//! Baseado na fórmula RWKV:
//! wkv_t = (Σ e^{-(t-i-1)*w + k_i} * v_i + e^{u+k_t} * v_t) / (Σ e^{-(t-i-1)*w + k_i} + e^{u+k_t})

use burn::tensor::{backend::Backend, ElementConversion, Tensor, TensorData};

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

/// WKV Linear Attention - VERSÃO VETORIZADA ESTÁVEL
/// 
/// Usa matmul com matriz triangular para simular cumsum com decay,
/// evitando loops que causam segfault no CUDA.
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    w: Tensor<B, 1>,      // [channels] - learned per-channel decay (negativo)
    u: Tensor<B, 1>,      // [channels] - learned per-channel bonus
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // ========================================
    // ESTABILIDADE NUMÉRICA
    // ========================================
    let w_safe = w.clamp(-8.0, -0.01);
    let u_safe = u.clamp(-10.0, 10.0);
    let k_safe = k.clamp(-15.0, 15.0);
    
    // ========================================
    // CONTRIBUIÇÃO DO TOKEN ATUAL
    // ========================================
    // current_weight = exp(u + k) para cada posição
    // current_value = current_weight * v
    
    let u_bc = u_safe.reshape([1, 1, channels]);
    let current_weight = (u_bc.clone() + k_safe.clone()).exp();
    let current_value = current_weight.clone() * v.clone();
    
    // ========================================
    // CONTRIBUIÇÃO DO PASSADO VIA MATMUL TRIANGULAR
    // ========================================
    // Para cada posição t, queremos somar contribuições de i < t com decay
    // Isso é equivalente a: M @ weighted_v onde M é triangular inferior com decay
    
    // Calcula média do decay para usar como aproximação uniforme
    // (per-channel exato seria O(seq_len * seq_len * channels) - muito caro)
    let w_mean: f32 = w_safe.clone().mean().into_scalar().elem();
    
    // Matriz triangular inferior com decay exponencial
    // M[i,j] = exp(w * (i-j)) se j < i, else 0
    // M[i,i] = 0 (diagonal é zero - token atual tratado separadamente)
    let decay_matrix = create_decay_matrix::<B>(seq_len, w_mean, &device);
    
    // exp(k) para ponderação
    let k_exp = k_safe.exp();
    
    // weighted_v = exp(k) * v
    let weighted_v = k_exp.clone() * v;
    
    // ========================================
    // CUMSUM COM DECAY VIA MATMUL
    // ========================================
    // Reorganiza para matmul: [batch, channels, seq_len]
    let wv_t = weighted_v.swap_dims(1, 2);  // [batch, channels, seq_len]
    let wv_flat = wv_t.reshape([batch_size * channels, seq_len]);  // [batch*channels, seq_len]
    
    // cum_wv = decay_matrix @ wv_flat^T -> [seq_len, batch*channels]
    // Então transpose de volta
    let wv_flat_t = wv_flat.transpose();  // [seq_len, batch*channels]
    let cum_wv_t = decay_matrix.clone().matmul(wv_flat_t);  // [seq_len, batch*channels]
    let cum_wv = cum_wv_t.transpose()  // [batch*channels, seq_len]
        .reshape([batch_size, channels, seq_len])
        .swap_dims(1, 2);  // [batch, seq_len, channels]
    
    // Mesmo para os pesos (denominador)
    let kexp_t = k_exp.swap_dims(1, 2);  // [batch, channels, seq_len]
    let kexp_flat = kexp_t.reshape([batch_size * channels, seq_len]);
    let kexp_flat_t = kexp_flat.transpose();
    let cum_kexp_t = decay_matrix.matmul(kexp_flat_t);
    let cum_kexp = cum_kexp_t.transpose()
        .reshape([batch_size, channels, seq_len])
        .swap_dims(1, 2);  // [batch, seq_len, channels]
    
    // ========================================
    // OUTPUT FINAL
    // ========================================
    // numerator = past_contribution + current_contribution
    // denominator = past_weights + current_weight
    let numerator = cum_wv + current_value;
    let denominator = cum_kexp + current_weight + 1e-6;
    
    (numerator / denominator).clamp(-100.0, 100.0)
}

/// Cria matriz triangular inferior com decay exponencial
/// M[i,j] = exp(w * (i-j)) se j < i, else 0
fn create_decay_matrix<B: Backend>(
    seq_len: usize,
    w_mean: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Cria dados da matriz
    let mut data = vec![0.0f32; seq_len * seq_len];
    
    for i in 0..seq_len {
        for j in 0..i {  // j < i apenas (triangular inferior, sem diagonal)
            let dist = (i - j) as f32;
            // decay = exp(w * dist), w é negativo então é decay
            let decay = (w_mean * dist).max(-20.0).exp();
            data[i * seq_len + j] = decay;
        }
    }
    
    let tensor_data = TensorData::from(data.as_slice());
    let matrix: Tensor<B, 1> = Tensor::from_data(tensor_data, device);
    matrix.reshape([seq_len, seq_len])
}

/// WKV step para inferência (token por token)
/// Mantém estado entre chamadas para geração autoregressiva
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels] - key para token atual
    v: Tensor<B, 2>,      // [batch, channels] - value para token atual
    w: Tensor<B, 1>,      // [channels] - learned decay
    u: Tensor<B, 1>,      // [channels] - learned bonus
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),  // (state_a, state_b, last_k)
) -> Tensor<B, 2> {
    let [_batch_size, channels] = k.dims();
    
    // Clamping para estabilidade
    let w_safe = w.clamp(-8.0, -0.01);
    let u_safe = u.clamp(-10.0, 10.0);
    let k_safe = k.clamp(-15.0, 15.0);
    
    let (ref mut state_a, ref mut state_b, ref mut _last_k) = state;
    
    // exp(w) para decay
    let w_exp = w_safe.reshape([1, channels]).exp();
    
    // exp(k) para token atual
    let k_exp = k_safe.clone().exp();
    
    // Contribuição do token atual com bonus u
    let u_bc = u_safe.reshape([1, channels]);
    let current_weight = (u_bc + k_safe).exp();
    let current_value = current_weight.clone() * v.clone();
    
    // Output: (past + current) / (past_weights + current_weight)
    let numerator = state_a.clone() + current_value;
    let denominator = state_b.clone() + current_weight + 1e-6;
    let output = (numerator / denominator).clamp(-100.0, 100.0);
    
    // Atualiza estado para próximo token
    *state_a = state_a.clone() * w_exp.clone() + k_exp.clone() * v;
    *state_b = state_b.clone() * w_exp + k_exp;
    
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