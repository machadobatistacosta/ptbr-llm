//! WKV Linear Attention - Versão Vetorizada para GPU
//! Otimizado para reduzir kernel launches no CUDA
//! Com estabilidade numérica melhorada

use burn::tensor::{backend::Backend, Tensor, TensorData};

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

/// WKV Linear Attention - Versão Vetorizada com Estabilidade Numérica
/// 
/// Usa operações matriciais paralelas em vez de loops sequenciais.
/// Inclui proteções contra overflow/underflow.
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    _w: Tensor<B, 1>,     // [channels] - decay (usando aproximação fixa por performance)
    u: Tensor<B, 1>,      // [channels] - bonus
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // ========================================
    // ESTABILIDADE NUMÉRICA
    // ========================================
    // Clamp agressivo para evitar overflow em exp()
    let k_safe = k.clamp(-15.0, 15.0);
    let u_safe = u.clamp(-15.0, 15.0);
    
    // Normaliza k subtraindo o máximo por sequência (LogSumExp trick)
    // Isso previne overflow em exp(k)
    let k_max = k_safe.clone().max_dim(1); // [batch, 1, channels]
    let k_centered = k_safe - k_max.clone(); // Centraliza k
    
    // exp(k - k_max) é numericamente estável
    let k_exp = k_centered.exp(); // [batch, seq_len, channels]
    
    // v ponderado por exp(k - k_max)
    let weighted_v = k_exp.clone() * v.clone(); // [batch, seq_len, channels]
    
    // Bonus u para token atual (também normalizado)
    let u_exp = u_safe.clamp(-10.0, 10.0).exp().reshape([1, 1, channels]);
    
    // Contribuição do token atual com bonus
    let current_contrib = k_exp.clone() * u_exp.clone() * v.clone();
    let current_weight = k_exp.clone() * u_exp;
    
    // ========================================
    // Cumulative sum via matriz triangular inferior
    // ========================================
    
    // Cria matriz de decay triangular inferior [seq_len, seq_len]
    // decay_matrix[i,j] = exp(w_mean * (i-j)) para j < i, 0 caso contrário
    // Usando decay mais suave para estabilidade
    let w_mean: f32 = -0.3; // Decay mais suave
    let mut decay_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..i {
            let dist = (i - j) as f32;
            // Limita o decay para não ficar muito pequeno
            let decay_val = (w_mean * dist).max(-10.0).exp();
            decay_data[i * seq_len + j] = decay_val;
        }
    }
    
    let decay_tensor_data = TensorData::from(decay_data.as_slice());
    let sum_matrix: Tensor<B, 1> = Tensor::from_data(decay_tensor_data, &device);
    let sum_matrix = sum_matrix.reshape([seq_len, seq_len]);
    
    // Aplica às contribuições
    let wv_t = weighted_v.swap_dims(1, 2);
    let wv_flat = wv_t.reshape([batch_size * channels, seq_len]);
    let wv_flat_t = wv_flat.transpose();
    let cum_wv_t = sum_matrix.clone().matmul(wv_flat_t);
    let cum_wv = cum_wv_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // Mesma coisa para pesos (denominador)
    let kexp_t = k_exp.clone().swap_dims(1, 2);
    let kexp_flat = kexp_t.reshape([batch_size * channels, seq_len]);
    let kexp_flat_t = kexp_flat.transpose();
    let cum_kexp_t = sum_matrix.matmul(kexp_flat_t);
    let cum_kexp = cum_kexp_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    
    // Output final com epsilon maior para estabilidade
    let numerator = cum_wv + current_contrib;
    let denominator = cum_kexp + current_weight + 1e-6; // Epsilon maior
    
    // Clamp final para evitar valores extremos
    let output = numerator / denominator;
    output.clamp(-100.0, 100.0)
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

    // Clamps para estabilidade
    let w_exp = w.clone().clamp(-8.0, -0.1).reshape([1, channels]).exp();
    let u_exp = u.clamp(-10.0, 10.0).reshape([1, channels]).exp();
    let k_exp = k.clone().clamp(-15.0, 15.0).exp();
    
    // Bonus para token atual
    let uk_exp = k_exp.clone() * u_exp;

    // Output: (state_num + uk_exp * v) / (state_den + uk_exp)
    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp + 1e-6;
    let output = (num / den).clamp(-100.0, 100.0);

    // Atualiza estado
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