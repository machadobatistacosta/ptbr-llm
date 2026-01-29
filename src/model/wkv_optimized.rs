//! WKV Linear Attention - Versão Vetorizada para GPU
//! Otimizado para reduzir kernel launches no CUDA

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

/// WKV Linear Attention - Versão Vetorizada
/// 
/// Usa operações matriciais paralelas em vez de loops sequenciais.
/// Isso reduz de ~1000 kernels CUDA para ~20 kernels grandes.
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    _w: Tensor<B, 1>,      // [channels] - decay (usando aproximação fixa por performance)
    u: Tensor<B, 1>,      // [channels] - bonus
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Clamp para estabilidade numérica
    let k_safe = k.clamp(-20.0, 20.0);
    
    // ========================================
    // ESTRATÉGIA: EMA Vetorizado
    // ========================================
    
    // exp(k) para pesos de atenção
    let k_exp = k_safe.clone().exp(); // [batch, seq_len, channels]
    
    // v ponderado por exp(k)
    let weighted_v = k_exp.clone() * v.clone(); // [batch, seq_len, channels]
    
    // Bonus u para token atual: exp(u)
    let u_exp = u.exp().reshape([1, 1, channels]); // [1, 1, channels]
    
    // Contribuição do token atual com bonus
    let current_contrib = k_exp.clone() * u_exp.clone() * v.clone();
    let current_weight = k_exp.clone() * u_exp;
    
    // ========================================
    // Cumulative sum via matriz triangular inferior
    // ========================================
    // 
    // Para calcular "soma de todos os tokens anteriores", usamos:
    // cumsum = tril @ input
    // Onde tril é matriz triangular inferior com decay
    
    // Cria matriz de decay triangular inferior [seq_len, seq_len]
    // decay_matrix[i,j] = exp(w_mean * (i-j)) para j < i, 0 caso contrário
    let w_mean: f32 = -0.5; // Aproximação conservadora do decay médio
    let mut decay_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..i {  // j < i (não inclui diagonal - diagonal é handled pelo bonus u)
            let dist = (i - j) as f32;
            decay_data[i * seq_len + j] = (w_mean * dist).exp();
        }
    }
    
    // Cria tensor usando TensorData
    let decay_tensor_data = TensorData::from(decay_data.as_slice());
    let sum_matrix: Tensor<B, 1> = Tensor::from_data(decay_tensor_data, &device);
    let sum_matrix = sum_matrix.reshape([seq_len, seq_len]); // [seq_len, seq_len]
    
    // Aplica às contribuições: matmul entre sum_matrix e sequência
    // Para cada batch e canal: cumsum_v[i] = sum_{j<i} decay^(i-j) * weighted_v[j]
    
    // Transpor weighted_v: [batch, seq_len, channels] -> [batch, channels, seq_len]
    let wv_t = weighted_v.swap_dims(1, 2);
    
    // Flatten batch*channels: [batch*channels, seq_len]
    let wv_flat = wv_t.reshape([batch_size * channels, seq_len]);
    
    // Matmul: [seq_len, seq_len] @ [seq_len, batch*channels] = [seq_len, batch*channels]
    let wv_flat_t = wv_flat.transpose(); // [seq_len, batch*channels]
    let cum_wv_t = sum_matrix.clone().matmul(wv_flat_t); // [seq_len, batch*channels]
    let cum_wv = cum_wv_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    // cum_wv: [batch, seq_len, channels]
    
    // Mesma coisa para pesos (denominador)
    let kexp_t = k_exp.clone().swap_dims(1, 2);
    let kexp_flat = kexp_t.reshape([batch_size * channels, seq_len]);
    let kexp_flat_t = kexp_flat.transpose();
    let cum_kexp_t = sum_matrix.matmul(kexp_flat_t);
    let cum_kexp = cum_kexp_t.transpose().reshape([batch_size, channels, seq_len]).swap_dims(1, 2);
    // cum_kexp: [batch, seq_len, channels]
    
    // Output final: (cum_wv + current_contrib) / (cum_kexp + current_weight + eps)
    let numerator = cum_wv + current_contrib;
    let denominator = cum_kexp + current_weight + 1e-8;
    
    numerator / denominator
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

    let w_exp = w.clone().clamp(-8.0, -0.1).reshape([1, channels]).exp();
    let u_exp = u.reshape([1, channels]).exp();

    // exp(k) para pesos
    let k_exp = k.clone().clamp(-20.0, 20.0).exp();
    
    // Bonus para token atual
    let uk_exp = k_exp.clone() * u_exp;

    // Output: (state_num + uk_exp * v) / (state_den + uk_exp)
    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp + 1e-8;
    let output = num / den;

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