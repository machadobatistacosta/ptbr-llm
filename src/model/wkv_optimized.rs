//! WKV Linear Attention - Versão CORRIGIDA
//! Usa decay real (w) por canal, não valor fixo

use burn::tensor::{backend::Backend, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self { chunk_size: 32 }
    }
}

impl WKVConfig {
    pub fn for_t4() -> Self { Self::default() }
    pub fn for_high_memory() -> Self { Self { chunk_size: 64 } }
}

/// WKV Linear Attention - CORRIGIDO
/// Usa o decay real (w) por canal em vez de valor fixo
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [batch, seq_len, channels]
    v: Tensor<B, 3>,      // [batch, seq_len, channels]
    w: Tensor<B, 1>,      // [channels] - decay POR CANAL (treinável!)
    u: Tensor<B, 1>,      // [channels] - bonus para token atual
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // ========================================
    // ESTABILIDADE NUMÉRICA
    // ========================================
    let k_safe = k.clamp(-15.0, 15.0);
    let w_safe = w.clamp(-8.0, -0.1);  // decay deve ser negativo
    let u_safe = u.clamp(-10.0, 10.0);
    
    // ========================================
    // VERSÃO VETORIZADA COM DECAY REAL
    // ========================================
    
    // Reshape para broadcast
    let w_3d = w_safe.reshape([1, 1, channels]);
    let u_3d = u_safe.reshape([1, 1, channels]);
    
    // exp(k) para weighted attention
    let k_exp = k_safe.clone().exp();
    
    // Weighted values
    let weighted_v = k_exp.clone() * v.clone();
    
    // Bonus para token atual: exp(u + k)
    let uk_exp = (u_3d + k_safe.clone()).clamp(-30.0, 30.0).exp();
    let current_contrib = uk_exp.clone() * v.clone();
    let current_weight = uk_exp;
    
    // ========================================
    // CUMULATIVE SUM COM DECAY REAL
    // ========================================
    
    // Decay por posição: w^t onde t é a distância
    // Para seq_len=128, criamos matriz de decay
    let w_exp = w_3d.exp();  // [1, 1, channels]
    
    // Acumuladores
    let mut cum_wv = Tensor::<B, 3>::zeros([batch_size, seq_len, channels], &device);
    let mut cum_w = Tensor::<B, 3>::zeros([batch_size, seq_len, channels], &device);
    
    // Estado acumulado (decai com w a cada passo)
    let mut acc_wv = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut acc_w = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    
    let w_exp_2d = w_exp.reshape([1, channels]);
    
    for t in 0..seq_len {
        // Extrai valores na posição t
        let wv_t = weighted_v.clone()
            .slice([0..batch_size, t..t+1, 0..channels])
            .reshape([batch_size, channels]);
        let w_t = k_exp.clone()
            .slice([0..batch_size, t..t+1, 0..channels])
            .reshape([batch_size, channels]);
        
        // Aplica decay ao estado anterior e adiciona novo
        acc_wv = acc_wv * w_exp_2d.clone() + wv_t;
        acc_w = acc_w.clone() * w_exp_2d.clone() + w_t;
        
        // Armazena (via slice assignment não disponível, usamos outro método)
        // Por enquanto, construímos lista e concatenamos no final
        if t == 0 {
            cum_wv = acc_wv.clone().reshape([batch_size, 1, channels]);
            cum_w = acc_w.clone().reshape([batch_size, 1, channels]);
        } else {
            cum_wv = Tensor::cat(vec![
                cum_wv,
                acc_wv.clone().reshape([batch_size, 1, channels])
            ], 1);
            cum_w = Tensor::cat(vec![
                cum_w,
                acc_w.clone().reshape([batch_size, 1, channels])
            ], 1);
        }
    }
    
    // Output final
    let numerator = cum_wv + current_contrib;
    let denominator = cum_w + current_weight + 1e-6;
    
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

    let w_exp = w.clone().clamp(-8.0, -0.1).reshape([1, channels]).exp();
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