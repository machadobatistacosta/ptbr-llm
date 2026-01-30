//! WKV Linear Attention - Versão Híbrida ESTÁVEL
//! 
//! Usa chunked processing com slice assignment (sem Tensor::stack)
//! Preserva per-channel decay aprendido
//!
//! Para seq_len=128 com batch=4, processa em chunks de 16

use burn::tensor::{backend::Backend, ElementConversion, Tensor};

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_fp32_accumulator: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 16,  // Chunk pequeno para evitar problemas de memória
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
            chunk_size: 32,
            use_fp32_accumulator: true,
        }
    }
}

/// WKV Linear Attention - VERSÃO SIMPLIFICADA E ESTÁVEL
/// 
/// Usa EMA recursiva com state carryover entre chunks.
/// Cada chunk computa output diretamente sem acumular em Vec.
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
    
    // Pré-computa exp(w) e exp(u) broadcastados
    let w_exp = w_safe.clone().exp().reshape([1, channels]);  // [1, channels]
    let u_bc = u_safe.reshape([1, channels]);  // [1, channels]
    
    // ========================================
    // PROCESSA SEQUÊNCIA COMPLETA EM CHUNKS
    // ========================================
    // Estado EMA: (numerator_state, denominator_state)
    let mut state_num = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut state_den = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    
    // Processa chunk por chunk e concatena
    let chunk_size = 16usize;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    
    let mut output_chunks: Vec<Tensor<B, 3>> = Vec::with_capacity(num_chunks);
    
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let chunk_len = end - start;
        
        // Extrai chunk
        let k_chunk = k_safe.clone().slice([0..batch_size, start..end, 0..channels]);
        let v_chunk = v.clone().slice([0..batch_size, start..end, 0..channels]);
        
        // Processa chunk com estado atual
        let (chunk_output, new_state_num, new_state_den) = process_chunk(
            k_chunk,
            v_chunk,
            w_exp.clone(),
            u_bc.clone(),
            state_num,
            state_den,
            batch_size,
            chunk_len,
            channels,
            &device,
        );
        
        output_chunks.push(chunk_output);
        state_num = new_state_num;
        state_den = new_state_den;
    }
    
    // Concatena chunks (mais seguro que stack para tensors 3D)
    if output_chunks.len() == 1 {
        output_chunks.pop().unwrap()
    } else {
        Tensor::cat(output_chunks, 1)
    }
}

/// Processa um chunk mantendo estado EMA
fn process_chunk<B: Backend>(
    k: Tensor<B, 3>,       // [batch, chunk_len, channels]
    v: Tensor<B, 3>,       // [batch, chunk_len, channels]
    w_exp: Tensor<B, 2>,   // [1, channels] - exp(decay)
    u_bc: Tensor<B, 2>,    // [1, channels] - bonus
    mut state_num: Tensor<B, 2>,   // [batch, channels]
    mut state_den: Tensor<B, 2>,   // [batch, channels]
    batch_size: usize,
    chunk_len: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
    // Pré-aloca output
    let mut output = Tensor::<B, 3>::zeros([batch_size, chunk_len, channels], device);
    
    for t in 0..chunk_len {
        // Extrai k_t e v_t
        let k_t = k.clone()
            .slice([0..batch_size, t..t+1, 0..channels])
            .reshape([batch_size, channels]);
        
        let v_t = v.clone()
            .slice([0..batch_size, t..t+1, 0..channels])
            .reshape([batch_size, channels]);
        
        // exp(k_t)
        let k_exp = k_t.clone().exp();
        
        // Contribuição do token atual: exp(u + k_t) * v_t
        let current_weight = (u_bc.clone() + k_t).exp();
        let current_value = current_weight.clone() * v_t.clone();
        
        // Output: (state_num + current_value) / (state_den + current_weight)
        let numerator = state_num.clone() + current_value;
        let denominator = state_den.clone() + current_weight + 1e-6;
        let output_t = (numerator / denominator).clamp(-50.0, 50.0);
        
        // IMPORTANTE: Usa slice_assign via reshape workaround
        // Burn não tem slice_assign direto, então construímos manualmente
        // Adiciona ao output via broadcast trick
        let output_t_3d = output_t.clone().reshape([batch_size, 1, channels]);
        
        // Cria máscara para esta posição
        if t == 0 && chunk_len == 1 {
            output = output_t_3d;
        } else {
            // Concatena com output existente ou usa slicing
            // Para evitar problemas, acumulamos via soma com máscara
            let before = if t > 0 {
                output.clone().slice([0..batch_size, 0..t, 0..channels])
            } else {
                Tensor::zeros([batch_size, 0, channels], device)
            };
            
            let after_len = chunk_len - t - 1;
            let after = if after_len > 0 {
                Tensor::zeros([batch_size, after_len, channels], device)
            } else {
                Tensor::zeros([batch_size, 0, channels], device)
            };
            
            // Reconstrói output
            if t == 0 {
                if after_len > 0 {
                    output = Tensor::cat(vec![output_t_3d, after], 1);
                } else {
                    output = output_t_3d;
                }
            } else if after_len > 0 {
                output = Tensor::cat(vec![before, output_t_3d, after], 1);
            } else {
                output = Tensor::cat(vec![before, output_t_3d], 1);
            }
        }
        
        // Atualiza estado: EMA com decay
        // state_num = exp(w) * state_num + exp(k_t) * v_t
        // state_den = exp(w) * state_den + exp(k_t)
        state_num = state_num * w_exp.clone() + k_exp.clone() * v_t;
        state_den = state_den * w_exp.clone() + k_exp;
    }
    
    (output, state_num, state_den)
}

/// WKV step para inferência (token por token)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels]
    v: Tensor<B, 2>,      // [batch, channels]
    w: Tensor<B, 1>,      // [channels] - learned decay
    u: Tensor<B, 1>,      // [channels] - learned bonus
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_batch_size, channels] = k.dims();
    
    let w_safe = w.clamp(-8.0, -0.01);
    let u_safe = u.clamp(-10.0, 10.0);
    let k_safe = k.clamp(-15.0, 15.0);
    
    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;
    
    let w_exp = w_safe.reshape([1, channels]).exp();
    let k_exp = k_safe.clone().exp();
    
    let u_bc = u_safe.reshape([1, channels]);
    let current_weight = (u_bc + k_safe).exp();
    let current_value = current_weight.clone() * v.clone();
    
    let numerator = state_num.clone() + current_value;
    let denominator = state_den.clone() + current_weight + 1e-6;
    let output = (numerator / denominator).clamp(-50.0, 50.0);
    
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