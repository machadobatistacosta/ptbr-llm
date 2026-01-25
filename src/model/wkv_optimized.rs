// src/model/wkv_optimized.rs
//! WKV simplificado - atenção linear sem loops para autodiff estável

use burn::tensor::{backend::Backend, Tensor};

/// Epsilon para estabilidade numérica
const EPS: f32 = 1e-6;

/// Configuração do WKV
#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_float64_accumulator: bool,
    pub parallel_heads: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32,
            use_float64_accumulator: true,
            parallel_heads: true,
        }
    }
}

/// WKV Simplificado - Linear Attention sem loops
/// 
/// Usa aproximação: wkv ≈ softmax(k) * v com decay exponencial
/// Isso evita o loop sequencial que explode o grafo de autodiff
/// 
/// Fórmula simplificada para treino:
/// 1. Aplica decay exponencial posicional
/// 2. Usa atenção linear: output = (softmax(k) * v)
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] - time decay (unused in simplified version)
    u: Tensor<B, 1>,      // [C] - time first bonus 
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [b, t, c] = k.dims();
    let device = k.device();
    
    // Simplified linear attention for stable autodiff:
    // We use a gating mechanism based on k values
    // This approximates RWKV behavior without the sequential dependency
    
    // Gate = sigmoid(k + u) - u provides "time first" bonus
    let u_broadcast = u.clone().reshape([1, 1, c]);
    let gate = burn::tensor::activation::sigmoid(k.clone() + u_broadcast);
    
    // Apply exponential weighting via w parameter
    // w is typically negative, so exp(w) < 1 provides decay
    let w_factor = w.clone().exp().reshape([1, 1, c]); // [1, 1, C], values < 1
    
    // Create position-dependent decay
    // decay[t] = w_factor ^ t (approximately)
    // We'll use a simpler approach: weight recent tokens more
    let position_weights = create_causal_weights::<B>(t as i64, &device); // [T, T]
    let position_weights = position_weights.reshape([1, t, t]); // [1, T, T]
    
    // Apply w_factor decay (broadcast over batch and time)
    // For stability, limit the decay
    let decay_scale = w_factor.clamp(-0.99, 0.99);
    
    // Compute attention: softmax on position weights, then apply to values
    // attention[i,j] = position_weights[i,j] (causal mask applied)
    let attention = burn::tensor::activation::softmax(position_weights, 2); // [1, T, T]
    
    // Output = attention @ v * gate * decay
    // attention: [1, T, T], v: [B, T, C]
    let attended = batched_matmul(attention, v.clone(), b, t, c); // [B, T, C]
    
    // Apply gating and decay
    let one = Tensor::<B, 3>::ones([b, t, c], &device);
    let half = (decay_scale + one.clone()) / 2.0;
    attended * gate.clone() * half + v * (one - gate) * 0.1
}

/// Creates causal position weights for attention
/// Lower triangular matrix with exponential decay
fn create_causal_weights<B: Backend>(t: i64, device: &B::Device) -> Tensor<B, 2> {
    // Create a lower triangular mask with decay
    // Simple approach: use distance-based weights
    let t_usize = t as usize;
    
    // Create weight matrix on CPU then transfer
    let mut weights = vec![0.0f32; t_usize * t_usize];
    
    for i in 0..t_usize {
        for j in 0..=i {
            // Exponential decay based on distance
            let dist = (i - j) as f32;
            // Nearby tokens get higher weight, decays with distance
            weights[i * t_usize + j] = (-dist * 0.1).exp();
        }
        // Positions j > i stay 0 (causal mask)
    }
    
    let data = burn::tensor::TensorData::from(weights.as_slice());
    let tensor: Tensor<B, 1> = Tensor::from_data(data, device);
    tensor.reshape([t_usize, t_usize])
}

/// Batched matrix multiplication: [1, T, T] @ [B, T, C] -> [B, T, C]
fn batched_matmul<B: Backend>(
    a: Tensor<B, 3>,  // [1, T, T] 
    b_tensor: Tensor<B, 3>,  // [B, T, C]
    batch: usize,
    _t: usize, 
    _c: usize,
) -> Tensor<B, 3> {
    // Expand a to [B, T, T]
    let a_expanded = a.repeat_dim(0, batch); // [B, T, T]
    
    // matmul: [B, T, T] @ [B, T, C] -> [B, T, C]
    // Burn handles batched matmul directly
    a_expanded.matmul(b_tensor)
}

/// WKV com Parallel Scan (mais rápido para GPU)
/// Usa associatividade do operador para paralelizar
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let [b, t, c] = k.dims();
    let device = k.device();

    // Para sequências curtas, usa método sequencial
    if t <= 64 {
        return wkv_linear(k, v, w, u, &WKVConfig::default());
    }

    // Divide em blocos para parallel scan
    let block_size = 64;
    let num_blocks = (t + block_size - 1) / block_size;

    // Fase 1: Processa cada bloco independentemente
    let mut block_outputs = Vec::with_capacity(num_blocks);
    let mut block_states = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(t);

        let k_blk = k.clone().slice([0..b, start..end, 0..c]);
        let v_blk = v.clone().slice([0..b, start..end, 0..c]);

        // Processa bloco
        let (out, state) = process_block(k_blk, v_blk, w.clone(), u.clone(), &device);
        block_outputs.push(out);
        block_states.push(state);
    }

    // Fase 2: Propaga estados entre blocos
    // TODO: Implementar parallel prefix sum para estados

    // Fase 3: Combina outputs
    Tensor::cat(block_outputs, 1)
}

fn process_block<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    device: &B::Device,
) -> (Tensor<B, 3>, (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)) {
    let config = WKVConfig::default();
    let output = wkv_linear(k, v, w, u, &config);

    // Estado final do bloco (placeholder)
    let [b, _, c] = output.dims();
    let state = (
        Tensor::zeros([b, c], device),
        Tensor::zeros([b, c], device),
        Tensor::zeros([b, c], device),
    );

    (output, state)
}