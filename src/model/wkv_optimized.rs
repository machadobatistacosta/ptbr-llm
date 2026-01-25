// src/model/wkv_optimized.rs
//! WKV otimizado - O(T) sequential scan para evitar memory explosion

use burn::tensor::{backend::Backend, Tensor};

/// Epsilon para estabilidade numérica
const EPS: f32 = 1e-7;

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

/// WKV com Linear Complexity O(T) usando scan sequencial
/// 
/// Fórmula RWKV:
/// wkv_t = (Σ_{i<t} e^{-(t-i-1)w + k_i} * v_i + e^{u+k_t} * v_t) /
///         (Σ_{i<t} e^{-(t-i-1)w + k_i} + e^{u+k_t})
///
/// Estados recorrentes (log-space para estabilidade):
/// aa = weighted value accumulator
/// bb = weight accumulator  
/// pp = max exponent tracker (log-sum-exp)
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] - time decay (raw param, typically negative)
    u: Tensor<B, 1>,      // [C] - time first bonus
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [b, t, c] = k.dims();
    let device = k.device();

    // Decay factor: w_log = -exp(w)
    // For w=-5, exp(-5)≈0.007, so w_log≈-0.007
    // This makes decay ≈ exp(-0.007) ≈ 0.993 per step
    let w_log = w.clone().exp().neg(); // [C]

    // Initialize states [B, C]
    let mut aa = Tensor::<B, 2>::zeros([b, c], &device);
    let mut bb = Tensor::<B, 2>::zeros([b, c], &device);
    let mut pp = Tensor::<B, 2>::zeros([b, c], &device) - 1e30; // Start at -inf for log-sum-exp

    // Collect outputs
    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(t);

    for t_idx in 0..t {
        // Extract k_t, v_t: [B, C]
        let k_t = k.clone().slice([0..b, t_idx..t_idx+1, 0..c]).reshape([b, c]);
        let v_t = v.clone().slice([0..b, t_idx..t_idx+1, 0..c]).reshape([b, c]);

        // Current token weight: ww = u + k_t
        let ww = u.clone().reshape([1, c]) + k_t.clone(); // [B, C]

        // Log-sum-exp for numerical stability
        // p = max(pp, ww) for stable exp computation
        let p = pp.clone().max_pair(ww.clone());
        
        // e1 = exp(pp - p): contribution from history
        // e2 = exp(ww - p): contribution from current token
        let e1 = (pp.clone() - p.clone()).exp();
        let e2 = (ww.clone() - p.clone()).exp();

        // WKV output: (e1 * aa + e2 * v_t) / (e1 * bb + e2)
        let denom = e1.clone() * bb.clone() + e2.clone() + EPS;
        let wkv_t = (e1.clone() * aa.clone() + e2.clone() * v_t.clone()) / denom;

        outputs.push(wkv_t);

        // Update states for next iteration
        // ww2 = w_log + pp (decay applied to history)
        let ww2 = w_log.clone().reshape([1, c]) + pp.clone();
        
        // p2 = max(ww2, k_t) for stable update
        let p2 = ww2.clone().max_pair(k_t.clone());
        let e1_2 = (ww2 - p2.clone()).exp();
        let e2_2 = (k_t - p2.clone()).exp();

        // Update accumulators
        aa = e1_2.clone() * aa + e2_2.clone() * v_t;
        bb = e1_2 * bb + e2_2;
        pp = p2;
    }

    // Stack outputs: Vec<[B, C]> -> [B, T, C]
    Tensor::stack(outputs, 1)
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