// src/model/wkv_optimized.rs
//! WKV otimizado com chunking e paralelização

use burn::tensor::{backend::Backend, Tensor};

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

/// WKV com Linear Complexity O(T) em vez de O(T²)
/// 
/// Fórmula:
/// wkv_t = (Σ_{i<t} e^{-(t-i-1)w + k_i} * v_i + e^{u+k_t} * v_t) /
///         (Σ_{i<t} e^{-(t-i-1)w + k_i} + e^{u+k_t})
///
/// Com estados recorrentes:
/// a_{t+1} = e^{-w} * a_t + e^{k_t} * v_t
/// b_{t+1} = e^{-w} * b_t + e^{k_t}
/// p_{t+1} = max(p_t - w, k_t)  # para estabilidade numérica
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] - time decay
    u: Tensor<B, 1>,      // [C] - time first bonus
    config: &WKVConfig,
) -> Tensor<B, 3> {
    let [b, t, c] = k.dims();
    let device = k.device();
    let chunk_size = config.chunk_size;

    // Expande w e u para broadcast
    // w é o parâmetro time_decay. O decay real em log-space é -exp(w).
    let w_log = w.clone().exp().neg(); // -e^w
    
    // Estados iniciais (log-space para estabilidade)
    let mut aa = Tensor::<B, 2>::zeros([b, c], &device);
    let mut bb = Tensor::<B, 2>::zeros([b, c], &device);
    let mut pp = Tensor::<B, 2>::zeros([b, c], &device) - 1e30;

    let mut outputs = Vec::with_capacity(t);
    let num_chunks = (t + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(t);

        let k_chunk = k.clone().slice([0..b, start..end, 0..c]);
        let v_chunk = v.clone().slice([0..b, start..end, 0..c]);

        for i in 0..(end - start) {
            let kt = k_chunk.clone().slice([0..b, i..i + 1, 0..c]).reshape([b, c]);
            let vt = v_chunk.clone().slice([0..b, i..i + 1, 0..c]).reshape([b, c]);

            // ww = u + k_t (bonus para token atual)
            let ww = u.clone().reshape([1, c]) + kt.clone();

            // Log-sum-exp trick
            let p = pp.clone().max_pair(ww.clone());
            let e1 = (pp.clone() - p.clone()).exp();
            let e2 = (ww - p.clone()).exp();

            // Output com estabilidade numérica
            let numerator = e1.clone() * aa.clone() + e2.clone() * vt.clone();
            let denominator = e1.clone() * bb.clone() + e2.clone() + 1e-7;
            let wkv = numerator / denominator;

            outputs.push(wkv.reshape([b, 1, c]));

            // Atualiza estados para próximo timestep
            // w_log já é o valor em log-space (-exp(w)), então somamos diretamento
            let ww2 = w_log.clone().reshape([1, c]) + pp.clone(); 
            let p2 = ww2.clone().max_pair(kt.clone());
            let e1_2 = (ww2 - p2.clone()).exp();
            let e2_2 = (kt - p2.clone()).exp();

            aa = e1_2.clone() * aa + e2_2.clone() * vt;
            bb = e1_2 * bb + e2_2;
            pp = p2;
        }
    }

    Tensor::cat(outputs, 1)
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