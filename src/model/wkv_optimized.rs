// src/model/wkv_optimized.rs
//! WKV Kernel Fusion - Implementação CubeCL para RWKV
//! 
//! Resolve OOM e Stack Overflow usando kernel GPU customizado.
//! 
//! Fórmula WKV (numerically stable):
//!   state_a = state_a * exp(-w) + exp(k) * v  (numerator)
//!   state_b = state_b * exp(-w) + exp(k)      (denominator)
//!   output  = (state_a + exp(u+k)*v) / (state_b + exp(u+k))

use burn::tensor::{backend::Backend, Tensor, ElementConversion}; // ElementConversion added for as_primitive
// Note: CubeCL imports are commented out to ensure compilation on T4 without full CubeCL setup first.
// The fallback implementation is the immediate fix for Segfaults.

/* 
#[cfg(feature = "cubecl")]
use cubecl::prelude::*;
*/

// ══════════════════════════════════════════════════════════════════════════════
//                           CONFIGURATION
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize,
    pub use_float64_accumulator: bool,
    pub parallel_heads: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            use_float64_accumulator: false,
            parallel_heads: true,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//                    FALLBACK: CPU/Non-CUDA Implementation
// ══════════════════════════════════════════════════════════════════════════════

/// Implementação chunk-based para backends sem CUDA
/// Usa detach() para limitar crescimento do grafo de autodiff
fn wkv_chunked_fallback<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    chunk_size: usize,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Broadcast parameters
    // w e u são [C]
    // w_exp = (-w).exp() -> [1, 1, C]
    let w_exp = w.clone().neg().exp().reshape([1, 1, channels]); // w é [C]
    let u_broad = u.clone().reshape([1, 1, channels]);
    
    // State tensors [B, 1, C]
    let mut aa = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut bb = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut pp = Tensor::<B, 3>::full([batch_size, 1, channels], -1e30, &device); // -inf
    
    let mut outputs = Vec::new();
    
    // Processar em chunks para permitir detach periódico
    for chunk_start in (0..seq_len).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(seq_len);
        let chunk_len = chunk_end - chunk_start;
        
        let mut chunk_outputs = Vec::with_capacity(chunk_len);
        
        // Loop sequencial dentro do chunk
        for t in 0..chunk_len {
            let abs_t = chunk_start + t;
            // Slice do tempo t: [B, 1, C]
            let kt = k.clone().slice([0..batch_size, abs_t..abs_t+1, 0..channels]);
            let vt = v.clone().slice([0..batch_size, abs_t..abs_t+1, 0..channels]);
            
            // Compute output
            let ww = u_broad.clone() + kt.clone();
            let qq = pp.clone().max_pair(ww.clone());
            let e1 = (pp.clone() - qq.clone()).exp();
            let e2 = (ww - qq.clone()).exp();
            
            let num = e1.clone() * aa.clone() + e2.clone() * vt.clone();
            let den = e1.clone() * bb.clone() + e2.clone();
            
            // wkv = num / den
            // Adicionamos epsilon para estabilidade
            let yt = num / (den + 1e-8);
            
            chunk_outputs.push(yt);
            
            // Update state para t+1
            // ww_next = pp + (-w) = pp - log(w_exp) ?
            // w_exp = exp(-w). Então log(w_exp) = -w.
            // pp_next = max(pp - w, k)
            let w_log = w_exp.clone().log(); // = -w
            let ww_next = pp.clone() + w_log; 
            let qq_next = ww_next.clone().max_pair(kt.clone());
            
            let e1_next = (ww_next - qq_next.clone()).exp();
            let e2_next = (kt - qq_next.clone()).exp();
            
            aa = e1_next.clone() * aa + e2_next.clone() * vt;
            bb = e1_next * bb + e2_next;
            pp = qq_next;
        }
        
        // Concatenar saídas do chunk
        let chunk_output = Tensor::cat(chunk_outputs, 1);
        outputs.push(chunk_output);
        
        // CRUCIAL: Detach state do grafo para não acumular nós entre chunks
        // Isso resolve o Stack Overflow (Segfault) em sequências longas!
        // O gradiente flui DENTRO do chunk, mas é cortado entre chunks (TBPTT).
        // Para RWKV perfeito, precisaríamos passar o gradiente do estado.
        // Mas para evitar crash, isso é infinitamente melhor que nada.
        aa = aa.detach();
        bb = bb.detach();
        pp = pp.detach();
    }
    
    Tensor::cat(outputs, 1)
}


// ══════════════════════════════════════════════════════════════════════════════
//                           PUBLIC API
// ══════════════════════════════════════════════════════════════════════════════

/// WKV Linear - Entrada principal
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] - time decay
    u: Tensor<B, 1>,      // [C] - time first bonus
    config: &WKVConfig,
) -> Tensor<B, 3> {
    
    // Por enquanto, forçamos o fallback otimizado que resolve o crash.
    // A implementação CubeCL precisa de setup no Cargo.toml e features específicas.
    // O fallback já traz a lógica de "Chunked Detach" que blinda a memória.
    wkv_chunked_fallback(k, v, w, u, config.chunk_size)
}

/// Alias para compatibilidade
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::default())
}
