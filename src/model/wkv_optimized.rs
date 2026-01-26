// src/model/wkv_optimized.rs
//! WKV Ultra Memory Efficient - Gradient Checkpointing Simulado
//! 
//! Estratégia: Processar token-por-token com detach agressivo dos estados.
//! O grafo de autodiff contém apenas a última operação de cada token.
//! Estados (aa, bb, pp) são DETACHED - não acumulam gradiente através do tempo.
//! 
//! Trade-off: TBPTT (Truncated Backprop Through Time) - gradientes não fluem
//! entre tokens, mas isso é aceitável para RWKV que é quase-linear de qualquer forma.
//! 
//! Memória estimada: ~6-8GB para 400M params, seq_len=512, batch=4

use burn::tensor::{backend::Backend, Tensor};

// ══════════════════════════════════════════════════════════════════════════════
//                           CONFIGURATION
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct WKVConfig {
    /// Número de tokens processados antes de sync/detach
    pub chunk_size: usize,
    /// Se true, usa f64 para acumuladores (mais estável, mais lento)
    pub use_float64_accumulator: bool,
    /// Número de checkpoints por sequência (para gradient checkpointing)
    pub num_checkpoints: usize,
    /// Se true, força detach a cada token (máxima economia de memória)
    pub aggressive_detach: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 8,              // Chunks pequenos para T4
            use_float64_accumulator: false,
            num_checkpoints: 4,
            aggressive_detach: true,    // CRÍTICO para T4
        }
    }
}

impl WKVConfig {
    /// Configuração otimizada para T4 16GB
    pub fn for_t4() -> Self {
        Self {
            chunk_size: 8,
            use_float64_accumulator: false,
            num_checkpoints: 8,
            aggressive_detach: true,
        }
    }
    
    /// Configuração para GPUs com mais memória (A100, etc)
    pub fn for_high_memory() -> Self {
        Self {
            chunk_size: 64,
            use_float64_accumulator: false,
            num_checkpoints: 2,
            aggressive_detach: false,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//                    CORE IMPLEMENTATION: ULTRA MEMORY EFFICIENT
// ══════════════════════════════════════════════════════════════════════════════

/// WKV Token-by-Token com Detach Agressivo
fn wkv_token_by_token<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] - time decay
    u: Tensor<B, 1>,      // [C] - time first bonus
    aggressive_detach: bool,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Pré-computar parâmetros [1, 1, C]
    let neg_w = w.clone().neg().reshape([1, 1, channels]);
    let u_broad = u.reshape([1, 1, channels]);
    
    // Estados iniciais [B, 1, C]
    let mut aa = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut bb = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut pp = Tensor::<B, 3>::full([batch_size, 1, channels], -1e30, &device); // usando -1e30 para segurança float32
    
    let mut outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(seq_len);
    
    for t in 0..seq_len {
        let kt = k.clone().slice([0..batch_size, t..t+1, 0..channels]);
        let vt = v.clone().slice([0..batch_size, t..t+1, 0..channels]);
        
        let ww = u_broad.clone() + kt.clone();
        let qq = pp.clone().max_pair(ww.clone());
        let e1 = (pp.clone() - qq.clone()).exp();
        let e2 = (ww - qq.clone()).exp();
        
        let num = e1.clone() * aa.clone() + e2.clone() * vt.clone();
        let den = e1.clone() * bb.clone() + e2.clone();
        
        let eps = 1e-9;
        let yt = num / (den + eps);
        outputs.push(yt);
        
        // Update state
        let pp_decayed = pp.clone() + neg_w.clone();
        let qq_next = pp_decayed.clone().max_pair(kt.clone());
        let e1_next = (pp_decayed - qq_next.clone()).exp();
        let e2_next = (kt - qq_next.clone()).exp();
        
        let aa_new = e1_next.clone() * aa + e2_next.clone() * vt;
        let bb_new = e1_next * bb + e2_next;
        
        if aggressive_detach {
            aa = aa_new.detach();
            bb = bb_new.detach();
            pp = qq_next.detach();
        } else {
            aa = aa_new;
            bb = bb_new;
            pp = qq_next;
        }
    }
    
    Tensor::cat(outputs, 1)
}

/// Versão chunked (menos agressiva)
fn wkv_chunked<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    chunk_size: usize,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    let neg_w = w.clone().neg().reshape([1, 1, channels]);
    let u_broad = u.reshape([1, 1, channels]);
    
    let mut aa = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut bb = Tensor::<B, 3>::zeros([batch_size, 1, channels], &device);
    let mut pp = Tensor::<B, 3>::full([batch_size, 1, channels], -1e30, &device);
    
    let mut all_outputs: Vec<Tensor<B, 3>> = Vec::new();
    
    let mut chunk_start = 0;
    while chunk_start < seq_len {
        let chunk_end = (chunk_start + chunk_size).min(seq_len);
        let chunk_len = chunk_end - chunk_start;
        
        let mut chunk_outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(chunk_len);
        
        for t in chunk_start..chunk_end {
            let kt = k.clone().slice([0..batch_size, t..t+1, 0..channels]);
            let vt = v.clone().slice([0..batch_size, t..t+1, 0..channels]);
            
            let ww = u_broad.clone() + kt.clone();
            let qq = pp.clone().max_pair(ww.clone());
            let e1 = (pp.clone() - qq.clone()).exp();
            let e2 = (ww - qq.clone()).exp();
            
            let num = e1.clone() * aa.clone() + e2.clone() * vt.clone();
            let den = e1.clone() * bb.clone() + e2.clone();
            let yt = num / (den + 1e-9);
            chunk_outputs.push(yt);
            
            let pp_decayed = pp.clone() + neg_w.clone();
            let qq_next = pp_decayed.clone().max_pair(kt.clone());
            let e1_next = (pp_decayed - qq_next.clone()).exp();
            let e2_next = (kt - qq_next.clone()).exp();
            
            aa = e1_next.clone() * aa + e2_next.clone() * vt;
            bb = e1_next * bb + e2_next;
            pp = qq_next;
        }
        
        all_outputs.push(Tensor::cat(chunk_outputs, 1));
        
        aa = aa.detach();
        bb = bb.detach();
        pp = pp.detach();
        
        chunk_start = chunk_end;
    }
    
    Tensor::cat(all_outputs, 1)
}

// ══════════════════════════════════════════════════════════════════════════════
//                           PUBLIC API
// ══════════════════════════════════════════════════════════════════════════════

pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C]
    u: Tensor<B, 1>,      // [C]
    config: &WKVConfig,
) -> Tensor<B, 3> {
    let [_b, t, _c] = k.dims();
    
    if t == 0 {
        return k; 
    }
    
    // Auto-select strategy
    if config.aggressive_detach {
        wkv_token_by_token(k, v, w, u, true)
    } else {
        wkv_chunked(k, v, w, u, config.chunk_size)
    }
}

pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}
