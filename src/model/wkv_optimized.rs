// src/model/wkv_optimized.rs
//! WKV Sequencial Puro - O(T) RAM, Matematicamente Exato
//!
//! Esta implementação usa um loop sequencial simples.
//! Vantagens:
//! 1. Garante OOM zero (uso de memória é linear, não quadrático).
//! 2. Matematicamente idêntico à formulação RWKV original.
//! 3. Numericamente estável (usa log-space para steps).
//!
//! Desvantagens:
//! 1. Throughput menor que chunks em GPUs top-tier (H100), mas ideal para T4.

use burn::tensor::{backend::Backend, Tensor};

/// Epsilon para estabilidade numérica
const EPS: f32 = 1e-9;

/// Configuração do WKV
#[derive(Debug, Clone)]
pub struct WKVConfig {
    pub chunk_size: usize, // Ignorado no modo sequencial
    pub use_float64_accumulator: bool,
    pub parallel_heads: bool,
}

impl Default for WKVConfig {
    fn default() -> Self {
        Self {
            chunk_size: 0, // Não usado
            use_float64_accumulator: true,
            parallel_heads: true,
        }
    }
}

/// WKV Sequencial - Implementação Canônica
///
/// wkv_t = (aa + e^{u+k} * v) / (bb + e^{u+k})
/// state_next = state * e^{-w} + e^k * v
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,      // [B, T, C]
    v: Tensor<B, 3>,      // [B, T, C]
    w: Tensor<B, 1>,      // [C] Time decay
    u: Tensor<B, 1>,      // [C] Time first
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [b, t, c] = k.dims();
    let device = k.device();

    // Parâmetros no formato correto
    // w_log = -exp(w)
    // No RWKV oficial, w é o parametro log-decay. A aplicação é state * exp(w_log)
    // Assumimos w vindo do modelo como o parametro bruto
    let w_log = w.clone().exp().neg().reshape([1, c]); // [1, C]

    // u shape [1, C]
    let u_vec = u.reshape([1, c]);

    // Inicializa estados
    // aa: numerador acumulado
    // bb: denominador acumulado
    // pp: expoente máximo para estabilidade (log-sum-exp)
    let mut aa = Tensor::<B, 2>::zeros([b, c], &device);
    let mut bb = Tensor::<B, 2>::zeros([b, c], &device);
    let mut pp = Tensor::<B, 2>::zeros([b, c], &device) - 1e30; // Start at -inf

    let mut outputs = Vec::with_capacity(t);

    for i in 0..t {
        // Pega slice do tempo t: [B, 1, C] -> [B, C]
        let k_t = k.clone().slice([0..b, i..i+1, 0..c]).reshape([b, c]);
        let v_t = v.clone().slice([0..b, i..i+1, 0..c]).reshape([b, c]);

        // 1. Calcula Output do passo t (Atenção ao estado atual)
        // p = max(pp, k_t + u)
        let ww = u_vec.clone() + k_t.clone(); // k_t + u
        let p = pp.clone().max_pair(ww.clone());
        
        // e1 = exp(pp - p)  (peso do estado anterior)
        // e2 = exp(ww - p)  (peso do token atual)
        let e1 = (pp.clone() - p.clone()).exp();
        let e2 = (ww - p.clone()).exp();
        
        // wkv = (e1*aa + e2*v) / (e1*bb + e2)
        let num = (aa.clone() * e1.clone()) + (v_t.clone() * e2.clone());
        let den = (bb.clone() * e1) + e2;
        let wkv_t = num / (den + EPS);
        
        outputs.push(wkv_t);

        // 2. Atualiza Estado para t+1
        // ww = pp + w_log (decay aplicado ao max_exp anterior)
        // p = max(ww, k_t)
        let ww = pp.clone() + w_log.clone();
        let p = ww.clone().max_pair(k_t.clone());
        
        // e1 = exp(ww - p)
        // e2 = exp(k_t - p)
        let e1 = (ww - p.clone()).exp();
        let e2 = (k_t - p.clone()).exp();
        
        // aa = e1*aa + e2*v
        // bb = e1*bb + e2
        aa = (aa * e1.clone()) + (v_t * e2.clone());
        bb = (bb * e1) + e2;
        pp = p;
    }

    // Stack output list into Tensor [B, T, C]
    Tensor::stack(outputs, 1)
}

/// Fallback compatibility
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::default())
}