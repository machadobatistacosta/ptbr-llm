//! WKV v4 — Max-Tracking Implementation (AUDIT FIX)
//!
//! Rewritten to match the RWKV-4 CUDA kernel algorithm:
//! - Uses log-space max-tracking (the `o` variable) for numerical stability
//! - No more clamp(-15, 15) limiting model expressiveness
//! - Mathematically equivalent to the reference but numerically superior
//!
//! Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_cuda.cu

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
}

/// WKV v4 — Forward for training (max-tracking, numerically stable)
///
/// Implements the exact algorithm from the RWKV-4 CUDA kernel:
/// ```text
/// p=0, q=0, o=-1e38  (running sums divided by exp(o))
/// for t in 0..T:
///     no = max(o, u+k[t]);  A = exp(o-no);  B = exp(u+k[t]-no)
///     y[t] = (A*p + B*v[t]) / (A*q + B)
///     no = max(w+o, k[t]);  A = exp(w+o-no);  B = exp(k[t]-no)
///     p = A*p + B*v[t];  q = A*q + B;  o = no
/// ```
///
/// Args:
///   k: [batch, seq_len, channels] - key
///   v: [batch, seq_len, channels] - value
///   w: [channels] - log decay rate (NEGATIVE, per-channel)
///   u: [channels] - bonus for current token (per-channel)
///
/// Returns:
///   [batch, seq_len, channels] - weighted values
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    // w: log decay rate per channel (should be negative)
    // u: bonus for current token per channel
    let w_1d = w;
    let u_1d = u;

    // Broadcast shapes for [batch, channels] operations
    let w_bc = w_1d.clone().reshape([1, channels]); // [1, C]
    let u_bc = u_1d.clone().reshape([1, channels]); // [1, C]

    // State: p, q are running sums divided by exp(o)
    // o is the max exponent seen so far (for numerical stability)
    let mut p = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut q = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    // Initialize o to very negative value (exp(o) ≈ 0)
    let mut o = Tensor::<B, 2>::full([batch_size, channels], -1e38_f32, &device);

    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        // Extract k[t], v[t]: [batch, channels]
        let kt = k.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);
        let vt = v.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);

        // === Output computation ===
        // no = max(o, u + k[t])
        let u_plus_kt = u_bc.clone() + kt.clone();
        let no = o.clone().max_pair(u_plus_kt.clone());

        // A = exp(o - no), B = exp(u + k[t] - no)
        let a = (o.clone() - no.clone()).exp();
        let b = (u_plus_kt - no.clone()).exp();

        // y[t] = (A*p + B*v[t]) / max(A*q + B, 1e-6)
        let numerator = a.clone() * p.clone() + b.clone() * vt.clone();
        let denominator = (a * q.clone() + b).clamp_min(1e-6);
        let out_t = numerator / denominator;
        outputs.push(out_t);

        // === State update ===
        // no2 = max(w + o, k[t])
        let w_plus_o = w_bc.clone() + o.clone();
        let no2 = w_plus_o.clone().max_pair(kt.clone());

        // A2 = exp(w + o - no2), B2 = exp(k[t] - no2)
        let a2 = (w_plus_o - no2.clone()).exp();
        let b2 = (kt - no2.clone()).exp();

        // p = A2*p + B2*v[t];  q = A2*q + B2;  o = no2
        p = a2.clone() * p + b2.clone() * vt;
        q = a2 * q + b2;
        o = no2;
    }

    // Stack: [batch, seq_len, channels]
    Tensor::stack(outputs, 1)
}

/// WKV v4 — Forward step for inference (one token at a time, max-tracking)
///
/// State tuple: (p, q, o) where p,q are running sums divided by exp(o)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,      // [batch, channels]
    v: Tensor<B, 2>,      // [batch, channels]
    w: Tensor<B, 1>,      // [channels]
    u: Tensor<B, 1>,      // [channels]
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_batch, channels] = k.dims();
    let (ref mut p, ref mut q, ref mut o) = state;

    let w_bc = w.reshape([1, channels]);
    let u_bc = u.reshape([1, channels]);

    // === Output computation ===
    let u_plus_k = u_bc + k.clone();
    let no = o.clone().max_pair(u_plus_k.clone());

    let a = (o.clone() - no.clone()).exp();
    let b = (u_plus_k - no).exp();

    let numerator = a.clone() * p.clone() + b.clone() * v.clone();
    let denominator = (a * q.clone() + b).clamp_min(1e-6);
    let output = numerator / denominator;

    // === State update ===
    let w_plus_o = w_bc + o.clone();
    let no2 = w_plus_o.clone().max_pair(k.clone());

    let a2 = (w_plus_o - no2.clone()).exp();
    let b2 = (k - no2.clone()).exp();

    *p = a2.clone() * p.clone() + b2.clone() * v;
    *q = a2 * q.clone() + b2;
    *o = no2;

    output
}

#[allow(dead_code)]
pub fn init_state<B: Backend>(
    batch_size: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    (
        Tensor::zeros([batch_size, channels], device),
        Tensor::zeros([batch_size, channels], device),
        // o starts at -1e38 so exp(o) ≈ 0 (no prior state)
        Tensor::full([batch_size, channels], -1e38_f32, device),
    )
}

// Backwards compatibility alias
pub fn wkv_parallel_scan<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    wkv_linear(k, v, w, u, &WKVConfig::for_t4())
}