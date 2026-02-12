//! WKV v4 - Per-channel with CUDA-accelerated forward + Burn autodiff backward
//!
//! Strategy: STE (Straight-Through Estimator) trick
//!   Forward VALUES come from CUDA kernel (numerically superior max-tracking)
//!   Forward GRAPH comes from Burn loop (autodiff-compatible)
//!   y_final = y_burn + (y_cuda - y_burn).detach()
//!   → Values of y_cuda, gradients from y_burn
//!
//! If CUDA kernel not found (feature "cuda" disabled), pure Burn loop.

use burn::tensor::{backend::Backend, Tensor};

use super::wkv_cuda_ffi;

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

/// WKV v4 - Forward with CUDA + autodiff support
///
/// If CUDA kernel available: uses STE trick for correct values + correct gradients
/// Otherwise: pure Burn loop
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    // Burn loop builds autodiff graph -> gradients are consistent with values
    let y_burn = wkv_linear_burn(k.clone(), v.clone(), w.clone(), u.clone());
    
    // If CUDA feature is enabled, use the kernel for better precision/performance on forward pass
    #[cfg(feature = "cuda")]
    {
        // Try to run CUDA kernel
        match wkv_cuda_forward(k, v, w, u) {
            Ok(y_cuda) => {
                // STE Trick: y = y_burn + (y_cuda - y_burn).detach()
                // Value is y_cuda, Gradient is y_burn's gradient
                y_burn.clone() + (y_cuda - y_burn).detach()
            }
            Err(e) => {
                eprintln!("⚠️ CUDA WKV kernel failed, using Burn fallback: {}", e);
                y_burn
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        y_burn
    }
}

/// CUDA kernel forward — extracts data, runs kernel, returns Burn tensor (leaf, no grad)
#[cfg(feature = "cuda")]
fn wkv_cuda_forward<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> crate::error::Result<Tensor<B, 3>> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Clamp to match Burn loop behavior
    let w_clamped = w.clamp(-5.0, -0.01);
    let u_clamped = u.clamp(-3.0, 3.0);

    // Extract CPU data
    let w_data: Vec<f32> = w_clamped.into_data().iter::<f32>().collect();
    let u_data: Vec<f32> = u_clamped.into_data().iter::<f32>().collect();
    let k_data: Vec<f32> = k.into_data().iter::<f32>().collect();
    let v_data: Vec<f32> = v.into_data().iter::<f32>().collect();

    // Prepare output buffer
    let mut y_data = vec![0.0; batch_size * seq_len * channels];
    let mut state = wkv_cuda_ffi::WKVState::new(batch_size, channels);

    // Run CUDA kernel via safe FFI
    wkv_cuda_ffi::wkv_forward(
        &w_data, &u_data, &k_data, &v_data,
        &mut y_data,
        &mut state,
        batch_size, seq_len, channels
    )?;

    // Create result tensor
    let y_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(y_data, [batch_size * seq_len * channels]),
        &device,
    );
    
    Ok(y_tensor.reshape([batch_size, seq_len, channels]).clamp(-20.0, 20.0))
}

/// Burn tensor loop — EMA-style WKV (autodiff-compatible, builds graph)
/// Matches CUDA kernel's log-space formulation for optional precision
fn wkv_linear_burn<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();

    let w = w.clamp(-5.0, -0.01);
    let u = u.clamp(-3.0, 3.0);

    let u_broad = u.reshape([1, channels]);
    let w_broad = w.reshape([1, channels]);

    // Log-space state — matches CUDA kernel's (aa, bb, pp) exactly
    let mut aa = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut bb = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    // pp = -1e30 means "log(0)" — no prior information
    let neg_inf = Tensor::<B, 2>::zeros([batch_size, channels], &device)
        .add_scalar(-1e30);
    let mut pp = neg_inf;

    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let kt = k.clone()
            .slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);
        let vt = v.clone()
            .slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);

        // --- Output computation (matches CUDA) ---
        let uk = u_broad.clone() + kt.clone();
        // e1 = max(pp, u + k) — for numerical stability
        let e1 = pp.clone().max_pair(uk.clone());

        let a_term = (uk.clone() - e1.clone()).exp() * vt.clone();
        let b_term = (pp.clone() - e1.clone()).exp() * aa.clone();
        let num = a_term + b_term;

        let c_term = (uk - e1.clone()).exp();
        let d_term = (pp.clone() - e1).exp() * bb.clone();
        let den = c_term + d_term + 1e-8;

        let out_t = (num / den).clamp(-20.0, 20.0);
        outputs.push(out_t);

        // --- State update (matches CUDA) ---
        // e2 = max(w + pp, k)
        let wp = w_broad.clone() + pp.clone();
        let e2 = wp.clone().max_pair(kt.clone());

        aa = (wp.clone() - e2.clone()).exp() * aa
           + (kt.clone() - e2.clone()).exp() * vt;
        bb = (wp - e2.clone()).exp() * bb
           + (kt - e2.clone()).exp();
        pp = e2;
    }

    Tensor::stack(outputs, 1)
}

/// WKV v4 - Forward step for inference (one token at a time)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,    // [batch, channels]
    v: Tensor<B, 2>,
    w: Tensor<B, 1>,    // [channels]
    u: Tensor<B, 1>,
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    // state = (aa, bb, pp) in log-space
) -> Tensor<B, 2> {
    let [_batch, channels] = k.dims();
    let (ref mut aa, ref mut bb, ref mut pp) = state;

    let w = w.clamp(-5.0, -0.01);
    let u = u.clamp(-3.0, 3.0);


    let u_broad = u.reshape([1, channels]);
    let w_broad = w.reshape([1, channels]);

    // Output
    let uk = u_broad + k.clone();
    let e1 = pp.clone().max_pair(uk.clone());

    let num = (uk.clone() - e1.clone()).exp() * v.clone()
            + (pp.clone() - e1.clone()).exp() * aa.clone();
    let den = (uk - e1.clone()).exp()
            + (pp.clone() - e1).exp() * bb.clone()
            + 1e-8;

    let output = (num / den).clamp(-20.0, 20.0);

    // State update
    let wp = w_broad + pp.clone();
    let e2 = wp.clone().max_pair(k.clone());

    *aa = (wp.clone() - e2.clone()).exp() * aa.clone()
        + (k.clone() - e2.clone()).exp() * v;
    *bb = (wp - e2.clone()).exp() * bb.clone()
        + (k - e2.clone()).exp();
    *pp = e2;

    output
}

#[allow(dead_code)]
pub fn init_state<B: Backend>(
    batch_size: usize,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    (
        Tensor::zeros([batch_size, channels], device),       // aa
        Tensor::zeros([batch_size, channels], device),       // bb
        Tensor::zeros([batch_size, channels], device)        // pp = -1e30
            .add_scalar(-1e30),
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