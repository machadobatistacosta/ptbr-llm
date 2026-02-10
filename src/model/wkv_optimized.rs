//! WKV v4 - Per-channel implementation with CUDA dispatch
//!
//! If libwkv_cuda.so is loaded, uses CUDA kernel for massive speedup.
//! Otherwise falls back to Burn tensor loop (correct but slow).

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

/// WKV v4 - Forward with CUDA dispatch
///
/// If CUDA kernel is available: uses GPU kernel (data roundtrip, ~40x faster)
/// Otherwise: falls back to Burn tensor loop
pub fn wkv_linear<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    _config: &WKVConfig,
) -> Tensor<B, 3> {
    // Try CUDA kernel first
    if let Some(kernel) = wkv_cuda_ffi::get_cuda_kernel() {
        return wkv_linear_cuda(k, v, w, u, kernel);
    }
    
    // Fallback: Burn tensor loop
    wkv_linear_burn(k, v, w, u)
}

/// CUDA kernel path — data roundtrip through CPU
fn wkv_linear_cuda<B: Backend>(
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    kernel: &wkv_cuda_ffi::WkvCudaKernel,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, channels] = k.dims();
    let device = k.device();
    
    // Clamp w, u to match Burn fallback behavior
    let w = w.clamp(-5.0, -0.01);
    let u = u.clamp(-3.0, 3.0);

    // Extract data to CPU (the roundtrip cost ~0.5ms is dwarfed by the kernel speedup)
    let w_data: Vec<f32> = w.into_data().iter::<f32>().collect();
    let u_data: Vec<f32> = u.into_data().iter::<f32>().collect();
    let k_data: Vec<f32> = k.into_data().iter::<f32>().collect();
    let v_data: Vec<f32> = v.into_data().iter::<f32>().collect();

    // Run CUDA kernel
    let y_data = kernel.forward_pass(
        batch_size, seq_len, channels,
        &w_data, &u_data, &k_data, &v_data,
    );

    // Create output tensor from result
    let y_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(y_data, [batch_size * seq_len * channels]),
        &device,
    );
    
    y_tensor.reshape([batch_size, seq_len, channels]).clamp(-20.0, 20.0)
}

/// Burn tensor loop fallback (correct but slow)
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

    let w_exp = w.exp().reshape([1, 1, channels]);
    let u_broad = u.reshape([1, channels]);

    let mut state_num = Tensor::<B, 2>::zeros([batch_size, channels], &device);
    let mut state_den = Tensor::<B, 2>::zeros([batch_size, channels], &device);

    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let kt = k.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);
        let vt = v.clone().slice([0..batch_size, t..t + 1, 0..channels])
            .reshape([batch_size, channels]);

        let uk = u_broad.clone() + kt.clone();
        let uk_exp = uk.clamp(-15.0, 15.0).exp();

        let num = state_num.clone() + uk_exp.clone() * vt.clone();
        let den = state_den.clone() + uk_exp + 1e-6;

        let out_t = num / den;
        outputs.push(out_t.clamp(-20.0, 20.0));

        let kt_exp = kt.clamp(-15.0, 15.0).exp();
        let w_exp_2d = w_exp.clone().reshape([1, channels]);

        state_num = state_num * w_exp_2d.clone() + kt_exp.clone() * vt;
        state_den = state_den * w_exp_2d + kt_exp;
    }

    Tensor::stack(outputs, 1)
}

/// WKV v4 - Forward step para inferência (um token por vez)
pub fn wkv_step<B: Backend>(
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    w: Tensor<B, 1>,
    u: Tensor<B, 1>,
    state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
) -> Tensor<B, 2> {
    let [_batch, channels] = k.dims();
    let (ref mut state_num, ref mut state_den, ref mut _last_k) = state;

    let w = w.clamp(-5.0, -0.01);
    let u = u.clamp(-3.0, 3.0);

    let w_exp = w.exp().reshape([1, channels]);
    let u_broad = u.reshape([1, channels]);

    let uk = u_broad + k.clone();
    let uk_exp = uk.clamp(-15.0, 15.0).exp();

    let num = state_num.clone() + uk_exp.clone() * v.clone();
    let den = state_den.clone() + uk_exp + 1e-6;
    let output = (num / den).clamp(-20.0, 20.0);

    let kt_exp = k.clamp(-15.0, 15.0).exp();
    *state_num = state_num.clone() * w_exp.clone() + kt_exp.clone() * v;
    *state_den = state_den.clone() * w_exp + kt_exp;

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
        Tensor::zeros([batch_size, channels], device),
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