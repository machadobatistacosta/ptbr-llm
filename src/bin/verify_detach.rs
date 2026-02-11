use burn::backend::Autodiff;
use burn::tensor::Tensor;

#[cfg(feature = "cuda")]
type MyBackend = burn::backend::cuda_jit::Cuda;

#[cfg(feature = "gpu")]
type MyBackend = burn::backend::wgpu::Wgpu<f32, i32>;

#[cfg(not(any(feature = "cuda", feature = "gpu")))]
type MyBackend = burn::backend::ndarray::NdArray<f32>;

type TestBackend = Autodiff<MyBackend>;

#[cfg(feature = "cuda")]
fn get_device() -> burn::backend::cuda_jit::CudaDevice {
    burn::backend::cuda_jit::CudaDevice::new(0)
}

#[cfg(feature = "gpu")]
fn get_device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::BestAvailable
}

#[cfg(not(any(feature = "cuda", feature = "gpu")))]
fn get_device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

fn main() {
    println!("🔍 Testing .detach() behavior for STE trick...");
    println!();

    let device = get_device();

    // x = 2.0
    let x = Tensor::<TestBackend, 1>::from_data([2.0f32], &device).require_grad();

    // y_main = 3 * x = 6.0  (this is our "Burn loop" path)
    let y_main = x.clone() * 3.0;

    // y_aux = 10.0  (this simulates the "CUDA kernel" output — leaf tensor, no grad)
    let y_aux = Tensor::<TestBackend, 1>::from_data([10.0f32], &device);

    // correction = y_aux - y_main = 10.0 - 6.0 = 4.0
    let correction = y_aux - y_main.clone();

    // STE trick: y_final = y_main + correction.detach()
    // Forward value: 6.0 + 4.0 = 10.0 (correct CUDA value)
    // Backward:      dy/dx should be 3.0 (from y_main = 3*x only)
    let y_final = y_main + correction.detach();

    // Verify forward value
    let fwd_val: f32 = y_final.clone().into_scalar();
    println!("  Forward value: {} (expected 10.0)", fwd_val);

    // Compute gradient
    let grads = y_final.backward();
    let x_grad = x.grad(&grads).unwrap();
    let grad_val: f32 = x_grad.into_scalar();

    println!("  Gradient dx:   {} (expected 3.0)", grad_val);
    println!();

    // Check results
    let fwd_ok = (fwd_val - 10.0f32).abs() < 1e-5;
    let grad_ok = (grad_val - 3.0f32).abs() < 1e-5;
    let grad_zero = grad_val.abs() < 1e-5;

    if fwd_ok && grad_ok {
        println!("✅ .detach() WORKS CORRECTLY!");
        println!("   Forward:  values from 'CUDA' path   ✓");
        println!("   Backward: gradients from 'Burn' path ✓");
        println!("   The STE trick in wkv_optimized.rs is valid.");
    } else if grad_zero {
        println!("❌ .detach() BROKEN — Gradient is ZERO!");
        println!("   This means y_final = y_aux - y_main + y_main cancels out.");
        println!("   The WKV layer receives NO gradients.");
        println!("   CRITICAL: Need to fix detach() or use data round-trip.");
        std::process::exit(1);
    } else {
        println!("❓ UNEXPECTED RESULT");
        println!("   Forward:  {} (expected 10.0) {}", fwd_val, if fwd_ok { "✓" } else { "✗" });
        println!("   Gradient: {} (expected 3.0)  {}", grad_val, if grad_ok { "✓" } else { "✗" });
        std::process::exit(1);
    }
}
