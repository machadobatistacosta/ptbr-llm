//! Backend Selector
//!
//! Selects the appropriate Burn backend based on feature flags.
//! Only one backend can be active at a time.

use burn::backend::Autodiff;

// ============ CUDA BACKEND ============
#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu")))]
mod backend_impl {
    pub use burn::backend::cuda_jit::{Cuda, CudaDevice};
    pub type MyBackend = Cuda;
    
    pub fn get_device() -> CudaDevice {
        CudaDevice::new(0)
    }
}

// ============ WGPU BACKEND ============
#[cfg(all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu")))]
mod backend_impl {
    pub use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type MyBackend = Wgpu<f32, i32>;
    
    pub fn get_device() -> WgpuDevice {
        WgpuDevice::BestAvailable
    }
}

// ============ LIBTORCH BACKEND ============
#[cfg(all(feature = "torch", not(feature = "cuda"), not(feature = "gpu"), not(feature = "cpu")))]
mod backend_impl {
    pub use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type MyBackend = LibTorch<f32>;
    
    pub fn get_device() -> LibTorchDevice {
        LibTorchDevice::Cuda(0)
    }
}

// ============ CPU (NDARRAY) BACKEND ============
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu"), not(feature = "torch")))]
mod backend_impl {
    pub use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type MyBackend = NdArray;
    
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

// ============ FALLBACK (NO FEATURE) ============
#[cfg(not(any(
    all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu"), not(feature = "torch")),
    all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu"), not(feature = "torch")),
    all(feature = "torch", not(feature = "cuda"), not(feature = "gpu"), not(feature = "cpu")),
    all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu"), not(feature = "torch"))
)))]
mod backend_impl {
    pub use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type MyBackend = NdArray;
    
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

// ============ PUBLIC EXPORTS ============
pub use backend_impl::{MyBackend, get_device};

/// Backend with autodiff for training
pub type TrainBackend = Autodiff<MyBackend>;
