//! CUDA FFI wrapper for WKV-4 kernel
//!
//! Loads libwkv_cuda.so at runtime and provides safe Rust API
//! for calling the CUDA forward/backward kernels.
//!
//! Build the CUDA kernel first: `bash cuda/build.sh`

use std::ffi::c_void;
use std::path::Path;
use std::sync::OnceLock;

// Type aliases for FFI function pointers
type WkvForwardFn = unsafe extern "C" fn(
    i32, i32, i32,                                     // B, T, C
    *mut f32, *mut f32, *mut f32, *mut f32,            // w, u, k, v
    *mut f32,                                           // y (output)
);

type WkvBackwardFn = unsafe extern "C" fn(
    i32, i32, i32,                                     // B, T, C
    *mut f32, *mut f32, *mut f32, *mut f32,            // w, u, k, v
    *mut f32,                                           // gy (grad output)
    *mut f32, *mut f32, *mut f32, *mut f32,            // gw, gu, gk, gv
);

type CudaMallocFn = unsafe extern "C" fn(usize) -> *mut c_void;
type CudaFreeFn = unsafe extern "C" fn(*mut c_void);
type CudaCopyH2DFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize);
type CudaCopyD2HFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize);
type CudaZeroFn = unsafe extern "C" fn(*mut c_void, usize);

/// CUDA kernel handle — loaded once, used everywhere
pub struct WkvCudaKernel {
    _lib: libloading::Library,
    forward: WkvForwardFn,
    backward: WkvBackwardFn,
    cuda_malloc: CudaMallocFn,
    cuda_free: CudaFreeFn,
    copy_h2d: CudaCopyH2DFn,
    copy_d2h: CudaCopyD2HFn,
    cuda_zero: CudaZeroFn,
}

// Safety: The CUDA kernel is thread-safe (each call gets its own GPU memory)
unsafe impl Send for WkvCudaKernel {}
unsafe impl Sync for WkvCudaKernel {}

/// Global singleton — loaded on first use
static CUDA_KERNEL: OnceLock<Option<WkvCudaKernel>> = OnceLock::new();

impl WkvCudaKernel {
    /// Try to load the CUDA kernel from the given path
    pub fn load(lib_path: &str) -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new(lib_path)
                .map_err(|e| format!("Failed to load {}: {}", lib_path, e))?;
            
            let forward: libloading::Symbol<WkvForwardFn> = lib.get(b"wkv_forward")
                .map_err(|e| format!("wkv_forward not found: {}", e))?;
            let backward: libloading::Symbol<WkvBackwardFn> = lib.get(b"wkv_backward")
                .map_err(|e| format!("wkv_backward not found: {}", e))?;
            let cuda_malloc: libloading::Symbol<CudaMallocFn> = lib.get(b"wkv_cuda_malloc")
                .map_err(|e| format!("wkv_cuda_malloc not found: {}", e))?;
            let cuda_free: libloading::Symbol<CudaFreeFn> = lib.get(b"wkv_cuda_free")
                .map_err(|e| format!("wkv_cuda_free not found: {}", e))?;
            let copy_h2d: libloading::Symbol<CudaCopyH2DFn> = lib.get(b"wkv_cuda_copy_h2d")
                .map_err(|e| format!("wkv_cuda_copy_h2d not found: {}", e))?;
            let copy_d2h: libloading::Symbol<CudaCopyD2HFn> = lib.get(b"wkv_cuda_copy_d2h")
                .map_err(|e| format!("wkv_cuda_copy_d2h not found: {}", e))?;
            let cuda_zero: libloading::Symbol<CudaZeroFn> = lib.get(b"wkv_cuda_zero")
                .map_err(|e| format!("wkv_cuda_zero not found: {}", e))?;

            Ok(Self {
                forward: *forward,
                backward: *backward,
                cuda_malloc: *cuda_malloc,
                cuda_free: *cuda_free,
                copy_h2d: *copy_h2d,
                copy_d2h: *copy_d2h,
                cuda_zero: *cuda_zero,
                _lib: lib,
            })
        }
    }

    /// Allocate GPU buffer
    fn gpu_alloc(&self, n_floats: usize) -> *mut f32 {
        unsafe { (self.cuda_malloc)(n_floats * 4) as *mut f32 }
    }

    /// Free GPU buffer
    fn gpu_free(&self, ptr: *mut f32) {
        unsafe { (self.cuda_free)(ptr as *mut c_void) }
    }

    /// Upload f32 slice to GPU
    fn upload(&self, dst: *mut f32, src: &[f32]) {
        unsafe { (self.copy_h2d)(dst as *mut c_void, src.as_ptr() as *const c_void, src.len() * 4) }
    }

    /// Download from GPU to f32 vec
    fn download(&self, src: *mut f32, n: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; n];
        unsafe { (self.copy_d2h)(result.as_mut_ptr() as *mut c_void, src as *const c_void, n * 4) }
        result
    }

    /// Zero GPU buffer
    fn gpu_zero(&self, ptr: *mut f32, n_floats: usize) {
        unsafe { (self.cuda_zero)(ptr as *mut c_void, n_floats * 4) }
    }

    /// WKV Forward pass via CUDA kernel
    ///
    /// Args:
    ///   w: [C] decay rates (negative)
    ///   u: [C] bonus for current token
    ///   k: [B*T*C] keys (row-major: batch, time, channel)
    ///   v: [B*T*C] values
    ///
    /// Returns: [B*T*C] output
    pub fn forward_pass(
        &self, b: usize, t: usize, c: usize,
        w: &[f32], u: &[f32], k: &[f32], v: &[f32],
    ) -> Vec<f32> {
        let btc = b * t * c;

        // Allocate GPU buffers
        let d_w = self.gpu_alloc(c);
        let d_u = self.gpu_alloc(c);
        let d_k = self.gpu_alloc(btc);
        let d_v = self.gpu_alloc(btc);
        let d_y = self.gpu_alloc(btc);

        // Upload
        self.upload(d_w, w);
        self.upload(d_u, u);
        self.upload(d_k, k);
        self.upload(d_v, v);

        // Launch kernel
        unsafe {
            (self.forward)(b as i32, t as i32, c as i32, d_w, d_u, d_k, d_v, d_y);
        }

        // Download result
        let result = self.download(d_y, btc);

        // Free
        self.gpu_free(d_w);
        self.gpu_free(d_u);
        self.gpu_free(d_k);
        self.gpu_free(d_v);
        self.gpu_free(d_y);

        result
    }

    /// WKV Backward pass via CUDA kernel
    ///
    /// Returns: (gw[B*C], gu[B*C], gk[B*T*C], gv[B*T*C])
    pub fn backward_pass(
        &self, b: usize, t: usize, c: usize,
        w: &[f32], u: &[f32], k: &[f32], v: &[f32], gy: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let btc = b * t * c;
        let bc = b * c;

        // Allocate GPU buffers
        let d_w = self.gpu_alloc(c);
        let d_u = self.gpu_alloc(c);
        let d_k = self.gpu_alloc(btc);
        let d_v = self.gpu_alloc(btc);
        let d_gy = self.gpu_alloc(btc);
        let d_gw = self.gpu_alloc(bc);
        let d_gu = self.gpu_alloc(bc);
        let d_gk = self.gpu_alloc(btc);
        let d_gv = self.gpu_alloc(btc);

        // Upload inputs
        self.upload(d_w, w);
        self.upload(d_u, u);
        self.upload(d_k, k);
        self.upload(d_v, v);
        self.upload(d_gy, gy);

        // Zero gradient buffers
        self.gpu_zero(d_gw, bc);
        self.gpu_zero(d_gu, bc);

        // Launch kernel
        unsafe {
            (self.backward)(
                b as i32, t as i32, c as i32,
                d_w, d_u, d_k, d_v, d_gy,
                d_gw, d_gu, d_gk, d_gv,
            );
        }

        // Download results
        let gw = self.download(d_gw, bc);
        let gu = self.download(d_gu, bc);
        let gk = self.download(d_gk, btc);
        let gv = self.download(d_gv, btc);

        // Free
        self.gpu_free(d_w);
        self.gpu_free(d_u);
        self.gpu_free(d_k);
        self.gpu_free(d_v);
        self.gpu_free(d_gy);
        self.gpu_free(d_gw);
        self.gpu_free(d_gu);
        self.gpu_free(d_gk);
        self.gpu_free(d_gv);

        (gw, gu, gk, gv)
    }
}

/// Get the global CUDA kernel instance (loaded lazily)
pub fn get_cuda_kernel() -> Option<&'static WkvCudaKernel> {
    CUDA_KERNEL.get_or_init(|| {
        // Try multiple paths for the .so file
        let paths = [
            "libwkv_cuda.so",
            "./libwkv_cuda.so",
            "/kaggle/working/ptbr-llm/libwkv_cuda.so",
            "/kaggle/working/libwkv_cuda.so",
        ];
        
        for path in &paths {
            if Path::new(path).exists() {
                match WkvCudaKernel::load(path) {
                    Ok(kernel) => {
                        eprintln!("  ✅ CUDA WKV kernel loaded: {}", path);
                        return Some(kernel);
                    }
                    Err(e) => {
                        eprintln!("  ⚠️ CUDA kernel load failed ({}): {}", path, e);
                    }
                }
            }
        }
        
        eprintln!("  ℹ️ CUDA WKV kernel not found, using Burn fallback (slow)");
        None
    }).as_ref()
}

/// Check if CUDA kernel is available
pub fn cuda_kernel_available() -> bool {
    get_cuda_kernel().is_some()
}
