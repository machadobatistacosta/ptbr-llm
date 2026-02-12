use crate::error::{PtbrError, Result};

// --- Raw bindings (never called directly outside this file) ---
mod ffi {
    #[link(name = "wkv_cuda")]
    extern "C" {
        pub fn wkv_forward_fp32(
            B: i32, T: i32, C: i32,
            w: *const f32, u: *const f32,
            k: *const f32, v: *const f32,
            y: *mut f32,
            aa: *mut f32, bb: *mut f32, pp: *mut f32,
        );
    }
}

// --- Safe public API ---

/// Validated dimensions for a WKV call.
struct WKVDims {
    batch: i32,
    time: i32,
    channels: i32,
}

impl WKVDims {
    fn validate(
        w_len: usize,
        u_len: usize,
        k_len: usize,
        v_len: usize,
        batch: usize,
        time: usize,
        channels: usize,
    ) -> Result<Self> {
        let expected = batch * time * channels;

        if k_len != expected || v_len != expected {
            return Err(PtbrError::ShapeMismatch {
                expected: format!("B*T*C = {expected}"),
                got: format!("k={k_len}, v={v_len}"),
            });
        }
        if w_len != channels || u_len != channels {
            return Err(PtbrError::ShapeMismatch {
                expected: format!("C = {channels}"),
                got: format!("w={w_len}, u={u_len}"),
            });
        }

        Ok(Self {
            batch: batch as i32,
            time: time as i32,
            channels: channels as i32,
        })
    }
}

pub struct WKVState {
    pub aa: Vec<f32>,  // [B, C]
    pub bb: Vec<f32>,  // [B, C]
    pub pp: Vec<f32>,  // [B, C]
}

impl WKVState {
    pub fn new(batch: usize, channels: usize) -> Self {
        let n = batch * channels;
        Self {
            aa: vec![0.0; n],
            bb: vec![0.0; n],
            pp: vec![-1e30; n],  // log-space "negative infinity"
        }
    }
}

#[cfg(feature = "cuda")]
pub fn wkv_forward(
    w: &[f32],          // [C]
    u: &[f32],          // [C]
    k: &[f32],          // [B, T, C]
    v: &[f32],          // [B, T, C]
    y: &mut [f32],      // [B, T, C] output
    state: &mut WKVState,
    batch: usize,
    time: usize,
    channels: usize,
) -> Result<()> {
    let dims = WKVDims::validate(
        w.len(), u.len(), k.len(), v.len(),
        batch, time, channels,
    )?;

    if y.len() != k.len() {
        return Err(PtbrError::ShapeMismatch {
            expected: format!("{}", k.len()),
            got: format!("{}", y.len()),
        });
    }

    unsafe {
        ffi::wkv_forward_fp32(
            dims.batch, dims.time, dims.channels,
            w.as_ptr(), u.as_ptr(),
            k.as_ptr(), v.as_ptr(),
            y.as_mut_ptr(),
            state.aa.as_mut_ptr(),
            state.bb.as_mut_ptr(),
            state.pp.as_mut_ptr(),
        );
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn wkv_forward(
    w: &[f32],
    u: &[f32],
    k: &[f32],
    v: &[f32],
    y: &mut [f32],
    state: &mut WKVState,
    batch: usize,
    time: usize,
    channels: usize,
) -> Result<()> {
    let _dims = WKVDims::validate(
        w.len(), u.len(), k.len(), v.len(),
        batch, time, channels,
    )?;
    
    // CPU fallback — reference implementation of WKV forward pass
    // y_t = (u[i] + k[t] + state) ... similar to original RWKV logic
    // but here we implement the simplified recurrence or the exact FFI logic?
    // The user snippet provided a reference implementation loop.
    
    // Using simple indexing
    let batch = batch;
    let time = time;
    let channels = channels;
    
    for b in 0..batch {
        for t in 0..time {
            for c in 0..channels {
                let idx = b * time * channels + t * channels + c;
                let kk = k[idx];
                let vv = v[idx];
                let ww = w[c];
                let uu = u[c];

                let aa = state.aa[b * channels + c];
                let bb = state.bb[b * channels + c];
                let pp = state.pp[b * channels + c];

                let e1 = pp.max(uu + kk);
                let _e2 = pp.max(ww + pp); // Wait, this variable e2 is unused in user snippet logic? 
                // Let's follow user's snippet logic exactly from prompt
                
                // Snippet:
                // let e1 = (pp.max(uu + kk)) ;
                // let e2 = (pp.max(ww + pp));
                // let wkv = ((uu + kk - e1).exp() * vv + (pp - e1).exp() * aa)
                //    / ((uu + kk - e1).exp() + (pp - e1).exp() * bb);
                // y[idx] = wkv;
                
                let wkv = ((uu + kk - e1).exp() * vv + (pp - e1).exp() * aa)
                    / ((uu + kk - e1).exp() + (pp - e1).exp() * bb);
                
                y[idx] = wkv;

                // State update
                // let e1 = (ww + pp).max(kk);
                // state.aa[..] = ...
                
                let e1 = (ww + pp).max(kk);
                state.aa[b * channels + c] =
                    (ww + pp - e1).exp() * aa + (kk - e1).exp() * vv;
                state.bb[b * channels + c] =
                    (ww + pp - e1).exp() * bb + (kk - e1).exp();
                state.pp[b * channels + c] = e1;
            }
        }
    }

    Ok(())
}
