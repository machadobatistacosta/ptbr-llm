//! WKV v7 — Data-dependent decay with LoRA-like state evolution
//!
//! Key differences from WKV v4:
//! - Multi-head: state is [H, N, N] per layer (H=n_head, N=head_size)
//! - Decay is DYNAMIC: w = -softplus(-(w0 + w_dynamic)) - 0.5
//! - State update: s = s * exp(w) + s @ ab + vk (LoRA-like adaptation)
//! - Output: y = state @ r (per head)
//!
//! Pure Burn implementation (autodiff-compatible).

use burn::tensor::{backend::Backend, Tensor};

/// WKV-7 forward pass for sequence training.
///
/// All inputs: [T, C] where C = H * N (flattened heads)
/// State: [H, N, N] (float32, maintained across timesteps)
///
/// Arguments:
/// - r: receptance [T, C]
/// - w: log-space decay (already processed: -softplus(-(w0+w))-0.5) [T, C]
/// - k: key (already modified: k * (1 + (a-1)*k_a)) [T, C]
/// - v: value [T, C]
/// - neg_kk: -L2_norm(k * k_k) [T, C]
/// - kk_a: L2_norm(k * k_k) * a [T, C]
/// - state: [H, N, N]
/// - n_head: number of heads
/// - head_size: per-head dimension
///
/// Returns: (output [T, C], updated_state [H, N, N])
pub fn wkv7_seq<B: Backend>(
    r: Tensor<B, 2>,      // [T, C]
    w: Tensor<B, 2>,      // [T, C]
    k: Tensor<B, 2>,      // [T, C]
    v: Tensor<B, 2>,      // [T, C]
    neg_kk: Tensor<B, 2>, // [T, C]
    kk_a: Tensor<B, 2>,   // [T, C]
    state: Tensor<B, 3>,  // [H, N, N]
    n_head: usize,
    head_size: usize,
) -> (Tensor<B, 2>, Tensor<B, 3>) {
    let _device = r.device();
    let t_len = r.dims()[0];
    let c = n_head * head_size;

    let mut current_state = state;
    let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(t_len);

    for t in 0..t_len {
        // Extract timestep: [C]
        let r_t = r.clone().slice([t..t + 1, 0..c]).reshape([c]);
        let w_t = w.clone().slice([t..t + 1, 0..c]).reshape([c]);
        let k_t = k.clone().slice([t..t + 1, 0..c]).reshape([c]);
        let v_t = v.clone().slice([t..t + 1, 0..c]).reshape([c]);
        let neg_kk_t = neg_kk.clone().slice([t..t + 1, 0..c]).reshape([c]);
        let kk_a_t = kk_a.clone().slice([t..t + 1, 0..c]).reshape([c]);

        // Reshape to per-head: [H, N]
        let r_h = r_t.reshape([n_head, head_size]);
        let w_h = w_t.reshape([n_head, head_size]);
        let k_h = k_t.reshape([n_head, head_size]);
        let v_h = v_t.reshape([n_head, head_size]);
        let neg_kk_h = neg_kk_t.reshape([n_head, head_size]);
        let kk_a_h = kk_a_t.reshape([n_head, head_size]);

        // vk = v.view(H,N,1) @ k.view(H,1,N) → outer product [H, N, N]
        let v_col = v_h.clone().reshape([n_head, head_size, 1]); // [H, N, 1]
        let k_row = k_h.reshape([n_head, 1, head_size]);          // [H, 1, N]
        let vk = v_col.matmul(k_row);                             // [H, N, N]

        // ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N) → LoRA state adaptation [H, N, N]
        let neg_kk_col = neg_kk_h.reshape([n_head, head_size, 1]); // [H, N, 1]
        let kk_a_row = kk_a_h.reshape([n_head, 1, head_size]);     // [H, 1, N]
        let ab = neg_kk_col.matmul(kk_a_row);                      // [H, N, N]

        // Decay: exp(w) where w is already in log-space [H, 1, N]
        let w_exp = w_h.exp().reshape([n_head, 1, head_size]); // [H, 1, N]

        // State update: state = state * exp(w) + state @ ab + vk
        // state * exp(w): broadcast [H, N, N] * [H, 1, N] = [H, N, N]
        let state_decayed = current_state.clone().mul(w_exp);
        // state @ ab: [H, N, N] @ [H, N, N] = [H, N, N]
        let state_adapted = current_state.matmul(ab);
        current_state = state_decayed + state_adapted + vk;

        // Output: y = state @ r.view(H,N,1) → [H, N, 1]
        let r_col = r_h.reshape([n_head, head_size, 1]); // [H, N, 1]
        let y_h = current_state.clone().matmul(r_col);     // [H, N, 1]
        let y_flat = y_h.reshape([1, c]);                  // [1, C]
        outputs.push(y_flat);
    }

    // Stack outputs: [T, C]
    let output = Tensor::cat(outputs, 0);

    (output, current_state)
}

/// WKV-7 forward pass for single-step inference.
///
/// All inputs: [C] where C = H * N
/// State: [H, N, N]
///
/// Returns: (output [C], updated_state [H, N, N])
pub fn wkv7_one<B: Backend>(
    r: Tensor<B, 1>,      // [C]
    w: Tensor<B, 1>,      // [C]
    k: Tensor<B, 1>,      // [C]
    v: Tensor<B, 1>,      // [C]
    neg_kk: Tensor<B, 1>, // [C]
    kk_a: Tensor<B, 1>,   // [C]
    state: Tensor<B, 3>,  // [H, N, N]
    n_head: usize,
    head_size: usize,
) -> (Tensor<B, 1>, Tensor<B, 3>) {
    let c = n_head * head_size;

    // Reshape to per-head: [H, N]
    let r_h = r.reshape([n_head, head_size]);
    let w_h = w.reshape([n_head, head_size]);
    let k_h = k.reshape([n_head, head_size]);
    let v_h = v.reshape([n_head, head_size]);
    let neg_kk_h = neg_kk.reshape([n_head, head_size]);
    let kk_a_h = kk_a.reshape([n_head, head_size]);

    // vk = outer(v, k) [H, N, N]
    let v_col = v_h.reshape([n_head, head_size, 1]);
    let k_row = k_h.reshape([n_head, 1, head_size]);
    let vk = v_col.matmul(k_row);

    // ab = outer(-kk, kk*a) [H, N, N]
    let neg_kk_col = neg_kk_h.reshape([n_head, head_size, 1]);
    let kk_a_row = kk_a_h.reshape([n_head, 1, head_size]);
    let ab = neg_kk_col.matmul(kk_a_row);

    // Decay [H, 1, N]
    let w_exp = w_h.exp().reshape([n_head, 1, head_size]);

    // State update
    let state_decayed = state.clone().mul(w_exp);
    let state_adapted = state.matmul(ab);
    let new_state = state_decayed + state_adapted + vk;

    // Output: state @ r [H, N, 1] → [C]
    let r_col = r_h.reshape([n_head, head_size, 1]);
    let y_h = new_state.clone().matmul(r_col);
    let y = y_h.reshape([c]);

    (y, new_state)
}
