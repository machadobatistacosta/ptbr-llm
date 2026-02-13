//! RWKV-7 "Goose" — Full implementation in Rust/Burn
//!
//! Architecture differences from v4:
//! - Multi-head WKV with head_size=64
//! - 7 projections: r, w, k, v, a, g (+ v_first mechanism)
//! - Data-dependent decay: w = -softplus(-(w0 + w_dynamic)) - 0.5
//! - LoRA-like state evolution: s = s*decay + s@ab + vk
//! - Lerp token shift with learned factors (not concat)
//! - GroupNorm on attention output (per head)
//! - Simplified channel mixing: single lerp factor, squared ReLU

use super::config::RWKVConfig;
use super::wkv_v7::{wkv7_seq, wkv7_one};
use burn::{
    module::{Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Initializer,
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, activation, Int, Tensor},
};

// ============================================================================
// TimeMixing V7 — The core attention mechanism
// ============================================================================

#[derive(Module, Debug)]
pub struct TimeMixingV7<B: Backend> {
    // Lerp factors for token shift (learned, per-component)
    x_r: Param<Tensor<B, 1>>,  // [C]
    x_w: Param<Tensor<B, 1>>,  // [C]
    x_k: Param<Tensor<B, 1>>,  // [C]
    x_v: Param<Tensor<B, 1>>,  // [C]
    x_a: Param<Tensor<B, 1>>,  // [C]
    x_g: Param<Tensor<B, 1>>,  // [C]

    // Main projections
    w_r: Linear<B>,  // receptance: [C, C]
    w_k: Linear<B>,  // key: [C, C]
    w_v: Linear<B>,  // value: [C, C]
    w_o: Linear<B>,  // output: [C, C]

    // Low-rank decay projection: w = tanh(xw @ w1) @ w2
    w0: Param<Tensor<B, 1>>,   // [C] bias
    w1: Linear<B>,              // [C, low_rank]
    w2: Linear<B>,              // [low_rank, C]

    // Low-rank state adaptation: a = sigmoid(a0 + (xa @ a1) @ a2)
    a0: Param<Tensor<B, 1>>,   // [C] bias
    a1: Linear<B>,              // [C, low_rank]
    a2: Linear<B>,              // [low_rank, C]

    // v_first mechanism: v = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)
    v0: Param<Tensor<B, 1>>,   // [C]
    v1: Linear<B>,              // [C, low_rank]
    v2: Linear<B>,              // [low_rank, C]

    // Gate: g = sigmoid(xg @ g1) @ g2
    g1: Linear<B>,              // [C, low_rank]
    g2: Linear<B>,              // [low_rank, C]

    // Per-channel learned scaling
    k_k: Param<Tensor<B, 1>>,  // [C] key normalization scaling
    k_a: Param<Tensor<B, 1>>,  // [C] key-adaptation scaling
    r_k: Param<Tensor<B, 1>>,  // [C] bonus term scaling (flattened [H*N])

    // GroupNorm for output (num_groups = n_head)
    ln_x_weight: Param<Tensor<B, 1>>,  // [C]
    ln_x_bias: Param<Tensor<B, 1>>,    // [C]

    // Architecture params
    n_head: usize,
    head_size: usize,
    d_model: usize,
    layer_id: usize,
}

impl<B: Backend> TimeMixingV7<B> {
    pub fn new(config: &RWKVConfig, layer_id: usize, device: &B::Device) -> Self {
        let c = config.d_model;
        let low_rank = config.effective_low_rank();
        let n_head = config.n_head();
        let head_size = config.head_size;

        // Lerp factors initialized to 0.5 (equal mix of current and shifted)
        let half = Tensor::zeros([c], device).add_scalar(0.5);
        let x_r = Param::from_tensor(half.clone());
        let x_w = Param::from_tensor(half.clone());
        let x_k = Param::from_tensor(half.clone());
        let x_v = Param::from_tensor(half.clone());
        let x_a = Param::from_tensor(half.clone());
        let x_g = Param::from_tensor(half);

        // Main projections: Xavier uniform init
        let init = Initializer::XavierUniform { gain: 1.0 };
        let w_r = LinearConfig::new(c, c)
            .with_bias(false)
            .with_initializer(init.clone())
            .init(device);
        let w_k = LinearConfig::new(c, c)
            .with_bias(false)
            .with_initializer(init.clone())
            .init(device);
        let w_v = LinearConfig::new(c, c)
            .with_bias(false)
            .with_initializer(init.clone())
            .init(device);
        let w_o = LinearConfig::new(c, c)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        // Low-rank projections (small init for stability)
        let small_init = Initializer::Uniform { min: -0.01, max: 0.01 };
        let zero_init = Initializer::Zeros;

        let w0 = Param::from_tensor(Tensor::zeros([c], device));
        let w1 = LinearConfig::new(c, low_rank).with_bias(false).with_initializer(small_init.clone()).init(device);
        let w2 = LinearConfig::new(low_rank, c).with_bias(false).with_initializer(zero_init.clone()).init(device);

        let a0 = Param::from_tensor(Tensor::zeros([c], device));
        let a1 = LinearConfig::new(c, low_rank).with_bias(false).with_initializer(small_init.clone()).init(device);
        let a2 = LinearConfig::new(low_rank, c).with_bias(false).with_initializer(zero_init.clone()).init(device);

        let v0 = Param::from_tensor(Tensor::zeros([c], device));
        let v1 = LinearConfig::new(c, low_rank).with_bias(false).with_initializer(small_init.clone()).init(device);
        let v2 = LinearConfig::new(low_rank, c).with_bias(false).with_initializer(zero_init.clone()).init(device);

        let g1 = LinearConfig::new(c, low_rank).with_bias(false).with_initializer(small_init).init(device);
        let g2 = LinearConfig::new(low_rank, c).with_bias(false).with_initializer(zero_init).init(device);

        // Per-channel scaling initialized to ones for k_k, zeros for others
        let k_k = Param::from_tensor(Tensor::ones([c], device));
        let k_a = Param::from_tensor(Tensor::zeros([c], device));
        let r_k = Param::from_tensor(Tensor::zeros([n_head * head_size], device));

        // GroupNorm: weight=ones, bias=zeros
        let ln_x_weight = Param::from_tensor(Tensor::ones([c], device));
        let ln_x_bias = Param::from_tensor(Tensor::zeros([c], device));

        Self {
            x_r, x_w, x_k, x_v, x_a, x_g,
            w_r, w_k, w_v, w_o,
            w0, w1, w2,
            a0, a1, a2,
            v0, v1, v2,
            g1, g2,
            k_k, k_a, r_k,
            ln_x_weight, ln_x_bias,
            n_head, head_size, d_model: c, layer_id,
        }
    }

    /// Forward for training (sequence mode)
    /// x: [B, T, C], x_prev: [B, C] (last token of previous chunk, zeros if first)
    /// v_first: [B, T, C] (v from layer 0, carried through all layers)
    /// state: [B, H, N, N]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,          // [B, T, C]
        x_prev: Tensor<B, 2>,     // [B, C]
        v_first: Option<Tensor<B, 3>>,  // [B, T, C] or None (layer 0)
        state: Tensor<B, 4>,      // [B, H, N, N]
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 3>) {
        let [b, t, c] = x.dims();
        let h = self.n_head;
        let n = self.head_size;

        // Token shift: xx = x_prev - x (shifted difference)
        // x_shifted = cat([x_prev.unsqueeze(1), x[:, :-1, :]], dim=1)
        let x_shifted = if t > 1 {
            let prev = x_prev.clone().reshape([b, 1, c]);
            let x_trunc = x.clone().slice([0..b, 0..t - 1, 0..c]);
            Tensor::cat(vec![prev, x_trunc], 1) // [B, T, C]
        } else {
            x_prev.clone().reshape([b, 1, c])
        };
        let xx = x_shifted - x.clone(); // [B, T, C]

        // Lerp mix: xr = x + xx * x_r (broadcast x_r [C] over [B, T, C])
        let xr = x.clone() + xx.clone() * self.x_r.val().unsqueeze::<2>().unsqueeze::<3>(); // [B, T, C]
        let xw = x.clone() + xx.clone() * self.x_w.val().unsqueeze::<2>().unsqueeze::<3>();
        let xk = x.clone() + xx.clone() * self.x_k.val().unsqueeze::<2>().unsqueeze::<3>();
        let xv = x.clone() + xx.clone() * self.x_v.val().unsqueeze::<2>().unsqueeze::<3>();
        let xa = x.clone() + xx.clone() * self.x_a.val().unsqueeze::<2>().unsqueeze::<3>();
        let xg = x.clone() + xx * self.x_g.val().unsqueeze::<2>().unsqueeze::<3>();

        // === Projections ===
        // r: [B, T, C]
        let r = self.w_r.forward(xr);
        // k: [B, T, C]
        let k_raw = self.w_k.forward(xk);
        // v: [B, T, C]
        let v = self.w_v.forward(xv.clone());

        // w (decay): tanh(xw @ w1) @ w2, then -softplus(-(w0 + w)) - 0.5
        let w_dynamic = self.w2.forward(activation::tanh(self.w1.forward(xw)));
        let w0_expanded = self.w0.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let w_pre = w0_expanded + w_dynamic; // [B, T, C]
        // w = -softplus(-(w0 + w)) - 0.5
        // softplus(x) = log(1 + exp(x)), so -softplus(-x) = log(sigmoid(x))
        let w = w_pre.neg().exp().add_scalar(1.0).log().neg().add_scalar(-0.5); // [B, T, C]

        // a: sigmoid(a0 + (xa @ a1) @ a2) [B, T, C]
        let a0_expanded = self.a0.val().unsqueeze::<2>().unsqueeze::<3>();
        let a = activation::sigmoid(a0_expanded + self.a2.forward(self.a1.forward(xa)));

        // g: sigmoid(xg @ g1) @ g2 [B, T, C]
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        // === Key normalization ===
        // kk = L2_norm((k * k_k).view(B*T, H, N)) per head
        let k_k_val = self.k_k.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let k_scaled = (k_raw.clone() * k_k_val).reshape([b * t, h, n]); // [B*T, H, N]
        // L2 norm per head: norm over last dim
        let k_norm = k_scaled.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-12); // [B*T, H, 1]
        let kk = k_scaled / k_norm; // [B*T, H, N]
        let kk = kk.reshape([b, t, c]); // [B, T, C]

        // k = k * (1 + (a - 1) * k_a)
        let k_a_val = self.k_a.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let device = x.device();
        let k = k_raw * (Tensor::<B, 3>::ones([b, t, c], &device) + (a.clone() - 1.0) * k_a_val);

        // === v_first mechanism ===
        let (v_final, v_first_out) = if self.layer_id == 0 {
            // Layer 0: v_first = v
            (v.clone(), v.clone())
        } else {
            // Other layers: v = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)
            let v0_expanded = self.v0.val().unsqueeze::<2>().unsqueeze::<3>();
            let v_gate = activation::sigmoid(v0_expanded + self.v2.forward(self.v1.forward(xv)));
            let vf = v_first.unwrap();
            let v_blended = v.clone() + (vf.clone() - v.clone()) * v_gate;
            (v_blended, vf)
        };

        // === Per-batch WKV ===
        // neg_kk = -kk, kk_a = kk * a
        let neg_kk = kk.clone().neg();
        let kk_a = kk * a;

        let mut all_outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(b);
        let mut all_states: Vec<Tensor<B, 4>> = Vec::with_capacity(b);

        for bi in 0..b {
            let r_bi = r.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let w_bi = w.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let k_bi = k.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let v_bi = v_final.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let neg_kk_bi = neg_kk.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let kk_a_bi = kk_a.clone().slice([bi..bi + 1, 0..t, 0..c]).reshape([t, c]);
            let state_bi = state.clone().slice([bi..bi + 1, 0..h, 0..n, 0..n]).reshape([h, n, n]);

            let (y_bi, new_state_bi) = wkv7_seq(
                r_bi, w_bi, k_bi, v_bi, neg_kk_bi, kk_a_bi,
                state_bi, h, n,
            );

            all_outputs.push(y_bi); // [T, C]
            all_states.push(new_state_bi.unsqueeze::<4>()); // [1, H, N, N]
        }

        let y = Tensor::cat(all_outputs, 0).reshape([b, t, c]); // [B, T, C]
        let new_state: Tensor<B, 4> = Tensor::cat(all_states, 0); // [B, H, N, N]

        // === GroupNorm (per head) ===
        // group_norm(y, num_groups=H)
        let y_grouped = y.reshape([b * t, h, n]); // [B*T, H, N]
        // Per-group mean and variance
        let mean = y_grouped.clone().mean_dim(2); // [B*T, H, 1]
        let var = (y_grouped.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2); // [B*T, H, 1]
        let y_normed = (y_grouped - mean) / (var + 64e-5).sqrt(); // [B*T, H, N]
        let y_normed = y_normed.reshape([b, t, c]); // [B, T, C]
        // Apply weight and bias
        let ln_w = self.ln_x_weight.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let ln_b = self.ln_x_bias.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let y_normed = y_normed * ln_w + ln_b;

        // === Bonus term: (r * k * r_k).view(B,T,H,N).sum(-1,keepdim=True) * v.view(B,T,H,N) ===
        let r_k_val = self.r_k.val().unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, C]
        let bonus = (r.clone() * k.clone() * r_k_val).reshape([b, t, h, n]);
        let bonus_sum = bonus.sum_dim(3); // [B, T, H, 1]
        let v_heads = v_final.reshape([b, t, h, n]);
        let bonus_out = (bonus_sum * v_heads).reshape([b, t, c]); // [B, T, C]

        let y_final = y_normed + bonus_out;

        // === Gate + output projection ===
        let output = self.w_o.forward(y_final * g);

        // x_prev for next chunk = last token
        let new_x_prev = x.clone().slice([0..b, t - 1..t, 0..c]).reshape([b, c]);

        (output, new_x_prev, new_state, v_first_out)
    }

    /// Forward for single-step inference
    pub fn forward_one(
        &self,
        x: Tensor<B, 2>,          // [B, C]
        x_prev: Tensor<B, 2>,     // [B, C]
        v_first: Option<Tensor<B, 2>>,  // [B, C]
        state: Tensor<B, 4>,      // [B, H, N, N]
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 2>) {
        let [b, c] = x.dims();
        let h = self.n_head;
        let n = self.head_size;

        // Token shift
        let xx = x_prev.clone() - x.clone(); // [B, C]

        let xr = x.clone() + xx.clone() * self.x_r.val().unsqueeze::<2>();
        let xw = x.clone() + xx.clone() * self.x_w.val().unsqueeze::<2>();
        let xk = x.clone() + xx.clone() * self.x_k.val().unsqueeze::<2>();
        let xv = x.clone() + xx.clone() * self.x_v.val().unsqueeze::<2>();
        let xa = x.clone() + xx.clone() * self.x_a.val().unsqueeze::<2>();
        let xg = x.clone() + xx * self.x_g.val().unsqueeze::<2>();

        let r = self.w_r.forward(xr);
        let k_raw = self.w_k.forward(xk);
        let v = self.w_v.forward(xv.clone());

        let w_dynamic = self.w2.forward(activation::tanh(self.w1.forward(xw)));
        let w0_expanded = self.w0.val().unsqueeze::<2>();
        let w_pre = w0_expanded + w_dynamic;
        let w = w_pre.neg().exp().add_scalar(1.0).log().neg().add_scalar(-0.5);

        let a0_expanded = self.a0.val().unsqueeze::<2>();
        let a = activation::sigmoid(a0_expanded + self.a2.forward(self.a1.forward(xa)));

        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        // Key normalization
        let k_k_val = self.k_k.val().unsqueeze::<2>();
        let k_scaled = (k_raw.clone() * k_k_val).reshape([b, h, n]);
        let k_norm = k_scaled.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-12);
        let kk = (k_scaled / k_norm).reshape([b, c]);

        let k_a_val = self.k_a.val().unsqueeze::<2>();
        let device = x.device();
        let k = k_raw * (Tensor::<B, 2>::ones([b, c], &device) + (a.clone() - 1.0) * k_a_val);

        let (v_final, v_first_out) = if self.layer_id == 0 {
            (v.clone(), v.clone())
        } else {
            let v0_expanded = self.v0.val().unsqueeze::<2>();
            let v_gate = activation::sigmoid(v0_expanded + self.v2.forward(self.v1.forward(xv)));
            let vf = v_first.unwrap();
            let v_blended = v.clone() + (vf.clone() - v.clone()) * v_gate;
            (v_blended, vf)
        };

        let neg_kk = kk.clone().neg();
        let kk_a = kk * a;

        let mut all_outputs: Vec<Tensor<B, 1>> = Vec::with_capacity(b);
        let mut all_states: Vec<Tensor<B, 4>> = Vec::with_capacity(b);

        for bi in 0..b {
            let r_bi = r.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let w_bi = w.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let k_bi = k.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let v_bi = v_final.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let neg_kk_bi = neg_kk.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let kk_a_bi = kk_a.clone().slice([bi..bi + 1, 0..c]).reshape([c]);
            let state_bi = state.clone().slice([bi..bi + 1, 0..h, 0..n, 0..n]).reshape([h, n, n]);

            let (y_bi, new_state_bi) = wkv7_one(
                r_bi, w_bi, k_bi, v_bi, neg_kk_bi, kk_a_bi,
                state_bi, h, n,
            );

            all_outputs.push(y_bi);
            all_states.push(new_state_bi.unsqueeze::<4>()); // [1, H, N, N]
        }

        let y = Tensor::cat(all_outputs.into_iter().map(|o| o.reshape([1, c])).collect(), 0); // [B, C]
        let new_state: Tensor<B, 4> = Tensor::cat(all_states, 0);

        // GroupNorm
        let y_grouped = y.reshape([b, h, n]);
        let mean = y_grouped.clone().mean_dim(2);
        let var = (y_grouped.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2);
        let y_normed = (y_grouped - mean) / (var + 64e-5).sqrt();
        let y_normed = y_normed.reshape([b, c]);
        let ln_w = self.ln_x_weight.val().unsqueeze::<2>();
        let ln_b = self.ln_x_bias.val().unsqueeze::<2>();
        let y_normed = y_normed * ln_w + ln_b;

        // Bonus term
        let r_k_val = self.r_k.val().unsqueeze::<2>();
        let bonus = (r.clone() * k.clone() * r_k_val).reshape([b, h, n]);
        let bonus_sum = bonus.sum_dim(2); // [B, H, 1]
        let v_heads = v_final.reshape([b, h, n]);
        let bonus_out = (bonus_sum * v_heads).reshape([b, c]);

        let y_final = y_normed + bonus_out;
        let output = self.w_o.forward(y_final * g);

        // x_prev for next step
        (output, x.clone(), new_state, v_first_out)
    }
}

// ============================================================================
// ChannelMixing V7 — Simplified FFN
// ============================================================================

#[derive(Module, Debug)]
pub struct ChannelMixingV7<B: Backend> {
    x_k: Param<Tensor<B, 1>>,  // [C] lerp factor
    w_k: Linear<B>,             // [C, d_ffn]
    w_v: Linear<B>,             // [d_ffn, C]
    d_model: usize,
}

impl<B: Backend> ChannelMixingV7<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        let c = config.d_model;
        let d_ffn = config.d_ffn;

        let x_k = Param::from_tensor(Tensor::zeros([c], device).add_scalar(0.5));

        let w_k = LinearConfig::new(c, d_ffn)
            .with_bias(false)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        let w_v = LinearConfig::new(d_ffn, c)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Self { x_k, w_k, w_v, d_model: c }
    }

    /// Forward for sequence [B, T, C]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_prev: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [b, t, c] = x.dims();

        // Token shift
        let x_shifted = if t > 1 {
            let prev = x_prev.reshape([b, 1, c]);
            let x_trunc = x.clone().slice([0..b, 0..t - 1, 0..c]);
            Tensor::cat(vec![prev, x_trunc], 1)
        } else {
            x_prev.reshape([b, 1, c])
        };
        let xx = x_shifted - x.clone();

        // k = x + xx * x_k
        let k = x.clone() + xx * self.x_k.val().unsqueeze::<2>().unsqueeze::<3>();

        // k = relu(k @ K)^2
        let k = activation::relu(self.w_k.forward(k)).powf_scalar(2.0);

        // output = k @ V
        let output = self.w_v.forward(k);

        let new_x_prev = x.slice([0..b, t - 1..t, 0..c]).reshape([b, c]);

        (output, new_x_prev)
    }

    /// Forward for single step [B, C]
    pub fn forward_one(
        &self,
        x: Tensor<B, 2>,
        x_prev: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let xx = x_prev - x.clone();
        let k = x.clone() + xx * self.x_k.val().unsqueeze();
        let k = activation::relu(self.w_k.forward(k)).powf_scalar(2.0);
        let output = self.w_v.forward(k);
        (output, x)
    }
}

// ============================================================================
// RWKVBlockV7 — One transformer block
// ============================================================================

#[derive(Module, Debug)]
pub struct RWKVBlockV7<B: Backend> {
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    time_mix: TimeMixingV7<B>,
    channel_mix: ChannelMixingV7<B>,
}

impl<B: Backend> RWKVBlockV7<B> {
    pub fn new(config: &RWKVConfig, layer_id: usize, device: &B::Device) -> Self {
        let ln_config = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps);

        Self {
            ln1: ln_config.clone().init(device),
            ln2: ln_config.init(device),
            time_mix: TimeMixingV7::new(config, layer_id, device),
            channel_mix: ChannelMixingV7::new(config, device),
        }
    }
}

// ============================================================================
// RWKV_V7 — Full model
// ============================================================================

#[derive(Module, Debug)]
#[allow(non_camel_case_types)]
pub struct RWKV_V7<B: Backend> {
    embedding: Embedding<B>,
    ln0: LayerNorm<B>,   // Pre-embedding LayerNorm (v7 specific)
    blocks: Vec<RWKVBlockV7<B>>,
    ln_out: LayerNorm<B>,
    head: Option<Linear<B>>,  // None if weight_tying
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_head: usize,
    pub head_size: usize,
    use_weight_tying: bool,
}

impl<B: Backend> RWKV_V7<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        assert_eq!(config.rwkv_version, 7, "RWKV_V7 requires rwkv_version=7");
        assert_eq!(config.d_model % config.head_size, 0,
            "d_model ({}) must be divisible by head_size ({})",
            config.d_model, config.head_size);

        let n_head = config.n_head();

        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
            .init(device);

        let ln0 = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        let blocks: Vec<_> = (0..config.n_layers)
            .map(|i| RWKVBlockV7::new(config, i, device))
            .collect();

        let ln_out = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        let head = if config.weight_tying {
            None
        } else {
            Some(
                LinearConfig::new(config.d_model, config.vocab_size)
                    .with_bias(false)
                    .init(device),
            )
        };

        Self {
            embedding,
            ln0,
            blocks,
            ln_out,
            head,
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_head,
            head_size: config.head_size,
            use_weight_tying: config.weight_tying,
        }
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.num_params()
    }

    fn num_params(&self) -> usize {
        // Estimate — actual count from Burn modules
        let c = self.d_model;
        let vs = self.vocab_size;
        let lr = (c / 16).max(32);
        let d_ffn = c * 4; // approximate

        let embed = vs * c;
        let ln0 = 2 * c;
        let per_layer = {
            // TimeMixing: 6 lerp factors + 4 main linears + 3 low-rank pairs + k_k/k_a/r_k + ln_x
            let lerp = 6 * c;
            let main = 4 * c * c;  // r, k, v, o
            let low_rank = 3 * (c + c * lr + lr * c) + (c * lr + lr * c); // w,a,v + g
            let scaling = 3 * c + 2 * c; // k_k, k_a, r_k, ln_x_w, ln_x_b
            let ln = 4 * c; // ln1, ln2

            // ChannelMixing: x_k + K + V
            let cmix = c + c * d_ffn + d_ffn * c;

            lerp + main + low_rank + scaling + ln + cmix
        };
        let ln_out = 2 * c;
        let head = if self.use_weight_tying { 0 } else { c * vs };

        embed + ln0 + self.blocks.len() * per_layer + ln_out + head
    }

    /// Forward pass for training
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, _t] = input_ids.dims();
        let device = input_ids.device();
        let c = self.d_model;
        let h = self.n_head;
        let n = self.head_size;

        // Embedding + ln0
        let mut x = self.embedding.forward(input_ids);
        x = self.ln0.forward(x);

        // Initialize states: zeros
        let mut att_x_prev: Vec<Tensor<B, 2>> = (0..self.blocks.len())
            .map(|_| Tensor::zeros([b, c], &device))
            .collect();
        let mut att_state: Vec<Tensor<B, 4>> = (0..self.blocks.len())
            .map(|_| Tensor::zeros([b, h, n, n], &device))
            .collect();
        let mut ffn_x_prev: Vec<Tensor<B, 2>> = (0..self.blocks.len())
            .map(|_| Tensor::zeros([b, c], &device))
            .collect();

        let mut v_first: Option<Tensor<B, 3>> = None;

        for (i, block) in self.blocks.iter().enumerate() {
            let xx = block.ln1.forward(x.clone());

            let (att_out, new_att_prev, new_att_state, vf) =
                block.time_mix.forward(
                    xx,
                    att_x_prev[i].clone(),
                    v_first.clone(),
                    att_state[i].clone(),
                );
            x = x + att_out;
            att_x_prev[i] = new_att_prev;
            att_state[i] = new_att_state;

            if i == 0 {
                v_first = Some(vf);
            }

            let xx = block.ln2.forward(x.clone());
            let (ffn_out, new_ffn_prev) = block.channel_mix.forward(xx, ffn_x_prev[i].clone());
            x = x + ffn_out;
            ffn_x_prev[i] = new_ffn_prev;
        }

        x = self.ln_out.forward(x);

        // Head / weight tying
        let logits = if self.use_weight_tying {
            let [b, t, d] = x.dims();
            let emb_weight = self.embedding.weight.val();
            let x_flat = x.reshape([b * t, d]);
            x_flat.matmul(emb_weight.transpose()).reshape([b, t, self.vocab_size])
        } else {
            self.head.as_ref().unwrap().forward(x)
        };

        logits.clamp(-65.0, 65.0)
    }

    /// Single-step inference
    pub fn forward_step(
        &self,
        token_id: Tensor<B, 2, Int>,  // [B, 1]
        states: &mut Vec<(Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 2>)>, // (att_prev, att_state, ffn_prev) per layer
        v_first: &mut Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let [b, _] = token_id.dims();

        let mut x = self.embedding.forward(token_id);
        x = self.ln0.forward(x);
        let x = x.reshape([b, self.d_model]); // [B, C]
        let mut x = x;

        for (i, block) in self.blocks.iter().enumerate() {
            let xx = block.ln1.forward(x.clone().reshape([b, 1, self.d_model])).reshape([b, self.d_model]);

            // Clone state data to avoid borrow checker issues
            let att_prev = states[i].0.clone();
            let att_state = states[i].1.clone();
            let ffn_prev = states[i].2.clone();

            let (att_out, new_att_prev, new_att_state, vf) = block.time_mix.forward_one(
                xx,
                att_prev,
                v_first.clone(),
                att_state,
            );
            x = x + att_out;

            if i == 0 {
                *v_first = Some(vf);
            }

            states[i].0 = new_att_prev;
            states[i].1 = new_att_state;

            let xx = block.ln2.forward(x.clone().reshape([b, 1, self.d_model])).reshape([b, self.d_model]);
            let (ffn_out, new_ffn_prev) = block.channel_mix.forward_one(xx, ffn_prev);
            x = x + ffn_out;
            states[i].2 = new_ffn_prev;
        }

        let x = self.ln_out.forward(x.reshape([b, 1, self.d_model])).reshape([b, self.d_model]);

        let logits = if self.use_weight_tying {
            let emb_weight = self.embedding.weight.val();
            x.matmul(emb_weight.transpose())
        } else {
            self.head.as_ref().unwrap().forward(x.reshape([b, 1, self.d_model])).reshape([b, self.vocab_size])
        };

        logits.clamp(-65.0, 65.0)
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
}

// ============================================================================
// RWKVModel trait implementation
// ============================================================================

use super::traits::RWKVModel;

#[allow(non_camel_case_types)]
impl<B: Backend> RWKVModel<B> for RWKV_V7<B> {
    fn forward_train(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(input_ids)
    }

    fn num_parameters(&self) -> usize {
        self.num_params()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn representative_param_snapshot(&self) -> Vec<f32> {
        // Use w0 (decay bias) from first block's TimeMixingV7 — equivalent to v4's time_decay
        self.blocks[0]
            .time_mix.w0.val()
            .into_data().iter::<f32>().collect()
    }
}
