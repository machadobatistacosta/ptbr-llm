//! RWKV Model - Implementação CORRIGIDA
//!
//! Fixes aplicados:
//! 1. REMOVIDO scaling de logits por sqrt(d_model)
//! 2. REMOVIDO residual_scale (GPT-J style não é usado no RWKV)
//! 3. Inicialização mantida (estava ok)
//! 4. forward_step corrigido para consistência

use super::config::RWKVConfig;
use super::wkv_optimized::{wkv_linear, wkv_step, WKVConfig};
use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer,
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{activation, backend::Backend, Int, Tensor},
};

/// Estado do modelo para geração autoregressiva
/// Bug #4 fix: Separated prev_embedding into prev_time_input (post-LN1) and prev_channel_input (post-LN2)
#[derive(Clone, Debug)]
pub struct RWKVState<B: Backend> {
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    pub channel_state: Vec<Tensor<B, 2>>,
    /// Bug #4: Post-LN1 output from previous timestep (for token_shift in TimeMixing)
    pub prev_time_input: Vec<Tensor<B, 2>>,
    /// Bug #4: Post-LN2 output from previous timestep (for token_shift in ChannelMixing)
    pub prev_channel_input: Vec<Tensor<B, 2>>,
}

impl<B: Backend> RWKVState<B> {
    pub fn new(n_layers: usize, d_model: usize, batch_size: usize, device: &B::Device) -> Self {
        Self {
            time_state: (0..n_layers)
                .map(|_| (
                    Tensor::zeros([batch_size, d_model], device),
                    Tensor::zeros([batch_size, d_model], device),
                    // Max-tracking 'o' starts at -1e38 so exp(o) ≈ 0
                    Tensor::full([batch_size, d_model], -1e38_f32, device),
                ))
                .collect(),
            channel_state: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
            prev_time_input: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
            prev_channel_input: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
        }
    }

    pub fn reset(&mut self, device: &B::Device) {
        let n_layers = self.time_state.len();
        let [batch_size, d_model] = self.time_state[0].0.dims();
        for i in 0..n_layers {
            self.time_state[i] = (
                Tensor::zeros([batch_size, d_model], device),
                Tensor::zeros([batch_size, d_model], device),
                Tensor::full([batch_size, d_model], -1e38_f32, device),
            );
            self.channel_state[i] = Tensor::zeros([batch_size, d_model], device);
            self.prev_time_input[i] = Tensor::zeros([batch_size, d_model], device);
            self.prev_channel_input[i] = Tensor::zeros([batch_size, d_model], device);
        }
    }
}

/// Modelo RWKV principal
#[derive(Module, Debug)]
pub struct RWKV<B: Backend> {
    embedding: Embedding<B>,
    ln_pre: LayerNorm<B>,
    blocks: Vec<RWKVBlock<B>>,
    ln_out: LayerNorm<B>,
    head: Option<Linear<B>>,

    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    d_model: usize,
    #[module(skip)]
    n_layers: usize,
    #[module(skip)]
    use_weight_tying: bool,
}

impl<B: Backend> RWKV<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
            .init(device);

        let ln_pre = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        let blocks: Vec<RWKVBlock<B>> = (0..config.n_layers)
            .map(|layer_id| RWKVBlock::new(config, layer_id, device))
            .collect();

        let ln_out = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        let head = if config.weight_tying {
            None
        } else {
            Some(LinearConfig::new(config.d_model, config.vocab_size)
                .with_bias(false)
                .init(device))
        };

        Self {
            embedding,
            ln_pre,
            blocks,
            ln_out,
            head,
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_layers: config.n_layers,
            use_weight_tying: config.weight_tying,
        }
    }

    /// Forward pass para treinamento
    /// Output: [batch_size, seq_len, vocab_size] logits
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input_ids);
        x = self.ln_pre.forward(x);

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x = self.ln_out.forward(x);

        // ✅ AUDIT FIX: NO scaling — RWKV-4 reference uses raw logits
        let logits = if self.use_weight_tying {
            let [b, t, d] = x.dims();
            let emb_weight = self.embedding.weight.val();
            let x_flat = x.reshape([b * t, d]);
            let logits_flat = x_flat.matmul(emb_weight.transpose());
            logits_flat.reshape([b, t, self.vocab_size])
        } else {
            self.head.as_ref().unwrap().forward(x)
        };

        // Clamp suave para segurança numérica
        logits.clamp(-30.0, 30.0)
    }

    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [b, s, v] = logits.dims();
        logits.slice([0..b, s - 1..s, 0..v]).reshape([b, v])
    }

    pub fn forward_step(
        &self,
        token_id: Tensor<B, 2, Int>,
        state: &mut RWKVState<B>,
    ) -> Tensor<B, 2> {
        let [b, _] = token_id.dims();
        let x = self.embedding.forward(token_id);
        let x = self.ln_pre.forward(x);
        let mut x = x.reshape([b, self.d_model]);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // Bug #4 fix: Pass mutable refs to prev_time_input and prev_channel_input
            // Block will save post-LN1 and post-LN2 outputs respectively
            x = block.forward_step(
                x,
                &mut state.time_state[layer_idx],
                &mut state.channel_state[layer_idx],
                &mut state.prev_time_input[layer_idx],
                &mut state.prev_channel_input[layer_idx],
            );
        }

        let x = x.reshape([b, 1, self.d_model]);
        let x = self.ln_out.forward(x);

        // ✅ AUDIT FIX: NO scaling — RWKV-4 reference uses raw logits
        let logits = if self.use_weight_tying {
            let x_flat = x.reshape([b, self.d_model]);
            let emb_weight = self.embedding.weight.val();
            x_flat.matmul(emb_weight.transpose())
        } else {
            self.head.as_ref().unwrap().forward(x).reshape([b, self.vocab_size])
        };

        logits.clamp(-30.0, 30.0)
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn d_model(&self) -> usize { self.d_model }
    pub fn n_layers(&self) -> usize { self.n_layers }
}

/// Bloco RWKV
#[derive(Module, Debug)]
pub struct RWKVBlock<B: Backend> {
    ln1: LayerNorm<B>,
    time_mixing: TimeMixing<B>,
    ln2: LayerNorm<B>,
    channel_mixing: ChannelMixing<B>,
    dropout: Dropout,
}

impl<B: Backend> RWKVBlock<B> {
    pub fn new(config: &RWKVConfig, layer_id: usize, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            time_mixing: TimeMixing::new(
                config.d_model, layer_id, config.n_layers, device,
            ),
            ln2: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            channel_mixing: ChannelMixing::new(
                config.d_model, config.d_ffn, layer_id, config.n_layers, device,
            ),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// ✅ FIX: Residual connections DIRETAS (sem scaling)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let ln1_out = self.ln1.forward(x.clone());
        let tm = self.time_mixing.forward(ln1_out);
        let x = x + self.dropout.forward(tm);

        let ln2_out = self.ln2.forward(x.clone());
        let cm = self.channel_mixing.forward(ln2_out);
        x + self.dropout.forward(cm)
    }
    /// Bug #4 COMPLETE FIX: forward_step now correctly saves post-LN outputs
    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        time_state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        channel_state: &mut Tensor<B, 2>,
        prev_time_input: &mut Tensor<B, 2>,
        prev_channel_input: &mut Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        // TimeMixing with post-LN1 token shift
        let x_3d = x.clone().reshape([b, 1, c]);
        let ln1_out = self.ln1.forward(x_3d).reshape([b, c]);
        let tm = self.time_mixing.forward_step(ln1_out.clone(), time_state, prev_time_input.clone());
        *prev_time_input = ln1_out;  // Bug #4: Save POST-LN1 for next timestep
        
        let x = x + tm;

        // ChannelMixing with post-LN2 token shift
        let ln2_out = self.ln2.forward(x.clone().reshape([b, 1, c])).reshape([b, c]);
        let cm = self.channel_mixing.forward_step(ln2_out.clone(), channel_state);
        *prev_channel_input = ln2_out;  // Bug #4: Save POST-LN2 for next timestep
        
        x + cm
    }
}

/// Time Mixing - atenção linear RWKV
#[derive(Module, Debug)]
pub struct TimeMixing<B: Backend> {
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    time_decay: Param<Tensor<B, 1>>,
    time_first: Param<Tensor<B, 1>>,
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_v: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> TimeMixing<B> {
    pub fn new(
        d_model: usize,
        layer_id: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);
        // Note: output uses same Kaiming init, scaling handled at logit level

        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        // Time decay: SEMPRE negativo, per-channel
        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                let base_decay = -5.0 + 4.5 * channel_ratio;
                let layer_factor = 0.8 + 0.4 * ratio_0_to_1;
                let decay = base_decay * layer_factor;
                (decay as f32).clamp(-5.0, -0.01)
            })
            .collect();

        // Time first: bonus pequeno
        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                let base = -0.5 + channel_ratio;
                (base as f32).clamp(-0.5, 0.5)
            })
            .collect();

        // Bug #12 fix: Separate mix ratio initialization per RWKV-4 paper
        // mix_k: ddd = (1 - i/d_model) ^ ratio_1_to_almost_0
        let mix_k_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let ratio = i as f64 / (d_model - 1).max(1) as f64;
                let ddd = 1.0 - ratio;
                ddd.powf(ratio_1_to_almost_0) as f32
            })
            .collect();

        // mix_v: same as mix_k but with +0.3 * layer_ratio offset
        let mix_v_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let ratio = i as f64 / (d_model - 1).max(1) as f64;
                let ddd = 1.0 - ratio;
                (ddd.powf(ratio_1_to_almost_0) + 0.3 * ratio_0_to_1) as f32
            })
            .collect();

        // mix_r: uses HALF the ratio (more conservative)
        let mix_r_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let ratio = i as f64 / (d_model - 1).max(1) as f64;
                let ddd = 1.0 - ratio;
                ddd.powf(0.5 * ratio_1_to_almost_0) as f32
            })
            .collect();

        Self {
            receptance: linear_config.clone().init(device),
            key: linear_config.clone().init(device),
            value: linear_config.clone().init(device),
            output: linear_config.init(device),
            time_decay: Param::from_tensor(
                Tensor::from_floats(decay_values.as_slice(), device),
            ),
            time_first: Param::from_tensor(
                Tensor::from_floats(first_values.as_slice(), device),
            ),
            time_mix_k: Param::from_tensor(
                Tensor::from_floats(mix_k_values.as_slice(), device),
            ),
            time_mix_v: Param::from_tensor(
                Tensor::from_floats(mix_v_values.as_slice(), device),
            ),
            time_mix_r: Param::from_tensor(
                Tensor::from_floats(mix_r_values.as_slice(), device),
            ),

            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, 1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, 1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, 1, c]);

        let x_prev = self.token_shift(&x, b, t, c);
        let x_diff = x.clone() - x_prev.clone();
        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xv = x_prev.clone() + mix_v * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);

        // ✅ WKV agora é per-channel correto!
        let wkv = wkv_linear(
            k, v,
            self.time_decay.val(),
            self.time_first.val(),
            &WKVConfig::for_t4(),
        );

        self.output.forward(r * wkv)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_diff = x.clone() - prev_embedding.clone();
        let xk = prev_embedding.clone() + mix_k * x_diff.clone();
        let xv = prev_embedding.clone() + mix_v * x_diff.clone();
        let xr = prev_embedding + mix_r * x_diff;

        let r = activation::sigmoid(
            self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]),
        );
        let k = self.key.forward(xk.clone().reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.clone().reshape([b, 1, c])).reshape([b, c]);

        let wkv = wkv_step(
            k, v,
            self.time_decay.val(),
            self.time_first.val(),
            state,
        );

        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }
}

/// Channel Mixing - FFN com gating
#[derive(Module, Debug)]
pub struct ChannelMixing<B: Backend> {
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> ChannelMixing<B> {
    pub fn new(
        d_model: usize,
        d_ffn: usize,
        layer_id: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let ratio_1_to_almost_0 =
            1.0 - (layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64);

        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio)
                    + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: LinearConfig::new(d_model, d_model)
                .with_bias(false).init(device),
            key: LinearConfig::new(d_model, d_ffn)
                .with_bias(false).init(device),
            // Note: value uses Kaiming init, scaling handled at logit level
            value: LinearConfig::new(d_ffn, d_model)
                .with_bias(false)
                .init(device),
            time_mix_k: Param::from_tensor(
                Tensor::from_floats(mix_values.as_slice(), device),
            ),
            time_mix_r: Param::from_tensor(
                Tensor::from_floats(mix_values.as_slice(), device),
            ),
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, 1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, 1, c]);

        let x_prev = self.token_shift(&x, b, t, c);
        let x_diff = x.clone() - x_prev.clone();
        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = activation::relu(self.key.forward(xk));
        let k_sq = k.clone() * k;

        r * self.value.forward(k_sq)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        state: &mut Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_prev = state.clone();
        let x_diff = x.clone() - x_prev.clone();
        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        let r = activation::sigmoid(
            self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]),
        );
        let k_logits = self.key.forward(xk.clone().reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        let k_sq = k.clone() * k;

        let [b_dim, d_ffn_dim] = k_sq.dims();
        let output = r
            * self.value
                .forward(k_sq.reshape([b_dim, 1, d_ffn_dim]))
                .reshape([b, c]);

        *state = x;
        output
    }

    fn token_shift(
        &self,
        x: &Tensor<B, 3>,
        b: usize,
        t: usize,
        c: usize,
    ) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }
}