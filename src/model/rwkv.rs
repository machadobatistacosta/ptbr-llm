// src/model/rwkv.rs
//! RWKV Model - Versão Estável que FUNCIONA

use super::config::RWKVConfig;
use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{activation, backend::Backend, Int, Tensor},
};

// ============================================================
// ESTADO PARA INFERÊNCIA
// ============================================================

#[derive(Clone, Debug)]
pub struct RWKVState<B: Backend> {
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    pub channel_state: Vec<Tensor<B, 2>>,
    pub prev_embedding: Vec<Tensor<B, 2>>,
}

impl<B: Backend> RWKVState<B> {
    pub fn new(n_layers: usize, d_model: usize, batch_size: usize, device: &B::Device) -> Self {
        Self {
            time_state: (0..n_layers)
                .map(|_| {
                    (
                        Tensor::zeros([batch_size, d_model], device),
                        Tensor::zeros([batch_size, d_model], device),
                        Tensor::full([batch_size, d_model], -30.0, device),
                    )
                })
                .collect(),
            channel_state: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
            prev_embedding: (0..n_layers)
                .map(|_| Tensor::zeros([batch_size, d_model], device))
                .collect(),
        }
    }
}

// ============================================================
// RWKV PRINCIPAL
// ============================================================

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
    #[module(skip)]
    emb_scale: f32,
    #[module(skip)]
    logit_scale: f32,
}

impl<B: Backend> RWKV<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);

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
            Some(
                LinearConfig::new(config.d_model, config.vocab_size)
                    .with_bias(false)
                    .init(device),
            )
        };

        let emb_scale = (config.d_model as f32).powf(-0.5);
        let logit_scale = (config.d_model as f32).powf(-0.5);

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
            emb_scale,
            logit_scale,
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input_ids);
        x = x * self.emb_scale;
        x = self.ln_pre.forward(x);

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x = self.ln_out.forward(x);

        if self.use_weight_tying {
            let [b, t, d] = x.dims();
            let emb_weight = self.embedding.weight.val();
            let x_flat = x.reshape([b * t, d]);
            let logits_flat = x_flat.matmul(emb_weight.transpose()) * self.logit_scale;
            logits_flat.reshape([b, t, self.vocab_size])
        } else {
            self.head.as_ref().unwrap().forward(x) * self.logit_scale
        }
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

        let x = self.embedding.forward(token_id) * self.emb_scale;
        let x = self.ln_pre.forward(x);
        let mut x = x.reshape([b, self.d_model]);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let prev_emb = state.prev_embedding[layer_idx].clone();
            x = block.forward_step(
                x,
                &mut state.time_state[layer_idx],
                &mut state.channel_state[layer_idx],
                prev_emb,
            );
            state.prev_embedding[layer_idx] = x.clone();
        }

        let x = x.reshape([b, 1, self.d_model]);
        let x = self.ln_out.forward(x);

        if self.use_weight_tying {
            let x_flat = x.reshape([b, self.d_model]);
            let emb_weight = self.embedding.weight.val();
            x_flat.matmul(emb_weight.transpose()) * self.logit_scale
        } else {
            self.head.as_ref().unwrap().forward(x).reshape([b, self.vocab_size]) * self.logit_scale
        }
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn d_model(&self) -> usize { self.d_model }
    pub fn n_layers(&self) -> usize { self.n_layers }
}

// ============================================================
// RWKV BLOCK
// ============================================================

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
            time_mixing: TimeMixing::new(config.d_model, layer_id, config.n_layers, device),
            ln2: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            channel_mixing: ChannelMixing::new(
                config.d_model,
                config.d_ffn,
                layer_id,
                config.n_layers,
                device,
            ),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let ln1_out = self.ln1.forward(x.clone());
        let tm = self.time_mixing.forward(ln1_out);
        let x = x + self.dropout.forward(tm);

        let ln2_out = self.ln2.forward(x.clone());
        let cm = self.channel_mixing.forward(ln2_out);
        x + self.dropout.forward(cm)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        time_state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        channel_state: &mut Tensor<B, 2>,
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let x_3d = x.clone().reshape([b, 1, c]);
        let ln1_out = self.ln1.forward(x_3d).reshape([b, c]);
        let tm = self.time_mixing.forward_step(ln1_out, time_state, prev_embedding);
        let x = x + tm;

        let ln2_out = self.ln2.forward(x.clone().reshape([b, 1, c])).reshape([b, c]);
        let cm = self.channel_mixing.forward_step(ln2_out, channel_state);
        x + cm
    }
}

// ============================================================
// TIME MIXING - COM WKV PARALELO (SEM LOOP!)
// ============================================================

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
    pub fn new(d_model: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);

        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                ((-5.0 - 1.0 * ratio_0_to_1 + 0.3 * channel_ratio) as f32).clamp(-8.0, -2.0)
            })
            .collect();

        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                ((1.0 - channel_ratio) * 0.5 + channel_ratio * 0.2) as f32
            })
            .collect();

        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio) + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: linear_config.clone().init(device),
            key: linear_config.clone().init(device),
            value: linear_config.clone().init(device),
            output: linear_config.init(device),
            time_decay: Param::from_tensor(Tensor::from_floats(decay_values.as_slice(), device)),
            time_first: Param::from_tensor(Tensor::from_floats(first_values.as_slice(), device)),
            time_mix_k: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_v: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_r: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
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

        // ✨ WKV PARALELO - Sem loop, totalmente vetorizado
        let wkv = self.wkv_parallel(k, v);
        
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

        let xk = prev_embedding.clone() + mix_k * (x.clone() - prev_embedding.clone());
        let xv = prev_embedding.clone() + mix_v * (x.clone() - prev_embedding.clone());
        let xr = prev_embedding.clone() + mix_r * (x.clone() - prev_embedding);

        let r = activation::sigmoid(
            self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]),
        );
        let k = self.key.forward(xk.reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.reshape([b, 1, c])).reshape([b, c]);

        let wkv = self.wkv_step(k, v, state);
        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    fn wkv_step(
        &self,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    ) -> Tensor<B, 2> {
        let [_b, c] = k.dims();
        let w = self.time_decay.val().clamp(-10.0, 0.0).exp().neg();
        let u = self.time_first.val().clamp(-5.0, 5.0);

        let (aa, bb, pp) = state;

        let k_clamped = k.clone().clamp(-30.0, 30.0);
        let ww = u.clone().reshape([1, c]) + k_clamped.clone();
        let p = pp.clone().max_pair(ww.clone());
        let e1 = (pp.clone() - p.clone()).exp();
        let e2 = (ww - p.clone()).exp();

        let denom = e1.clone() * bb.clone() + e2.clone() + 1e-9;
        let wkv = (e1.clone() * aa.clone() + e2.clone() * v.clone()) / denom;

        let ww2 = w.reshape([1, c]) + pp.clone();
        let p2 = ww2.clone().max_pair(k_clamped.clone());
        let e1_2 = (ww2 - p2.clone()).exp();
        let e2_2 = (k_clamped - p2.clone()).exp();

        *aa = e1_2.clone() * aa.clone() + e2_2.clone() * v;
        *bb = e1_2 * bb.clone() + e2_2;
        *pp = p2;

        wkv
    }

    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }

    /// ✨ WKV PARALELO - Aproximação linear sem loop temporal
    /// Usa exponential moving average ponderado, totalmente vetorizado
    fn wkv_parallel(&self, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, channels] = k.dims();
        let device = k.device();

        // Parâmetros
        let w = self.time_decay.val().clamp(-10.0, -0.1);  // Decay (negativo)
        let u = self.time_first.val().clamp(-5.0, 5.0);    // Bonus

        // Clamp k para estabilidade
        let k = k.clamp(-30.0, 30.0);

        // Criar máscara de posição causal: [T, T]
        // mask[i,j] = decay^(i-j) se j <= i, senão 0
        let positions: Vec<f32> = (0..seq_len as i32)
            .flat_map(|i| (0..seq_len as i32).map(move |j| {
                if j <= i { (i - j) as f32 } else { -1e9 }
            }))
            .collect();
        
        let pos_matrix = Tensor::<B, 2>::from_floats(
            positions.as_slice(),
            &device,
        ).reshape([seq_len, seq_len]);

        // decay_matrix[i,j] = exp(w * (i-j)) para j <= i
        // w é negativo, então exp(w * dist) < 1 para dist > 0
        let w_broad = w.reshape([1, 1, channels]);  // [1, 1, C]
        
        // Para cada canal, calcular matriz de decay
        // Simplificação: usar decay médio
        let w_mean: f32 = w.clone().mean().into_scalar().elem();
        let decay_matrix = (pos_matrix.clone() * w_mean).exp();  // [T, T]
        
        // Adicionar bonus u para diagonal (posição atual)
        let u_mean: f32 = u.clone().mean().into_scalar().elem();
        let identity = Tensor::<B, 2>::eye(seq_len, &device);
        let bonus_matrix = identity * (u_mean.exp() - 1.0);  // Bonus na diagonal
        let weights = decay_matrix + bonus_matrix;  // [T, T]
        
        // Normalizar por linha (softmax-like, mas sem exp adicional)
        let weights_sum = weights.clone().sum_dim(1).clamp_min(1e-9);  // [T, 1]
        let weights_norm = weights / weights_sum;  // [T, T]

        // Aplicar atenção: output[b, i, c] = sum_j(weights[i,j] * v[b, j, c])
        // Reshape para batch matmul
        // v: [B, T, C] -> precisamos [B, T, T] @ [B, T, C]
        let weights_expanded = weights_norm
            .unsqueeze::<3>()  // [1, T, T]
            .repeat(&[batch_size, 1, 1]);  // [B, T, T]
        
        // output = weights @ v
        let output = weights_expanded.matmul(v);  // [B, T, C]

        // Modular pelo key (similar ao original RWKV)
        // Isso adiciona dependência do k de forma differentiable
        let k_sigmoid = activation::sigmoid(k * 0.1);  // Suaviza influência do k
        
        output * k_sigmoid
    }
}

// ============================================================
// CHANNEL MIXING
// ============================================================

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
        let ratio_1_to_almost_0 = 1.0 - (layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64);

        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio) + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            key: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            value: LinearConfig::new(d_ffn, d_model).with_bias(false).init(device),
            time_mix_k: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_r: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
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
        let k_sq = k.powf_scalar(2.0);

        r * self.value.forward(k_sq)
    }

    pub fn forward_step(&self, x: Tensor<B, 2>, state: &mut Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_prev = state.clone();

        let xk = x_prev.clone() + mix_k * (x.clone() - x_prev.clone());
        let xr = x_prev.clone() + mix_r * (x.clone() - x_prev);

        let r = activation::sigmoid(
            self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]),
        );
        let k_logits = self.key.forward(xk.reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        let k_sq = k.powf_scalar(2.0);

        let [b_dim, d_ffn_dim] = k_sq.dims();
        let output = r * self.value.forward(k_sq.reshape([b_dim, 1, d_ffn_dim])).reshape([b, c]);

        *state = x;
        output
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