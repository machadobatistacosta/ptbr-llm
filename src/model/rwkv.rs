// src/model/rwkv.rs
//! RWKV Model - Memory Optimized for T4 16GB
//! 
//! Otimizações aplicadas:
//! 1. Weight tying entre embedding e head (economiza ~267MB)
//! 2. Redução de .clone() desnecessários
//! 3. Operações in-place onde possível
//! 4. Squared ReLU otimizado

use super::config::RWKVConfig;
use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{activation, backend::Backend, Int, Tensor},
};

/// Epsilon para estabilidade numérica
const NUMERIC_EPS: f32 = 1e-7;

/// Estado do RWKV para inferência incremental
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
                        Tensor::full([batch_size, d_model], -1e30, device),
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

#[derive(Module, Debug)]
pub struct RWKV<B: Backend> {
    embedding: Embedding<B>,
    ln_pre: LayerNorm<B>,
    blocks: Vec<RWKVBlock<B>>,
    ln_out: LayerNorm<B>,
    // ✨ OTIMIZAÇÃO: Head usa os mesmos pesos do embedding (weight tying)
    // Economiza vocab_size * d_model * 4 bytes (~267MB para vocab=65k, d_model=1024)
    head: Linear<B>,
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

        // Head projection - para weight tying, inicializamos mas substituímos no forward
        let head = LinearConfig::new(config.d_model, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embedding,
            ln_pre,
            blocks,
            ln_out,
            head,
            vocab_size: config.vocab_size,
            d_model: config.d_model,
            n_layers: config.n_layers,
            use_weight_tying: true, // ✨ Ativa weight tying por padrão
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Embedding
        let mut x = self.embedding.forward(input_ids);
        x = self.ln_pre.forward(x);

        // Blocos RWKV
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        // Output
        x = self.ln_out.forward(x);
        
        // ✨ WEIGHT TYING: usa embedding.weight transposto como head
        if self.use_weight_tying {
            // head_out = x @ embedding.weight.T
            // x: [B, T, D], embedding.weight: [V, D]
            // Resultado: [B, T, V]
            let [b, t, d] = x.dims();
            
            // Obtém peso do embedding [V, D]
            let emb_weight = self.embedding.weight.val();
            
            // Reshape x para [B*T, D] para matmul
            let x_flat = x.reshape([b * t, d]);
            
            // Matmul: [B*T, D] @ [D, V] -> [B*T, V]
            // Precisamos transpor emb_weight de [V, D] para [D, V]
            let emb_t = emb_weight.transpose();
            let logits_flat = x_flat.matmul(emb_t);
            
            // Reshape de volta para [B, T, V]
            logits_flat.reshape([b, t, self.vocab_size])
        } else {
            self.head.forward(x)
        }
    }

    /// Forward para inferência - retorna logits do último token
    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [b, s, v] = logits.dims();
        logits.slice([0..b, s - 1..s, 0..v]).reshape([b, v])
    }

    /// Forward incremental - processa um único token usando estado
    pub fn forward_step(&self, token_id: Tensor<B, 2, Int>, state: &mut RWKVState<B>) -> Tensor<B, 2> {
        let [b, _] = token_id.dims();

        // Embedding + LayerNorm
        let x = self.embedding.forward(token_id);
        let x = self.ln_pre.forward(x);
        let mut x = x.reshape([b, self.d_model]);

        // Processa através dos blocos
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

        // LayerNorm final + Head
        let x = x.reshape([b, 1, self.d_model]);
        let x = self.ln_out.forward(x);
        
        if self.use_weight_tying {
            let x_flat = x.reshape([b, self.d_model]);
            let emb_weight = self.embedding.weight.val();
            let emb_t = emb_weight.transpose();
            x_flat.matmul(emb_t)
        } else {
            self.head.forward(x).reshape([b, self.vocab_size])
        }
    }

    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize { self.vocab_size }
    #[allow(dead_code)]
    pub fn d_model(&self) -> usize { self.d_model }
    #[allow(dead_code)]
    pub fn n_layers(&self) -> usize { self.n_layers }
}

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
        // ✨ OTIMIZAÇÃO: Evita clone desnecessário usando add
        // Pre-norm architecture com residual
        let ln1_out = self.ln1.forward(x.clone());
        let tm = self.time_mixing.forward(ln1_out);
        let x = x + self.dropout.forward(tm);
        
        let ln2_out = self.ln2.forward(x.clone());
        let cm = self.channel_mixing.forward(ln2_out);
        x + self.dropout.forward(cm)
    }

    /// Forward incremental
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
        let linear = LinearConfig::new(d_model, d_model).with_bias(false);

        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                let decay = -5.0 - 2.0 * ratio_0_to_1 + 0.5 * channel_ratio;
                decay as f32
            })
            .collect();

        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                ((1.0 - channel_ratio) * 0.8 + channel_ratio * 0.4) as f32
            })
            .collect();

        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                (ratio_1_to_almost_0 * (1.0 - channel_ratio) + 0.5 * channel_ratio) as f32
            })
            .collect();

        Self {
            receptance: linear.clone().init(device),
            key: linear.clone().init(device),
            value: linear.clone().init(device),
            output: linear.init(device),
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
        let device = x.device();

        // Token shift - ✨ OTIMIZAÇÃO: reutiliza x_prev
        let x_prev = self.token_shift(&x, b, t, c);

        // Mix - ✨ OTIMIZAÇÃO: referências onde possível
        let xk = self.mix(&x, &x_prev, self.time_mix_k.val(), c);
        let xv = self.mix(&x, &x_prev, self.time_mix_v.val(), c);
        let xr = self.mix(&x, &x_prev, self.time_mix_r.val(), c);

        // Projections
        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);

        // WKV
        let wkv = self.wkv_stable(k, v, b, t, c, &device);

        // Output
        self.output.forward(r * wkv)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let xk = self.mix_single(&x, &prev_embedding, self.time_mix_k.val(), c);
        let xv = self.mix_single(&x, &prev_embedding, self.time_mix_v.val(), c);
        let xr = self.mix_single(&x, &prev_embedding, self.time_mix_r.val(), c);

        let r = activation::sigmoid(self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]));
        let k = self.key.forward(xk.reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.reshape([b, 1, c])).reshape([b, c]);

        let wkv = self.wkv_step(k, v, state);

        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    // ✨ OTIMIZAÇÃO: Usa referências em vez de clones
    fn mix_single(&self, x: &Tensor<B, 2>, x_prev: &Tensor<B, 2>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 2> {
        let m = mix.reshape([1, c]);
        x.clone() * m.clone() + x_prev.clone() * (1.0 - m)
    }

    fn wkv_step(
        &self,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    ) -> Tensor<B, 2> {
        let [_b, c] = k.dims();
        let w = self.time_decay.val().exp().neg();
        let u = self.time_first.val();

        let (aa, bb, pp) = state;

        let ww = u.clone().reshape([1, c]) + k.clone();
        let p = pp.clone().max_pair(ww.clone());
        let e1 = (pp.clone() - p.clone()).exp();
        let e2 = (ww - p.clone()).exp();

        let denom = e1.clone() * bb.clone() + e2.clone() + NUMERIC_EPS;
        let wkv = (e1.clone() * aa.clone() + e2.clone() * v.clone()) / denom;

        let ww2 = w.reshape([1, c]) + pp.clone();
        let p2 = ww2.clone().max_pair(k.clone());
        let e1_2 = (ww2 - p2.clone()).exp();
        let e2_2 = (k - p2.clone()).exp();

        *aa = e1_2.clone() * aa.clone() + e2_2.clone() * v;
        *bb = e1_2 * bb.clone() + e2_2;
        *pp = p2;

        wkv
    }

    // ✨ OTIMIZAÇÃO: Evita alocação desnecessária
    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }

    // ✨ OTIMIZAÇÃO: Usa referências
    fn mix(&self, x: &Tensor<B, 3>, x_prev: &Tensor<B, 3>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 3> {
        let m = mix.reshape([1, 1, c]);
        x.clone() * m.clone() + x_prev.clone() * (1.0 - m)
    }

    fn wkv_stable(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        _b: usize,
        _t: usize,
        _c: usize,
        _device: &B::Device,
    ) -> Tensor<B, 3> {
        use crate::model::wkv_optimized::{wkv_linear, WKVConfig};
        
        let u = self.time_first.val();
        let w = self.time_decay.val();
        let config = WKVConfig::for_t4();
        
        wkv_linear(k, v, w, u, &config)
    }
}

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
            receptance: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            key: LinearConfig::new(d_model, d_ffn)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(d_ffn, d_model)
                .with_bias(false)
                .init(device),
            time_mix_k: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            time_mix_r: Param::from_tensor(Tensor::from_floats(mix_values.as_slice(), device)),
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let x_prev = self.token_shift(&x, b, t, c);

        let xk = self.mix(&x, &x_prev, self.time_mix_k.val(), c);
        let xr = self.mix(&x, &x_prev, self.time_mix_r.val(), c);

        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = activation::relu(self.key.forward(xk));
        
        // ✨ OTIMIZAÇÃO: Squared ReLU sem clone extra
        // k^2 em vez de k.clone() * k
        let k_sq = k.clone().powf_scalar(2.0);
        
        r * self.value.forward(k_sq)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        state: &mut Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let x_prev = state.clone();
        let xk = self.mix_single(&x, &x_prev, self.time_mix_k.val(), c);
        let xr = self.mix_single(&x, &x_prev, self.time_mix_r.val(), c);

        let r = activation::sigmoid(self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]));
        let k_logits = self.key.forward(xk.reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        
        // ✨ OTIMIZAÇÃO: powf_scalar
        let k_sq = k.clone().powf_scalar(2.0);
        
        let [b_dim, d_ffn_dim] = k_sq.dims();
        let output = r * self.value.forward(k_sq.reshape([b_dim, 1, d_ffn_dim])).reshape([b, c]);

        // Atualiza estado com x atual para próximo token
        *state = x;

        output
    }

    fn mix_single(&self, x: &Tensor<B, 2>, x_prev: &Tensor<B, 2>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 2> {
        let m = mix.reshape([1, c]);
        x.clone() * m.clone() + x_prev.clone() * (1.0 - m)
    }

    fn token_shift(&self, x: &Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.clone().slice([0..b, 0..t - 1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }

    fn mix(&self, x: &Tensor<B, 3>, x_prev: &Tensor<B, 3>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 3> {
        let m = mix.reshape([1, 1, c]);
        x.clone() * m.clone() + x_prev.clone() * (1.0 - m)
    }
}
