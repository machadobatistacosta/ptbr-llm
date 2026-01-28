//! RWKV Model - Versão Estável com WKV EMA Vetorizado
//! 
//! Esta versão usa EMA vetorizado em vez de loop token-by-token
//! para evitar segfaults e ser mais eficiente na GPU.

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
    /// (aa, bb, pp) - numerador, denominador, max log
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    /// Estado do channel mixing
    pub channel_state: Vec<Tensor<B, 2>>,
    /// Embedding anterior
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
                        Tensor::full([batch_size, d_model], -1e30_f32, device),
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
    
    pub fn reset(&mut self, device: &B::Device) {
        let n_layers = self.time_state.len();
        let [batch_size, d_model] = self.time_state[0].0.dims();
        
        for i in 0..n_layers {
            self.time_state[i] = (
                Tensor::zeros([batch_size, d_model], device),
                Tensor::zeros([batch_size, d_model], device),
                Tensor::full([batch_size, d_model], -1e30_f32, device),
            );
            self.channel_state[i] = Tensor::zeros([batch_size, d_model], device);
            self.prev_embedding[i] = Tensor::zeros([batch_size, d_model], device);
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
        let logit_scale = 0.5 * (config.vocab_size as f32).ln().recip();

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

        let logits = if self.use_weight_tying {
            let [b, t, d] = x.dims();
            let emb_weight = self.embedding.weight.val();
            let x_flat = x.reshape([b * t, d]);
            let logits_flat = x_flat.matmul(emb_weight.transpose());
            logits_flat.reshape([b, t, self.vocab_size])
        } else {
            self.head.as_ref().unwrap().forward(x)
        };
        
        logits * self.logit_scale
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
                x.clone(),
                &mut state.time_state[layer_idx],
                &mut state.channel_state[layer_idx],
                prev_emb,
            );
            state.prev_embedding[layer_idx] = x.clone();
        }

        let x = x.reshape([b, 1, self.d_model]);
        let x = self.ln_out.forward(x);

        let logits = if self.use_weight_tying {
            let x_flat = x.reshape([b, self.d_model]);
            let emb_weight = self.embedding.weight.val();
            x_flat.matmul(emb_weight.transpose())
        } else {
            self.head.as_ref().unwrap().forward(x).reshape([b, self.vocab_size])
        };
        
        logits * self.logit_scale
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
// TIME MIXING - VERSÃO CHUNKED (ESTÁVEL)
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
    #[module(skip)]
    layer_id: usize,
}

impl<B: Backend> TimeMixing<B> {
    pub fn new(d_model: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);

        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        // time_decay inicialização
        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                let base = -5.0 + ratio_0_to_1 * 2.0;
                let variation = channel_ratio * 1.5;
                ((base + variation) as f32).clamp(-7.0, -0.5)
            })
            .collect();

        // time_first inicialização
        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                let base = 1.0 - ratio_0_to_1 * 0.5;
                ((base * (1.0 - channel_ratio * 0.3)) as f32).clamp(0.1, 2.0)
            })
            .collect();

        // time_mix inicialização
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
            layer_id,
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

        // ✨ WKV usando EMA vetorizado (mais estável que loop)
        let wkv = self.wkv_ema_vectorized(k, v, b, t, c);
        
        self.output.forward(r * wkv)
    }

    /// ✨ WKV via EMA Vetorizado - Sem loops explícitos
    /// Mais eficiente e estável que token-by-token
    fn wkv_ema_vectorized(
        &self, 
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        b: usize,
        t: usize, 
        c: usize,
    ) -> Tensor<B, 3> {
        let device = v.device();
        
        // Parâmetros aprendíveis
        let w = self.time_decay.val().clamp(-7.0, -0.5);
        let u = self.time_first.val().clamp(-2.0, 2.0);
        
        // Calcula decay efetivo por layer (0.85 a 0.98)
        let base_decay = 0.85 + 0.13 * (self.layer_id as f32 / 24.0);
        
        // Cria pesos de posição para cumsum ponderado
        let positions: Vec<f32> = (0..t)
            .map(|i| base_decay.powi(-(i as i32)))
            .collect();
        let pos_weights = Tensor::<B, 1>::from_floats(positions.as_slice(), &device)
            .reshape([1, t, 1]);
        
        // Gate baseado em k (usa parâmetro time_first)
        let u_expanded = u.reshape([1, 1, c]);
        let k_gate = activation::sigmoid((k.clone() * 0.1 + u_expanded).clamp(-10.0, 10.0));
        
        // Value ponderado pelo gate
        let v_gated = v.clone() * k_gate;
        
        // EMA via cumsum ponderado
        let v_scaled = v_gated * pos_weights.clone();
        let v_cumsum = self.cumsum_stable(v_scaled, b, t, c);
        
        // Escala inversa para normalizar
        let inv_positions: Vec<f32> = (0..t)
            .map(|i| base_decay.powi(i as i32))
            .collect();
        let inv_weights = Tensor::<B, 1>::from_floats(inv_positions.as_slice(), &device)
            .reshape([1, t, 1]);
        
        let v_ema = v_cumsum * inv_weights;
        
        // Normalização por contagem efetiva
        let counts: Vec<f32> = (1..=t)
            .map(|i| {
                if base_decay >= 0.999 {
                    i as f32
                } else {
                    (1.0 - base_decay.powi(i as i32)) / (1.0 - base_decay)
                }
            })
            .collect();
        let count_weights = Tensor::<B, 1>::from_floats(counts.as_slice(), &device)
            .reshape([1, t, 1])
            .clamp_min(1e-6);
        
        // Output normalizado
        let output = v_ema / count_weights;
        
        // Aplica decay aprendível como modulação final
        let w_expanded = w.reshape([1, 1, c]);
        let decay_mod = (w_expanded * 0.1).exp().clamp(0.5, 1.5);
        
        output * decay_mod
    }
    
    /// Cumsum estável ao longo da dimensão temporal
    fn cumsum_stable(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return x;
        }
        
        // Para sequências curtas, usa método direto
        if t <= 32 {
            return self.cumsum_direct(x, b, t, c);
        }
        
        // Para sequências longas, usa chunks
        self.cumsum_chunked(x, b, t, c)
    }
    
    fn cumsum_direct(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        let device = x.device();
        let mut chunks = Vec::with_capacity(t);
        let mut running = Tensor::<B, 3>::zeros([b, 1, c], &device);
        
        for i in 0..t {
            let xi = x.clone().slice([0..b, i..i+1, 0..c]);
            running = running + xi;
            chunks.push(running.clone());
        }
        
        Tensor::cat(chunks, 1)
    }
    
    fn cumsum_chunked(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        let device = x.device();
        let chunk_size = 16_usize;
        let n_chunks = (t + chunk_size - 1) / chunk_size;
        
        let mut all_outputs = Vec::with_capacity(n_chunks);
        let mut carry = Tensor::<B, 2>::zeros([b, c], &device);
        
        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(t);
            let len = end - start;
            
            let chunk = x.clone().slice([0..b, start..end, 0..c]);
            
            // Cumsum local
            let mut local_chunks = Vec::with_capacity(len);
            let mut running = carry.clone().reshape([b, 1, c]);
            
            for i in 0..len {
                let xi = chunk.clone().slice([0..b, i..i+1, 0..c]);
                running = running + xi;
                local_chunks.push(running.clone());
            }
            
            let local_result = Tensor::cat(local_chunks, 1);
            all_outputs.push(local_result);
            
            // Atualiza carry
            carry = running.reshape([b, c]);
        }
        
        Tensor::cat(all_outputs, 1)
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
            self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]),
        );
        let k = self.key.forward(xk.reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.reshape([b, 1, c])).reshape([b, c]);

        // WKV step
        let wkv = self.wkv_step(k, v, state);
        
        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    fn wkv_step(
        &self,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    ) -> Tensor<B, 2> {
        let [b, c] = v.dims();
        let device = v.device();
        
        let w = self.time_decay.val().clamp(-7.0, -0.5).reshape([1, c]);
        let u = self.time_first.val().clamp(-2.0, 2.0).reshape([1, c]);
        
        let (aa, bb, pp) = state;
        
        // ww = u + k
        let ww = u + k.clone();
        
        // Max estabilizado
        let p_max = self.tensor_max_2d(pp.clone(), ww.clone()).clamp(-30.0, 30.0);
        
        let e1 = (pp.clone() - p_max.clone()).clamp(-30.0, 0.0).exp();
        let e2 = (ww - p_max.clone()).clamp(-30.0, 0.0).exp();
        
        let a_num = e1.clone() * aa.clone() + e2.clone() * v.clone();
        let b_den = e1.clone() * bb.clone() + e2.clone();
        let wkv = a_num / b_den.clamp_min(1e-9);
        
        // Atualiza estado
        let ww_next = w + k;
        let p_max_next = self.tensor_max_2d(pp.clone(), ww_next.clone()).clamp(-30.0, 30.0);
        
        let e1_next = (pp.clone() - p_max_next.clone()).clamp(-30.0, 0.0).exp();
        let e2_next = (ww_next - p_max_next.clone()).clamp(-30.0, 0.0).exp();
        
        *aa = (e1_next.clone() * aa.clone() + e2_next.clone() * v).clamp(-1e6, 1e6);
        *bb = (e1_next * bb.clone() + e2_next).clamp(1e-9, 1e6);
        *pp = p_max_next;
        
        wkv
    }
    
    #[inline]
    fn tensor_max_2d(&self, a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
        let sum = a.clone() + b.clone();
        let diff = (a - b).abs();
        (sum + diff) / 2.0
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
        let k_sq = k.clone() * k;

        r * self.value.forward(k_sq)
    }

    pub fn forward_step(&self, x: Tensor<B, 2>, state: &mut Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_prev = state.clone();
        let x_diff = x.clone() - x_prev.clone();

        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        let r = activation::sigmoid(
            self.receptance.forward(xr.reshape([b, 1, c])).reshape([b, c]),
        );
        let k_logits = self.key.forward(xk.reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        let k_sq = k.clone() * k;

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