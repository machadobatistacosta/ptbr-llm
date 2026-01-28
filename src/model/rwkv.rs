// src/model/rwkv.rs
//! RWKV Model - Memory Optimized para T4 16GB
//! 
//! Mudanças principais:
//! - WKV usa EMA em vez de matriz T×T
//! - Memória O(T) em vez de O(T²)
//! - forward_step usa wkv_ema_step (compatível com treino)

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
    /// (ema_sum, ema_weight, _unused) - compatível com EMA do treino
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
                        Tensor::zeros([batch_size, d_model], device),  // ema_sum
                        Tensor::zeros([batch_size, d_model], device),  // ema_weight (scalar broadcast)
                        Tensor::zeros([batch_size, d_model], device),  // unused (mantido para compatibilidade)
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
        let logit_scale = 0.5 * (config.d_model as f32).powf(-0.5);

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
// TIME MIXING - MEMORY OPTIMIZED (EMA, sem matriz T×T)
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
    #[module(skip)]
    n_layers: usize,
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
            layer_id,
            n_layers,
        }
    }

    /// Calcula decay para esta layer (usado em treino e inferência)
    fn get_decay(&self) -> f32 {
        0.90_f32 + 0.08 * (self.layer_id as f32 / self.n_layers.max(1) as f32)
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

        // ✨ WKV MEMORY-OPTIMIZED - EMA sem matriz T×T
        let wkv = self.wkv_ema(k, v, t);
        
        self.output.forward(r * wkv)
    }

    /// ✨ WKV via EMA - O(T) memória em vez de O(T²)
    fn wkv_ema(&self, k: Tensor<B, 3>, v: Tensor<B, 3>, t: usize) -> Tensor<B, 3> {
        let [b, _, c] = v.dims();
        let decay = self.get_decay();
        
        // Para seq curtas (≤64), método direto é mais eficiente
        if t <= 64 {
            return self.wkv_direct(k, v, t, b, c, decay);
        }
        
        // Para seq maiores, chunked EMA
        self.wkv_chunked(k, v, t, b, c, decay)
    }
    
    /// WKV direto para sequências pequenas
    fn wkv_direct(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        t: usize,
        b: usize,
        c: usize,
        decay: f32,
    ) -> Tensor<B, 3> {
        let device = v.device();
        
        // Escala por posição para cumsum ponderado
        let positions: Vec<f32> = (0..t).map(|i| decay.powi(-(i as i32))).collect();
        let pos_scale = Tensor::<B, 1>::from_floats(positions.as_slice(), &device)
            .reshape([1, t, 1]);
        
        let v_scaled = v.clone() * pos_scale;
        let v_cumsum = self.cumsum_dim1(v_scaled, b, t, c);
        
        // Re-escalar
        let inv_pos: Vec<f32> = (0..t).map(|i| decay.powi(i as i32)).collect();
        let inv_pos_scale = Tensor::<B, 1>::from_floats(inv_pos.as_slice(), &device)
            .reshape([1, t, 1]);
        
        let v_ema = v_cumsum * inv_pos_scale;
        
        // Normalização
        let norm_weights: Vec<f32> = (1..=t)
            .map(|i| {
                if decay >= 0.999 { i as f32 }
                else { (1.0 - decay.powi(i as i32)) / (1.0 - decay) }
            })
            .collect();
        let norm = Tensor::<B, 1>::from_floats(norm_weights.as_slice(), &device)
            .reshape([1, t, 1])
            .clamp_min(1e-9);
        
        let v_normalized = v_ema / norm;
        
        // Gate com k
        let k_gate = activation::sigmoid(k.clamp(-30.0, 30.0) * 0.1);
        
        v_normalized * k_gate
    }
    
    /// WKV chunked para sequências maiores
    fn wkv_chunked(
        &self,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        t: usize,
        b: usize,
        c: usize,
        decay: f32,
    ) -> Tensor<B, 3> {
        let device = v.device();
        let chunk_size = 32_usize;
        let n_chunks = (t + chunk_size - 1) / chunk_size;
        
        let mut outputs: Vec<Tensor<B, 3>> = Vec::with_capacity(n_chunks);
        let mut carry_sum = Tensor::<B, 2>::zeros([b, c], &device);
        let mut carry_weight: f32 = 0.0;
        
        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(t);
            let chunk_len = end - start;
            
            let v_chunk = v.clone().slice([0..b, start..end, 0..c]);
            let k_chunk = k.clone().slice([0..b, start..end, 0..c]);
            
            // EMA local
            let positions: Vec<f32> = (0..chunk_len).map(|i| decay.powi(-(i as i32))).collect();
            let pos_scale = Tensor::<B, 1>::from_floats(positions.as_slice(), &device)
                .reshape([1, chunk_len, 1]);
            
            let v_scaled = v_chunk.clone() * pos_scale;
            let v_cumsum = self.cumsum_dim1(v_scaled, b, chunk_len, c);
            
            let inv_pos: Vec<f32> = (0..chunk_len).map(|i| decay.powi(i as i32)).collect();
            let inv_pos_scale = Tensor::<B, 1>::from_floats(inv_pos.as_slice(), &device)
                .reshape([1, chunk_len, 1]);
            
            let v_local = v_cumsum * inv_pos_scale;
            
            // Carry contribution
            let carry_contrib: Vec<f32> = (0..chunk_len)
                .map(|i| decay.powi((i + 1) as i32))
                .collect();
            let carry_scale = Tensor::<B, 1>::from_floats(carry_contrib.as_slice(), &device)
                .reshape([1, chunk_len, 1]);
            
            let carry_expanded = carry_sum.clone().reshape([b, 1, c]).repeat(&[1, chunk_len, 1]);
            let v_with_carry = v_local + carry_expanded * carry_scale;
            
            // Normalização
            let norm_weights: Vec<f32> = (0..chunk_len)
                .map(|i| {
                    let local_w = if decay >= 0.999 { (i + 1) as f32 }
                    else { (1.0 - decay.powi((i + 1) as i32)) / (1.0 - decay) };
                    let carry_w = carry_weight * decay.powi((i + 1) as i32);
                    (local_w + carry_w).max(1e-9)
                })
                .collect();
            
            let norm = Tensor::<B, 1>::from_floats(norm_weights.as_slice(), &device)
                .reshape([1, chunk_len, 1]);
            
            let v_normalized = v_with_carry / norm;
            
            // Gate
            let k_gate = activation::sigmoid(k_chunk.clamp(-30.0, 30.0) * 0.1);
            outputs.push(v_normalized * k_gate);
            
            // Update carry
            let decay_chunk = decay.powi(chunk_len as i32);
            let old_carry = carry_sum.clone() * decay_chunk;
            
            let weights_for_carry: Vec<f32> = (0..chunk_len)
                .map(|i| decay.powi((chunk_len - 1 - i) as i32))
                .collect();
            let w_carry = Tensor::<B, 1>::from_floats(weights_for_carry.as_slice(), &device)
                .reshape([1, chunk_len, 1]);
            
            let new_contrib = (v_chunk * w_carry).sum_dim(1).reshape([b, c]);
            carry_sum = old_carry + new_contrib;
            
            let old_weight = carry_weight * decay_chunk;
            let new_weight: f32 = weights_for_carry.iter().sum();
            carry_weight = old_weight + new_weight;
        }
        
        Tensor::cat(outputs, 1)
    }
    
    /// Cumsum eficiente ao longo da dimensão 1
    fn cumsum_dim1(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t == 1 {
            return x;
        }
        
        let device = x.device();
        let mut result = Vec::with_capacity(t);
        let mut running_sum = Tensor::<B, 3>::zeros([b, 1, c], &device);
        
        for i in 0..t {
            let x_i = x.clone().slice([0..b, i..i+1, 0..c]);
            running_sum = running_sum + x_i;
            result.push(running_sum.clone());
        }
        
        Tensor::cat(result, 1)
    }

    /// ✨ FORWARD STEP - Agora usa EMA compatível com treino
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

        // ✨ WKV EMA STEP - Compatível com treino
        let wkv = self.wkv_ema_step(k, v, state);
        
        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    /// ✨ WKV EMA Step - Versão incremental compatível com wkv_ema do treino
    /// 
    /// state.0 = ema_sum (soma ponderada acumulada)
    /// state.1 = ema_weight_tensor (peso acumulado, broadcast)
    /// state.2 = step_count (contador de passos)
    fn wkv_ema_step(
        &self,
        k: Tensor<B, 2>,
        v: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
    ) -> Tensor<B, 2> {
        let [b, c] = v.dims();
        let device = v.device();
        
        // Mesmo decay usado no treino
        let decay = self.get_decay();
        
        let (ema_sum, ema_weight_tensor, step_tensor) = state;
        
        // Extrai peso escalar atual (média do tensor)
        let ema_weight: f32 = ema_weight_tensor.clone()
            .mean()
            .into_scalar()
            .elem::<f32>();
        
        // Atualiza EMA: new_sum = decay * old_sum + v
        let new_sum = ema_sum.clone() * decay + v.clone();
        
        // Atualiza peso: new_weight = decay * old_weight + 1
        let new_weight = decay * ema_weight + 1.0;
        
        // Calcula output normalizado (evita divisão por zero)
        let safe_weight = new_weight.max(1e-9);
        let output = new_sum.clone() / safe_weight;
        
        // Gate com k (mesmo do treino)
        // Clamp para evitar overflow em sigmoid
        let k_clamped = k.clamp(-30.0, 30.0);
        let k_gate = activation::sigmoid(k_clamped * 0.1);
        let result = output * k_gate;
        
        // Atualiza estado
        *ema_sum = new_sum;
        *ema_weight_tensor = Tensor::full([b, c], new_weight, &device);
        
        // Incrementa contador (para debug/monitoramento)
        let step_val: f32 = step_tensor.clone()
            .mean()
            .into_scalar()
            .elem::<f32>();
        *step_tensor = Tensor::full([b, c], step_val + 1.0, &device);
        
        result
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