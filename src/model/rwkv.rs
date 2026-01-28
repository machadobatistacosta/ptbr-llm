//! RWKV Model - Implementação Completa Corrigida
//! 
//! Correções aplicadas:
//! 1. WKV usa time_decay corretamente (já é negativo)
//! 2. Inicialização de parâmetros conforme paper original
//! 3. Chunked backpropagation para T4 16GB

use super::config::RWKVConfig;
use super::wkv_optimized::{wkv_linear, wkv_step, WKVConfig};
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
    /// (aa, bb, pp) por layer - numerador, denominador, max log
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>,
    /// Estado do channel mixing por layer
    pub channel_state: Vec<Tensor<B, 2>>,
    /// Embedding anterior por layer
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
                        Tensor::full([batch_size, d_model], -1e38_f32, device),
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
                Tensor::full([batch_size, d_model], -1e38_f32, device),
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

        // Scaling factors para estabilidade
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

    /// Forward pass para training
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Embedding + scaling
        let mut x = self.embedding.forward(input_ids);
        x = x * self.emb_scale;
        x = self.ln_pre.forward(x);

        // Passa por todos os blocks
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        // LayerNorm final
        x = self.ln_out.forward(x);

        // Projeção para vocabulário
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

    /// Forward que retorna apenas o último token (para inferência)
    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [b, s, v] = logits.dims();
        logits.slice([0..b, s - 1..s, 0..v]).reshape([b, v])
    }

    /// Forward step-by-step para geração
    pub fn forward_step(
        &self,
        token_id: Tensor<B, 2, Int>,
        state: &mut RWKVState<B>,
    ) -> Tensor<B, 2> {
        let [b, _] = token_id.dims();

        // Embedding
        let x = self.embedding.forward(token_id) * self.emb_scale;
        let x = self.ln_pre.forward(x);
        let mut x = x.reshape([b, self.d_model]);

        // Passa por cada block com state
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

        // Output
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

    // Getters
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
        // Time mixing com residual
        let ln1_out = self.ln1.forward(x.clone());
        let tm = self.time_mixing.forward(ln1_out);
        let x = x + self.dropout.forward(tm);

        // Channel mixing com residual
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

        // Time mixing
        let x_3d = x.clone().reshape([b, 1, c]);
        let ln1_out = self.ln1.forward(x_3d).reshape([b, c]);
        let tm = self.time_mixing.forward_step(ln1_out, time_state, prev_embedding);
        let x = x + tm;

        // Channel mixing
        let ln2_out = self.ln2.forward(x.clone().reshape([b, 1, c])).reshape([b, c]);
        let cm = self.channel_mixing.forward_step(ln2_out, channel_state);
        x + cm
    }
}

// ============================================================
// TIME MIXING - VERSÃO CORRIGIDA
// ============================================================

#[derive(Module, Debug)]
pub struct TimeMixing<B: Backend> {
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    
    /// Time decay (NEGATIVO! ex: -5.0 para decay rápido)
    time_decay: Param<Tensor<B, 1>>,
    /// Time first (bonus para token atual)
    time_first: Param<Tensor<B, 1>>,
    
    /// Mixing weights para k, v, r
    time_mix_k: Param<Tensor<B, 1>>,
    time_mix_v: Param<Tensor<B, 1>>,
    time_mix_r: Param<Tensor<B, 1>>,
    
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> TimeMixing<B> {
    pub fn new(d_model: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);

        // Ratios para inicialização dependente do layer
        let ratio_0_to_1 = layer_id as f64 / (n_layers.max(1) - 1).max(1) as f64;
        let ratio_1_to_almost_0 = 1.0 - ratio_0_to_1;

        // ═══════════════════════════════════════════════════════════
        // INICIALIZAÇÃO CRÍTICA: time_decay (w)
        // Valores NEGATIVOS! Mais negativo = decay mais rápido
        // Fórmula do RWKV original para garantir bom aprendizado
        // ═══════════════════════════════════════════════════════════
        let decay_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                // Fórmula original do RWKV
                let decay = -5.0 + 8.0 * channel_ratio.powf(0.7 + 1.3 * ratio_0_to_1);
                (decay as f32).clamp(-8.0, -0.1)
            })
            .collect();

        // ═══════════════════════════════════════════════════════════
        // INICIALIZAÇÃO: time_first (u)
        // Controla bonus para o token atual vs histórico
        // ═══════════════════════════════════════════════════════════
        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
                // Zigzag pattern + base value
                let zigzag = match i % 3 {
                    0 => 0.1,
                    1 => -0.1,
                    _ => 0.0,
                };
                let base = (ratio_1_to_almost_0 * 0.5 + channel_ratio * 0.3) as f32;
                (base + zigzag).clamp(-1.0, 1.0)
            })
            .collect();

        // ═══════════════════════════════════════════════════════════
        // INICIALIZAÇÃO: time_mix (para k, v, r)
        // Controla interpolação entre token atual e anterior
        // ═══════════════════════════════════════════════════════════
        let mix_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model - 1).max(1) as f64;
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

        // Mixing weights expandidos para broadcast
        let mix_k = self.time_mix_k.val().reshape([1, 1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, 1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, 1, c]);

        // Token shift: x_prev[t] = x[t-1]
        let x_prev = self.token_shift(&x, b, t, c);
        
        // Interpolação: xk = x_prev + mix_k * (x - x_prev)
        let x_diff = x.clone() - x_prev.clone();
        let xk = x_prev.clone() + mix_k * x_diff.clone();
        let xv = x_prev.clone() + mix_v * x_diff.clone();
        let xr = x_prev + mix_r * x_diff;

        // Projeções lineares
        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);

        // ✅ WKV usando implementação corrigida
        let wkv = wkv_linear(
            k, v,
            self.time_decay.val(),
            self.time_first.val(),
            &WKVConfig::for_t4(),
        );
        
        // Output com gate
        self.output.forward(r * wkv)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        state: &mut (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
        prev_embedding: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [b, c] = x.dims();

        // Mixing
        let mix_k = self.time_mix_k.val().reshape([1, c]);
        let mix_v = self.time_mix_v.val().reshape([1, c]);
        let mix_r = self.time_mix_r.val().reshape([1, c]);

        let x_diff = x.clone() - prev_embedding.clone();
        let xk = prev_embedding.clone() + mix_k * x_diff.clone();
        let xv = prev_embedding.clone() + mix_v * x_diff.clone();
        let xr = prev_embedding + mix_r * x_diff;

        // Projeções
        let r = activation::sigmoid(
            self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]),
        );
        let k = self.key.forward(xk.clone().reshape([b, 1, c])).reshape([b, c]);
        let v = self.value.forward(xv.clone().reshape([b, 1, c])).reshape([b, c]);

        // ✅ WKV step corrigido
        let wkv = wkv_step(
            k, v,
            self.time_decay.val(),
            self.time_first.val(),
            state,
        );
        
        self.output.forward((r * wkv).reshape([b, 1, c])).reshape([b, c])
    }

    /// Shift tokens para a direita, preenchendo com zeros
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
// CHANNEL MIXING (FFN)
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

        // Gate
        let r = activation::sigmoid(self.receptance.forward(xr));
        
        // FFN com squared ReLU (característica do RWKV)
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
            self.receptance.forward(xr.clone().reshape([b, 1, c])).reshape([b, c]),
        );
        let k_logits = self.key.forward(xk.clone().reshape([b, 1, c]));
        let k = activation::relu(k_logits).flatten(1, 2);
        let k_sq = k.clone() * k;

        let [b_dim, d_ffn_dim] = k_sq.dims();
        let output = r * self.value.forward(k_sq.reshape([b_dim, 1, d_ffn_dim])).reshape([b, c]);

        // Atualiza state
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

// ============================================================
// TESTES
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_rwkv_forward() {
        let device = Default::default();
        let config = RWKVConfig {
            vocab_size: 1000,
            d_model: 64,
            n_layers: 2,
            d_ffn: 128,
            max_seq_len: 32,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        };
        
        let model: RWKV<TestBackend> = RWKV::new(&config, &device);
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        let output = model.forward(input);
        
        assert_eq!(output.dims(), [2, 8, 1000]);
    }
    
    #[test]
    fn test_rwkv_step() {
        let device = Default::default();
        let config = RWKVConfig {
            vocab_size: 100,
            d_model: 32,
            n_layers: 2,
            d_ffn: 64,
            max_seq_len: 16,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        };
        
        let model: RWKV<TestBackend> = RWKV::new(&config, &device);
        let mut state = RWKVState::new(2, 32, 1, &device);
        
        let token = Tensor::<TestBackend, 2, Int>::zeros([1, 1], &device);
        let output = model.forward_step(token, &mut state);
        
        assert_eq!(output.dims(), [1, 100]);
    }
    
    #[test]
    fn test_time_decay_is_negative() {
        let device = Default::default();
        let tm: TimeMixing<TestBackend> = TimeMixing::new(64, 0, 12, &device);
        
        let decay = tm.time_decay.val();
        let data = decay.to_data();
        let values = data.as_slice::<f32>().unwrap();
        
        // Todos os valores de decay devem ser negativos
        for &v in values.iter() {
            assert!(v < 0.0, "time_decay deve ser negativo, encontrado: {}", v);
        }
    }
}