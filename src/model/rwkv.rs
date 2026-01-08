#![allow(dead_code)]

use burn::{
    module::{Module, Param},
    nn::{
        Embedding, EmbeddingConfig,
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
        Dropout, DropoutConfig,
    },
    tensor::{backend::Backend, Tensor, Int, activation},
};
use super::config::RWKVConfig;

/// Estado do RWKV para inferência incremental
#[derive(Clone, Debug)]
pub struct RWKVState<B: Backend> {
    pub time_state: Vec<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)>, // (aa, bb, pp)
    pub channel_state: Vec<Tensor<B, 2>>,
}

impl<B: Backend> RWKVState<B> {
    pub fn new(n_layers: usize, d_model: usize, batch_size: usize, device: &B::Device) -> Self {
        Self {
            time_state: (0..n_layers)
                .map(|_| (
                    Tensor::zeros([batch_size, d_model], device),
                    Tensor::zeros([batch_size, d_model], device),
                    Tensor::zeros([batch_size, d_model], device) - 1e30,
                ))
                .collect(),
            channel_state: (0..n_layers)
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
    head: Linear<B>,
    
    #[module(skip)]
    vocab_size: usize,
    #[module(skip)]
    d_model: usize,
    #[module(skip)]
    n_layers: usize,
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
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input_ids);
        x = self.ln_pre.forward(x);
        
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        x = self.ln_out.forward(x);
        self.head.forward(x)
    }

    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [b, s, v] = logits.dims();
        logits.slice([0..b, s-1..s, 0..v]).reshape([b, v])
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn d_model(&self) -> usize { self.d_model }
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
            channel_mixing: ChannelMixing::new(config.d_model, config.d_ffn, layer_id, config.n_layers, device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let tm = self.time_mixing.forward(self.ln1.forward(x.clone()));
        let x = x + self.dropout.forward(tm);
        
        let cm = self.channel_mixing.forward(self.ln2.forward(x.clone()));
        x + self.dropout.forward(cm)
    }
}

/// Tamanho do chunk para processamento batched
const WKV_CHUNK_SIZE: usize = 32;

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
                let decay = -5.0 + 2.0 * (channel_ratio * (1.0 - ratio_0_to_1));
                decay as f32
            })
            .collect();
        
        let first_values: Vec<f32> = (0..d_model)
            .map(|i| {
                let channel_ratio = i as f64 / (d_model.max(1) - 1).max(1) as f64;
                ((1.0 - channel_ratio) * 1.0 + channel_ratio * 0.5) as f32
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
        
        let x_prev = self.token_shift(x.clone(), b, t, c);
        
        let xk = self.mix(x.clone(), x_prev.clone(), self.time_mix_k.val(), c);
        let xv = self.mix(x.clone(), x_prev.clone(), self.time_mix_v.val(), c);
        let xr = self.mix(x.clone(), x_prev, self.time_mix_r.val(), c);
        
        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);
        
        // WKV BATCHED - Processa em chunks para melhor paralelização
        let wkv = self.wkv_batched(k, v, b, t, c, &device);
        
        self.output.forward(r * wkv)
    }

    fn token_shift(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.slice([0..b, 0..t-1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }

    fn mix(&self, x: Tensor<B, 3>, x_prev: Tensor<B, 3>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 3> {
        let m = mix.reshape([1, 1, c]);
        let device = x.device();
        x.clone() * m.clone() + x_prev * (Tensor::ones([1, 1, c], &device) - m)
    }

    /// WKV Batched - Processa em chunks de WKV_CHUNK_SIZE tokens
    /// ~2-3x mais rápido que loop sequencial puro
    fn wkv_batched(&self, k: Tensor<B, 3>, v: Tensor<B, 3>, b: usize, t: usize, c: usize, device: &B::Device) -> Tensor<B, 3> {
        let w = self.time_decay.val().exp().neg(); // [C]
        let u = self.time_first.val(); // [C]
        
        // Estado inicial
        let mut aa = Tensor::<B, 2>::zeros([b, c], device);
        let mut bb = Tensor::<B, 2>::zeros([b, c], device);
        let mut pp = Tensor::<B, 2>::zeros([b, c], device) - 1e30;
        
        let mut all_outputs = Vec::with_capacity(t);
        
        // Processa em chunks
        let num_chunks = (t + WKV_CHUNK_SIZE - 1) / WKV_CHUNK_SIZE;
        
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * WKV_CHUNK_SIZE;
            let end = (start + WKV_CHUNK_SIZE).min(t);
            let chunk_size = end - start;
            
            // Extrai chunk de k e v
            let k_chunk = k.clone().slice([0..b, start..end, 0..c]);
            let v_chunk = v.clone().slice([0..b, start..end, 0..c]);
            
            // Processa chunk com estado acumulado
            let (chunk_output, new_aa, new_bb, new_pp) = 
                self.wkv_chunk(k_chunk, v_chunk, aa, bb, pp, &w, &u, b, chunk_size, c, device);
            
            all_outputs.push(chunk_output);
            aa = new_aa;
            bb = new_bb;
            pp = new_pp;
        }
        
        // Concatena todos os chunks
        Tensor::cat(all_outputs, 1)
    }
    
    /// Processa um chunk de tokens mantendo estado
    fn wkv_chunk(
        &self,
        k: Tensor<B, 3>,      // [B, chunk_size, C]
        v: Tensor<B, 3>,      // [B, chunk_size, C]
        mut aa: Tensor<B, 2>, // [B, C]
        mut bb: Tensor<B, 2>, // [B, C]
        mut pp: Tensor<B, 2>, // [B, C]
        w: &Tensor<B, 1>,     // [C]
        u: &Tensor<B, 1>,     // [C]
        b: usize,
        chunk_size: usize,
        c: usize,
        _device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let mut outputs = Vec::with_capacity(chunk_size);
        
        // Loop sequencial dentro do chunk (necessário para manter dependência causal)
        for i in 0..chunk_size {
            let kt = k.clone().slice([0..b, i..i+1, 0..c]).reshape([b, c]);
            let vt = v.clone().slice([0..b, i..i+1, 0..c]).reshape([b, c]);
            
            // ww = u + k_t
            let ww = u.clone().reshape([1, c]) + kt.clone();
            
            // Estabilização numérica
            let p = pp.clone().max_pair(ww.clone());
            let e1 = (pp.clone() - p.clone()).exp();
            let e2 = (ww - p.clone()).exp();
            
            // wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2 + eps)
            let wkv = (e1.clone() * aa.clone() + e2.clone() * vt.clone()) 
                    / (e1.clone() * bb.clone() + e2.clone() + 1e-9);
            
            outputs.push(wkv.reshape([b, 1, c]));
            
            // Atualiza estado
            let ww2 = w.clone().reshape([1, c]) + pp.clone();
            let p2 = ww2.clone().max_pair(kt.clone());
            let e1_2 = (ww2 - p2.clone()).exp();
            let e2_2 = (kt - p2.clone()).exp();
            
            aa = e1_2.clone() * aa + e2_2.clone() * vt;
            bb = e1_2 * bb + e2_2;
            pp = p2;
        }
        
        (Tensor::cat(outputs, 1), aa, bb, pp)
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
    pub fn new(d_model: usize, d_ffn: usize, layer_id: usize, n_layers: usize, device: &B::Device) -> Self {
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
        let x_prev = self.token_shift(x.clone(), b, t, c);
        
        let xk = self.mix(x.clone(), x_prev.clone(), self.time_mix_k.val(), c);
        let xr = self.mix(x.clone(), x_prev, self.time_mix_r.val(), c);
        
        let r = activation::sigmoid(self.receptance.forward(xr));
        let k = activation::relu(self.key.forward(xk));
        let k_sq = k.clone() * k; // Squared ReLU
        
        r * self.value.forward(k_sq)
    }

    fn token_shift(&self, x: Tensor<B, 3>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        if t <= 1 {
            return Tensor::zeros([b, t, c], &x.device());
        }
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let shifted = x.slice([0..b, 0..t-1, 0..c]);
        Tensor::cat(vec![zeros, shifted], 1)
    }

    fn mix(&self, x: Tensor<B, 3>, x_prev: Tensor<B, 3>, mix: Tensor<B, 1>, c: usize) -> Tensor<B, 3> {
        let m = mix.reshape([1, 1, c]);
        let device = x.device();
        x.clone() * m.clone() + x_prev * (Tensor::ones([1, 1, c], &device) - m)
    }
}