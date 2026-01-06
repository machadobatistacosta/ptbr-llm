// src/model/rwkv.rs

use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig,
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor, Int, activation},
};

use super::config::RWKVConfig;

/// Time Mixing block (equivalente à atenção)
#[derive(Module, Debug)]
pub struct TimeMixing<B: Backend> {
    key: Linear<B>,
    value: Linear<B>,
    receptance: Linear<B>,
    output: Linear<B>,
    
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> TimeMixing<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        let linear_config = LinearConfig::new(d_model, d_model).with_bias(false);
        
        Self {
            key: linear_config.clone().init(device),
            value: linear_config.clone().init(device),
            receptance: linear_config.clone().init(device),
            output: linear_config.init(device),
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = x.dims();
        let device = x.device();
        
        // Shift temporal
        let x_prev = self.shift_tokens(x.clone(), batch_size, seq_len, d_model, &device);
        
        // Mix simples
        let mixed = (x.clone() + x_prev) / 2.0;
        
        // Projeções
        let k = self.key.forward(mixed.clone());
        let v = self.value.forward(mixed.clone());
        let r = activation::sigmoid(self.receptance.forward(mixed));
        
        // Atenção simplificada
        let k_softmax = activation::softmax(k, 2);
        let wkv = k_softmax * v;
        
        self.output.forward(r * wkv)
    }

    fn shift_tokens(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 3> {
        if seq_len <= 1 {
            return Tensor::zeros([batch_size, seq_len, d_model], device);
        }
        
        let zeros: Tensor<B, 3> = Tensor::zeros([batch_size, 1, d_model], device);
        let x_shifted = x.slice([0..batch_size, 0..(seq_len-1), 0..d_model]);
        
        Tensor::cat(vec![zeros, x_shifted], 1)
    }
}

/// Channel Mixing block (FFN)
#[derive(Module, Debug)]
pub struct ChannelMixing<B: Backend> {
    key: Linear<B>,
    value: Linear<B>,
    receptance: Linear<B>,
    
    #[module(skip)]
    d_model: usize,
}

impl<B: Backend> ChannelMixing<B> {
    pub fn new(d_model: usize, d_ffn: usize, device: &B::Device) -> Self {
        Self {
            key: LinearConfig::new(d_model, d_ffn).with_bias(false).init(device),
            value: LinearConfig::new(d_ffn, d_model).with_bias(false).init(device),
            receptance: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = x.dims();
        let device = x.device();
        
        // Shift temporal
        let x_prev = self.shift_tokens(x.clone(), batch_size, seq_len, d_model, &device);
        
        // Mix simples
        let mixed = (x.clone() + x_prev) / 2.0;
        
        // FFN com squared ReLU
        let k = self.key.forward(mixed.clone());
        let k_relu = activation::relu(k);
        let k_squared = k_relu.clone() * k_relu;
        
        let r = activation::sigmoid(self.receptance.forward(mixed));
        
        r * self.value.forward(k_squared)
    }

    fn shift_tokens(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize, d_model: usize, device: &B::Device) -> Tensor<B, 3> {
        if seq_len <= 1 {
            return Tensor::zeros([batch_size, seq_len, d_model], device);
        }
        
        let zeros: Tensor<B, 3> = Tensor::zeros([batch_size, 1, d_model], device);
        let x_shifted = x.slice([0..batch_size, 0..(seq_len-1), 0..d_model]);
        Tensor::cat(vec![zeros, x_shifted], 1)
    }
}

/// Bloco RWKV completo
#[derive(Module, Debug)]
pub struct RWKVBlock<B: Backend> {
    ln1: LayerNorm<B>,
    time_mixing: TimeMixing<B>,
    ln2: LayerNorm<B>,
    channel_mixing: ChannelMixing<B>,
    dropout: Dropout,
}

impl<B: Backend> RWKVBlock<B> {
    pub fn new(config: &RWKVConfig, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            time_mixing: TimeMixing::new(config.d_model, device),
            ln2: LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            channel_mixing: ChannelMixing::new(config.d_model, config.d_ffn, device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let tm_out = self.time_mixing.forward(self.ln1.forward(x.clone()));
        let x = x + self.dropout.forward(tm_out);
        
        let cm_out = self.channel_mixing.forward(self.ln2.forward(x.clone()));
        x + self.dropout.forward(cm_out)
    }
}

/// Modelo RWKV completo
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
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .init(device);
        
        let ln_pre = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        
        let blocks: Vec<RWKVBlock<B>> = (0..config.n_layers)
            .map(|_| RWKVBlock::new(config, device))
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
        let x = self.embedding.forward(input_ids);
        let mut x = self.ln_pre.forward(x);
        
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        x = self.ln_out.forward(x);
        self.head.forward(x)
    }

    pub fn forward_inference(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        let [batch, seq_len, vocab] = logits.dims();
        
        logits.slice([0..batch, (seq_len-1)..seq_len, 0..vocab])
            .squeeze(1)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
}