// src/model/config.rs

use burn::config::Config;

#[derive(Config, Debug)]
pub struct RWKVConfig {
    #[config(default = "32000")]
    pub vocab_size: usize,
    
    #[config(default = "768")]
    pub d_model: usize,
    
    #[config(default = "12")]
    pub n_layers: usize,
    
    #[config(default = "2688")]
    pub d_ffn: usize,
    
    #[config(default = "2048")]
    pub max_seq_len: usize,
    
    #[config(default = "0.1")]
    pub dropout: f64,
    
    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,
}

impl RWKVConfig {
    /// Configuração 85M para 8GB RAM
    pub fn ptbr_85m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_layers: 12,
            d_ffn: 2688,
            max_seq_len: 2048,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// Configuração menor para testes (20M)
    pub fn ptbr_mini() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 384,
            n_layers: 6,
            d_ffn: 1344,
            max_seq_len: 512,  // Reduzido!
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// NOVO: Configuração micro para 8GB RAM (10M params)
    pub fn ptbr_micro() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 256,
            n_layers: 4,
            d_ffn: 1024,
            max_seq_len: 256,  // Pequeno para caber na RAM
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn num_parameters(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let time_mixing = 5 * self.d_model * self.d_model;
        let channel_mixing = 2 * self.d_model * self.d_ffn;
        let layer_norms = 4 * self.d_model;
        let per_layer = time_mixing + channel_mixing + layer_norms;
        let all_layers = self.n_layers * per_layer;
        embedding + all_layers
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = "3e-4")]
    pub learning_rate: f64,
    
    #[config(default = "2")]
    pub batch_size: usize,
    
    #[config(default = "16")]
    pub gradient_accumulation_steps: usize,
    
    #[config(default = "500")]
    pub warmup_steps: usize,
    
    #[config(default = "50000")]
    pub max_steps: usize,
    
    #[config(default = "0.01")]
    pub weight_decay: f64,
    
    #[config(default = "1.0")]
    pub gradient_clip: f64,
    
    #[config(default = "2500")]
    pub save_every: usize,
    
    #[config(default = "500")]
    pub eval_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 2,
            gradient_accumulation_steps: 16,
            warmup_steps: 500,
            max_steps: 50_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 2500,
            eval_every: 500,
        }
    }
}