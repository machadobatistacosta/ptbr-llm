use burn::config::Config;

#[derive(Config, Debug)]  // Config já implementa Clone
pub struct RWKVConfig {
    #[config(default = "32000")]
    pub vocab_size: usize,
    #[config(default = "768")]
    pub d_model: usize,
    #[config(default = "12")]
    pub n_layers: usize,
    #[config(default = "2688")]
    pub d_ffn: usize,
    #[config(default = "1024")]
    pub max_seq_len: usize,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,
}

impl RWKVConfig {
    /// 85M - Baseline rápido
    pub fn ptbr_85m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_layers: 12,
            d_ffn: 2688,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    /// 400M - Validação em T4
    pub fn ptbr_400m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1024,
            n_layers: 24,
            d_ffn: 4096,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    /// 800M - Sweet Spot para T4
    pub fn ptbr_800m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1536,
            n_layers: 24,
            d_ffn: 6144,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    /// 1B - Máximo confortável em T4
    pub fn ptbr_1b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2048,
            n_layers: 24,
            d_ffn: 8192,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    /// 1.5B - Limite T4 (requer gradient checkpointing)
    pub fn ptbr_1_5b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2304,
            n_layers: 28,
            d_ffn: 9216,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn num_parameters(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let per_layer = 
            4 * self.d_model + // LayerNorms
            5 * self.d_model * self.d_model + // TimeMix
            2 * self.d_model * self.d_ffn + self.d_model * self.d_model; // ChannelMix
        let head = self.d_model * self.vocab_size;
        embed + self.n_layers * per_layer + head
    }
}

#[derive(Config, Debug)]  // Config já implementa Clone
pub struct TrainingConfig {
    #[config(default = "3e-4")]
    pub learning_rate: f64,
    #[config(default = "1")]
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
    #[config(default = "1000")]
    pub save_every: usize,
    #[config(default = "10")]
    pub log_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 1,
            gradient_accumulation_steps: 16,
            warmup_steps: 500,
            max_steps: 50_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 1000,
            log_every: 10,
        }
    }
}