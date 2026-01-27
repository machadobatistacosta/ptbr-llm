// src/model/config.rs
//! Configurações otimizadas para T4 16GB

use burn::config::Config;

#[derive(Config, Debug, Clone)]
pub struct RWKVConfig {
    #[config(default = "65536")]
    pub vocab_size: usize,
    
    #[config(default = "768")]
    pub d_model: usize,
    
    #[config(default = "12")]
    pub n_layers: usize,
    
    #[config(default = "2688")]
    pub d_ffn: usize,
    
    #[config(default = "512")]
    pub max_seq_len: usize,
    
    #[config(default = "0.0")]
    pub dropout: f64,
    
    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,
    
    #[config(default = "true")]
    pub weight_tying: bool,
}

impl RWKVConfig {
    /// 85M - Baseline rápido
    pub fn ptbr_85m() -> Self {
        Self {
            vocab_size: 65_536,
            d_model: 768,
            n_layers: 12,
            d_ffn: 2688,
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// 400M - OTIMIZADO para T4 16GB
    pub fn ptbr_400m() -> Self {
        Self {
            vocab_size: 65_536,
            d_model: 1024,
            n_layers: 24,
            d_ffn: 3584,
            max_seq_len: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// 800M - Para T4 com seq_len reduzido
    pub fn ptbr_800m() -> Self {
        Self {
            vocab_size: 65_536,
            d_model: 1536,
            n_layers: 24,
            d_ffn: 5376,
            max_seq_len: 192,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// 1B - Limite T4
    pub fn ptbr_1b() -> Self {
        Self {
            vocab_size: 65_536,
            d_model: 2048,
            n_layers: 24,
            d_ffn: 7168,
            max_seq_len: 128,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    /// 1.5B - Apenas inferência
    pub fn ptbr_1_5b() -> Self {
        Self {
            vocab_size: 65_536,
            d_model: 2304,
            n_layers: 28,
            d_ffn: 8064,
            max_seq_len: 64,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
        }
    }

    pub fn num_parameters(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        
        let per_layer = 4 * self.d_model +
            4 * self.d_model * self.d_model + 5 * self.d_model +
            self.d_model * self.d_model + 2 * self.d_model * self.d_ffn;
        
        let head = if self.weight_tying { 0 } else { self.d_model * self.vocab_size };
        let global_ln = 4 * self.d_model;
        
        embed + self.n_layers * per_layer + head + global_ln
    }

    pub fn estimated_vram(&self, batch_size: usize, seq_len: usize) -> usize {
        let params = self.num_parameters();
        let params_mem = params * 4;
        
        let acts_per_layer = (5 * batch_size * seq_len * self.d_model 
            + batch_size * seq_len * self.d_ffn) * 4;
        let act_mem = acts_per_layer * self.n_layers;
        
        let grad_mem = params * 4;
        let opt_mem = params * 8;
        
        let total = params_mem + act_mem + grad_mem + opt_mem;
        total + total / 7
    }
    
    pub fn estimated_vram_gb(&self, batch_size: usize, seq_len: usize) -> f64 {
        self.estimated_vram(batch_size, seq_len) as f64 / 1_000_000_000.0
    }
    
    pub fn fits_in_t4(&self, batch_size: usize, seq_len: usize) -> bool {
        self.estimated_vram(batch_size, seq_len) < 14_500_000_000
    }
    
    pub fn suggest_t4_config(&self) -> (usize, usize) {
        for seq_len in [256, 192, 128, 96, 64].iter() {
            for batch in [4, 2, 1].iter() {
                if self.fits_in_t4(*batch, *seq_len) {
                    return (*batch, *seq_len);
                }
            }
        }
        (1, 64)
    }
}

#[derive(Config, Debug, Clone)]
pub struct TrainingConfig {
    #[config(default = "3e-4")]
    pub learning_rate: f64,
    
    #[config(default = "1")]
    pub batch_size: usize,
    
    #[config(default = "8")]
    pub gradient_accumulation_steps: usize,
    
    #[config(default = "200")]
    pub warmup_steps: usize,
    
    #[config(default = "10000")]
    pub max_steps: usize,
    
    #[config(default = "0.01")]
    pub weight_decay: f64,
    
    #[config(default = "1.0")]
    pub gradient_clip: f64,
    
    #[config(default = "500")]
    pub save_every: usize,
    
    #[config(default = "10")]
    pub log_every: usize,
    
    #[config(default = "0.1")]
    pub min_lr_ratio: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            warmup_steps: 200,
            max_steps: 10_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 500,
            log_every: 10,
            min_lr_ratio: 0.1,
        }
    }
}

impl TrainingConfig {
    pub fn for_t4_400m() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 1,
            gradient_accumulation_steps: 8,
            warmup_steps: 200,
            max_steps: 10_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 500,
            log_every: 3,
            min_lr_ratio: 0.1,
        }
    }
    
    pub fn for_t4_safe() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 1,
            gradient_accumulation_steps: 16,
            warmup_steps: 300,
            max_steps: 10_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 500,
            log_every: 3,
            min_lr_ratio: 0.1,
        }
    }
}