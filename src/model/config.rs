// src/model/config.rs
//! Configurações otimizadas para T4 16GB

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
    
    #[config(default = "1024")]
    pub max_seq_len: usize,
    
    #[config(default = "0.0")]
    pub dropout: f64,
    
    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,
    
    // ✨ NOVO: Weight tying economiza ~130MB para vocab=32k
    #[config(default = "true")]
    pub weight_tying: bool,
    
    // ✨ NOVO: Modo WKV para T4
    #[config(default = "true")]
    pub wkv_aggressive_detach: bool,
}

impl RWKVConfig {
    /// 85M - Baseline rápido
    pub fn ptbr_85m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_layers: 12,
            d_ffn: 2688,  // 768 × 3.5
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            wkv_aggressive_detach: true,
        }
    }

    /// 400M - OTIMIZADO para T4 16GB
    pub fn ptbr_400m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1024,
            n_layers: 24,
            // ✨ OTIMIZAÇÃO: 3584 em vez de 4096 (economiza ~100MB)
            d_ffn: 3584,  // 1024 × 3.5 (era 4096 = ×4)
            max_seq_len: 512,  // Reduzido para T4
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            wkv_aggressive_detach: true,
        }
    }

    /// 400M - Versão original (para comparação)
    pub fn ptbr_400m_original() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1024,
            n_layers: 24,
            d_ffn: 4096,
            max_seq_len: 1024,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: false,
            wkv_aggressive_detach: false,
        }
    }

    /// 800M - Para T4 com seq_len reduzido
    pub fn ptbr_800m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1536,
            n_layers: 24,
            d_ffn: 5376,  // 1536 × 3.5
            max_seq_len: 256,  // Muito reduzido para caber
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            wkv_aggressive_detach: true,
        }
    }

    /// 1B - Limite T4 (batch=1, seq=128)
    pub fn ptbr_1b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2048,
            n_layers: 24,
            d_ffn: 7168,  // 2048 × 3.5
            max_seq_len: 128,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            wkv_aggressive_detach: true,
        }
    }

    /// 1.5B - Apenas para inferência em T4
    pub fn ptbr_1_5b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2304,
            n_layers: 28,
            d_ffn: 8064,  // 2304 × 3.5
            max_seq_len: 64,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            wkv_aggressive_detach: true,
        }
    }

    pub fn num_parameters(&self) -> usize {
        // Embedding
        let embed = self.vocab_size * self.d_model;
        
        // Per layer:
        // - LayerNorm × 2: 2 × 2 × d_model = 4 × d_model
        // - TimeMixing: 4 × d_model² (r,k,v,o) + 5 × d_model (time params)
        // - ChannelMixing: d_model² + 2 × d_model × d_ffn
        let per_layer = 4 * self.d_model +  // LayerNorms
            4 * self.d_model * self.d_model + 5 * self.d_model +  // TimeMixing
            self.d_model * self.d_model + 2 * self.d_model * self.d_ffn;  // ChannelMixing
        
        // Head (se weight_tying, não conta)
        let head = if self.weight_tying { 0 } else { self.d_model * self.vocab_size };
        
        // LayerNorms globais
        let global_ln = 4 * self.d_model;  // ln_pre + ln_out
        
        embed + self.n_layers * per_layer + head + global_ln
    }

    /// ✨ Estimativa REALISTA de VRAM (inclui tudo)
    pub fn estimated_vram(&self, batch_size: usize, seq_len: usize) -> usize {
        let params = self.num_parameters();
        
        // Parâmetros em FP32
        let params_mem = params * 4;
        
        // Ativações por layer (conservador):
        // - Input/Output: 2 × B × T × D
        // - K, V, R: 3 × B × T × D
        // - WKV intermediários: 2 × B × T × D (se não agressivo)
        // - FFN intermediários: B × T × d_ffn
        let acts_per_layer = if self.wkv_aggressive_detach {
            // Com detach agressivo: menos intermediários
            (5 * batch_size * seq_len * self.d_model + batch_size * seq_len * self.d_ffn) * 4
        } else {
            // Sem detach: guarda tudo para backward
            (8 * batch_size * seq_len * self.d_model + batch_size * seq_len * self.d_ffn) * 4
        };
        let act_mem = acts_per_layer * self.n_layers;
        
        // Gradientes (mesmo tamanho dos parâmetros)
        let grad_mem = params * 4;
        
        // Optimizer states (AdamW: 2 estados por parâmetro)
        let opt_mem = params * 8;
        
        // Overhead CUDA (~10%)
        let overhead = (params_mem + act_mem + grad_mem + opt_mem) / 10;
        
        params_mem + act_mem + grad_mem + opt_mem + overhead
    }
    
    /// Verifica se cabe na T4 (15GB úteis)
    pub fn fits_in_t4(&self, batch_size: usize, seq_len: usize) -> bool {
        self.estimated_vram(batch_size, seq_len) < 15_000_000_000
    }
    
    /// Sugere configuração máxima para T4
    pub fn suggest_t4_config(&self) -> (usize, usize) {
        // Tenta encontrar batch_size e seq_len que cabem
        for seq_len in [512, 384, 256, 192, 128, 96, 64].iter() {
            for batch in [4, 2, 1].iter() {
                if self.fits_in_t4(*batch, *seq_len) {
                    return (*batch, *seq_len);
                }
            }
        }
        (1, 64)  // Fallback mínimo
    }
}

#[derive(Config, Debug)]
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
    
    #[config(default = "0.1")]
    pub min_lr_ratio: f64,
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
            min_lr_ratio: 0.1,
        }
    }
}

impl TrainingConfig {
    /// Config otimizada para T4 com modelo 400M
    pub fn for_t4_400m() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_size: 1,
            gradient_accumulation_steps: 4,
            warmup_steps: 100,
            max_steps: 20_000,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            save_every: 500,
            log_every: 10,
            min_lr_ratio: 0.1,
        }
    }
}