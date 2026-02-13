//! Configurações otimizadas — suporta RWKV v4 e v7
//!
//! v4: WKV com decay fixo (4 projeções)
//! v7: WKV com decay dinâmico + LoRA-like state (7 projeções)

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

    #[config(default = "512")]
    pub max_seq_len: usize,

    #[config(default = "0.0")]
    pub dropout: f64,

    #[config(default = "1e-5")]
    pub layer_norm_eps: f64,

    #[config(default = "true")]
    pub weight_tying: bool,

    // === RWKV-7 specific fields ===

    /// RWKV version: 4 or 7
    #[config(default = "4")]
    pub rwkv_version: usize,

    /// Head size (v7 only). n_head = d_model / head_size
    #[config(default = "64")]
    pub head_size: usize,

    /// Low-rank dimension for w, a, g, v projections (v7)
    /// Default = d_model / 16. Set to 0 to auto-compute.
    #[config(default = "0")]
    pub low_rank_dim: usize,
}

impl RWKVConfig {
    /// Effective low_rank dimension (auto-compute if 0)
    pub fn effective_low_rank(&self) -> usize {
        if self.low_rank_dim > 0 {
            self.low_rank_dim
        } else {
            (self.d_model / 16).max(32)
        }
    }

    /// Number of attention heads (v7) = d_model / head_size
    pub fn n_head(&self) -> usize {
        self.d_model / self.head_size
    }

    // ================================
    // RWKV-4 Presets
    // ================================

    /// ~140M params RWKV-4: d_model=768, n_layers=12
    pub fn ptbr_140m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_layers: 12,
            d_ffn: 3072,
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 4,
            head_size: 64,
            low_rank_dim: 0,
        }
    }

    /// 400M RWKV-4
    pub fn ptbr_400m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1024,
            n_layers: 24,
            d_ffn: 4096,
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 4,
            head_size: 64,
            low_rank_dim: 0,
        }
    }

    /// 800M RWKV-4
    pub fn ptbr_800m() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1536,
            n_layers: 24,
            d_ffn: 6144,
            max_seq_len: 256,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 4,
            head_size: 64,
            low_rank_dim: 0,
        }
    }

    /// 1B RWKV-4
    pub fn ptbr_1b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2048,
            n_layers: 24,
            d_ffn: 8192,
            max_seq_len: 128,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 4,
            head_size: 64,
            low_rank_dim: 0,
        }
    }

    /// 1.5B RWKV-4
    pub fn ptbr_1_5b() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 2304,
            n_layers: 28,
            d_ffn: 9216,
            max_seq_len: 64,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 4,
            head_size: 64,
            low_rank_dim: 0,
        }
    }

    // ================================
    // RWKV-7 Presets
    // ================================

    /// ~140M RWKV-7: d_model=768, 12 layers, head_size=64
    pub fn ptbr_140m_v7() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 768,
            n_layers: 12,
            d_ffn: 3072,
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 7,
            head_size: 64,
            low_rank_dim: 0,  // auto: 768/16 = 48
        }
    }

    /// ~400M RWKV-7: d_model=1024, 24 layers, head_size=64
    pub fn ptbr_400m_v7() -> Self {
        Self {
            vocab_size: 32_000,
            d_model: 1024,
            n_layers: 24,
            d_ffn: 4096,
            max_seq_len: 512,
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            weight_tying: true,
            rwkv_version: 7,
            head_size: 64,
            low_rank_dim: 0,  // auto: 1024/16 = 64
        }
    }

    pub fn num_parameters(&self) -> usize {
        let embed = self.vocab_size * self.d_model;

        let per_layer = {
            // Time mixing: 4 linear (d_model x d_model) + 5 param vectors (d_model)
            let time_mix = 4 * self.d_model * self.d_model + 5 * self.d_model;
            // Channel mixing: receptance(d,d) + key(d,ffn) + value(ffn,d) + 2 param vectors
            let channel_mix = self.d_model * self.d_model
                + self.d_model * self.d_ffn
                + self.d_ffn * self.d_model
                + 2 * self.d_model;
            // 2 LayerNorms: 2 * 2 * d_model
            let layer_norms = 4 * self.d_model;
            time_mix + channel_mix + layer_norms
        };

        // Head: 0 if weight tying, else d_model * vocab_size
        let head = if self.weight_tying {
            0
        } else {
            self.d_model * self.vocab_size
        };

        // Global: ln_pre + ln_out = 4 * d_model
        let global_ln = 4 * self.d_model;

        embed + self.n_layers * per_layer + head + global_ln
    }

    pub fn estimated_vram(&self, batch_size: usize, seq_len: usize) -> usize {
        let params = self.num_parameters();

        // bf16: 2 bytes per param
        let params_mem = params * 2;

        // Gradients: same as params
        let grad_mem = params * 2;

        // Optimizer states (AdamW): 2 moments in fp32 = 8 bytes per param
        let opt_mem = params * 8;

        // Activations per layer (rough estimate for RWKV)
        // Time mixing: 4 * batch * seq * d_model (k,v,r,output)
        // Channel mixing: 2 * batch * seq * d_ffn
        // All in bf16 (2 bytes)
        let acts_per_layer = (4 * batch_size * seq_len * self.d_model
            + 2 * batch_size * seq_len * self.d_ffn)
            * 2;
        let act_mem = acts_per_layer * self.n_layers;

        // WKV state per layer: 3 * batch * d_model * 4 (fp32 accumulators)
        let wkv_mem = 3 * batch_size * self.d_model * 4 * self.n_layers;

        // Embedding: batch * seq * d_model * 2
        let emb_mem = batch_size * seq_len * self.d_model * 2;

        // Logits: batch * seq * vocab * 2 (this can be big!)
        let logit_mem = batch_size * seq_len * self.vocab_size * 2;

        let total = params_mem + grad_mem + opt_mem + act_mem + wkv_mem + emb_mem + logit_mem;

        // 15% overhead for framework
        total + total / 7
    }

    pub fn estimated_vram_gb(&self, batch_size: usize, seq_len: usize) -> f64 {
        self.estimated_vram(batch_size, seq_len) as f64 / 1_000_000_000.0
    }

    pub fn fits_in_t4(&self, batch_size: usize, seq_len: usize) -> bool {
        self.estimated_vram(batch_size, seq_len) < 14_500_000_000
    }

    pub fn suggest_t4_config(&self) -> (usize, usize) {
        for seq_len in [512, 384, 256, 192, 128, 96, 64].iter() {
            for batch in [4, 2, 1].iter() {
                if self.fits_in_t4(*batch, *seq_len) {
                    return (*batch, *seq_len);
                }
            }
        }
        (1, 64)
    }
}

#[derive(Config, Debug)]
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

    #[config(default = "0.001")]
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
            batch_size: 4,
            gradient_accumulation_steps: 16,
            warmup_steps: 200,
            max_steps: 10_000,
            weight_decay: 0.001,  // Bug #10 fix: reduced from 0.01
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
            batch_size: 2,
            gradient_accumulation_steps: 32,  // ✅ effective batch = 64
            warmup_steps: 300,
            max_steps: 50_000,
            weight_decay: 0.001,  // Bug #20 fix: consistent with default
            gradient_clip: 1.0,
            save_every: 1000,
            log_every: 1,
            min_lr_ratio: 0.1,
        }
    }

    pub fn for_t4_safe() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 2,
            gradient_accumulation_steps: 32,
            warmup_steps: 500,
            max_steps: 50_000,
            weight_decay: 0.001,  // Bug #20 fix: consistent with default
            gradient_clip: 1.0,
            save_every: 1000,
            log_every: 1,
            min_lr_ratio: 0.1,
        }
    }
}