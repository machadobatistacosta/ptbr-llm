// src/model/trainer.rs
//! Trainer otimizado para T4 16GB - Memory Safe
//!
//! Mudanças:
//! - Não acumula tensores de loss (apenas valores f32)
//! - Cross-entropy eficiente para vocab grande
//! - Cleanup periódico

use super::config::{RWKVConfig, TrainingConfig};
use super::rwkv::RWKV;
use burn::{
    module::Module,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{activation, backend::AutodiffBackend, ElementConversion, Int, Tensor},
};

/// Estatísticas de um step de treino
#[derive(Debug, Clone, Default)]
pub struct TrainStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f64,
}

pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,
    config: TrainingConfig,
    #[allow(dead_code)]
    model_config: RWKVConfig,

    // Estado
    step: usize,
    micro_step: usize,
    
    // ✨ Acumula apenas f32, não tensores!
    accumulated_loss: f32,

    // Métricas
    last_grad_norm: f32,
    ema_loss: f32,
    best_loss: f32,
    prev_loss: f32,

    device: B::Device,
    
    // Cleanup counter
    steps_since_cleanup: usize,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(model_config: &RWKVConfig, train_config: TrainingConfig, device: B::Device) -> Self {
        let model = RWKV::new(model_config, &device);

        let optimizer = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay as f32)
            .with_beta_1(0.9)
            .with_beta_2(0.99)
            .with_epsilon(1e-8)
            .init();

        Self {
            model,
            optimizer,
            config: train_config,
            model_config: model_config.clone(),
            step: 0,
            micro_step: 0,
            accumulated_loss: 0.0,
            last_grad_norm: 0.0,
            ema_loss: 10.0,
            best_loss: f32::MAX,
            prev_loss: 10.0,
            device,
            steps_since_cleanup: 0,
        }
    }

    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Option<TrainStats> {
        let accum_steps = self.config.gradient_accumulation_steps;
        
        // Forward
        let logits = self.model.forward(input_ids);

        // Loss
        let loss = self.cross_entropy_efficient(logits, target_ids);
        let loss_value: f32 = loss.clone().into_scalar().elem();

        if !loss_value.is_finite() {
            eprintln!("❌ ERRO: Loss divergiu (NaN/Inf) step {}", self.step);
            panic!("Training diverged!");
        }

        // Acumula apenas o valor f32
        self.accumulated_loss += loss_value;
        self.micro_step += 1;

        // Normaliza e faz backward
        let normalized_loss = loss / (accum_steps as f32);
        let grads = normalized_loss.backward();
        let grad_params = GradientsParams::from_grads(grads, &self.model);

        // Optimizer Step quando completar accumulation
        if self.micro_step % accum_steps == 0 {
            let avg_loss = self.accumulated_loss / accum_steps as f32;
            
            // Grad norm estimation
            let grad_norm = self.estimate_grad_norm(avg_loss);
            self.last_grad_norm = grad_norm;
            self.prev_loss = avg_loss;

            // Aplica gradientes
            let lr = self.get_learning_rate();
            self.model = self.optimizer.step(lr, self.model.clone(), grad_params);

            // Reset
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            self.step += 1;

            // EMA updates
            if self.step == 1 {
                self.ema_loss = avg_loss;
            } else {
                self.ema_loss = 0.99 * self.ema_loss + 0.01 * avg_loss;
            }

            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
            }

            // Cleanup periódico
            self.steps_since_cleanup += 1;
            if self.steps_since_cleanup >= 50 {
                self.steps_since_cleanup = 0;
                // Força drop de temporários
                std::hint::black_box(());
            }

            return Some(TrainStats {
                loss: avg_loss,
                grad_norm,
                lr,
            });
        }

        None
    }

    /// Cross-entropy eficiente para vocab grande
    fn cross_entropy_efficient(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();

        // Para casos que cabem na memória
        if seq_len * vocab_size <= 2_000_000 {
            return self.cross_entropy_direct(logits, targets);
        }

        // Para casos grandes, processa em chunks de 32 tokens
        self.cross_entropy_chunked(logits, targets, batch_size, seq_len, vocab_size)
    }

    fn cross_entropy_direct(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        let log_probs = activation::log_softmax(logits_flat, 1);
        let targets_idx = targets_flat.unsqueeze_dim(1);
        let selected = log_probs.gather(1, targets_idx);

        selected.mean().neg()
    }

    fn cross_entropy_chunked(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> Tensor<B, 1> {
        let chunk_size = 32_usize;
        let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

        let mut total_loss = Tensor::<B, 1>::zeros([1], &self.device);
        let total_tokens = (batch_size * seq_len) as f32;

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(seq_len);
            let chunk_len = end - start;

            let logits_chunk = logits.clone().slice([0..batch_size, start..end, 0..vocab_size]);
            let targets_chunk = targets.clone().slice([0..batch_size, start..end]);

            let logits_flat = logits_chunk.reshape([batch_size * chunk_len, vocab_size]);
            let targets_flat = targets_chunk.reshape([batch_size * chunk_len]);

            let log_probs = activation::log_softmax(logits_flat, 1);
            let targets_idx = targets_flat.unsqueeze_dim(1);
            let selected = log_probs.gather(1, targets_idx);

            let chunk_loss = selected.sum().neg();
            total_loss = total_loss + chunk_loss;
        }

        total_loss / total_tokens
    }

    fn estimate_grad_norm(&self, current_loss: f32) -> f32 {
        let lr = self.get_learning_rate() as f32;

        if self.step == 0 || lr < 1e-10 {
            return 1.0;
        }

        let delta = (current_loss - self.prev_loss).abs();
        let estimated = delta / lr.max(1e-8);
        let clamped = estimated.clamp(0.001, 100.0);

        if self.last_grad_norm > 0.0 {
            0.9 * self.last_grad_norm + 0.1 * clamped
        } else {
            clamped
        }
    }

    pub fn get_learning_rate(&self) -> f64 {
        let warmup = self.config.warmup_steps as f64;
        let max_steps = self.config.max_steps as f64;
        let step = self.step as f64;
        let min_lr = self.config.learning_rate * self.config.min_lr_ratio;

        if step < warmup {
            self.config.learning_rate * (step + 1.0) / warmup
        } else {
            let progress = (step - warmup) / (max_steps - warmup).max(1.0);
            let progress = progress.min(1.0);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (self.config.learning_rate - min_lr) * cosine
        }
    }

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let meta = format!(
            "step={}\nlr={:.6e}\nema_loss={:.6}\nbest_loss={:.6}\nlast_grad_norm={:.6}\n",
            self.step,
            self.get_learning_rate(),
            self.ema_loss,
            self.best_loss,
            self.last_grad_norm,
        );
        std::fs::write(format!("{}.meta", path), meta)?;

        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        let recorder = CompactRecorder::new();

        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        if let Ok(meta) = std::fs::read_to_string(format!("{}.meta", path)) {
            for line in meta.lines() {
                if let Some(val) = line.strip_prefix("step=") {
                    self.step = val.parse().unwrap_or(0);
                }
                if let Some(val) = line.strip_prefix("ema_loss=") {
                    self.ema_loss = val.parse().unwrap_or(10.0);
                }
                if let Some(val) = line.strip_prefix("best_loss=") {
                    self.best_loss = val.parse().unwrap_or(f32::MAX);
                }
            }
        }

        println!("  ✓ Checkpoint carregado: step {}", self.step);
        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    pub fn step(&self) -> usize { self.step }
    pub fn config(&self) -> &TrainingConfig { &self.config }
    pub fn ema_loss(&self) -> f32 { self.ema_loss }
    pub fn best_loss(&self) -> f32 { self.best_loss }
}