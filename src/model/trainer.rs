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
    #[allow(dead_code)]
    pub tokens_per_sec: f32,
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
    accumulated_loss: f32,

    // Métricas
    last_grad_norm: f32,
    ema_loss: f32,
    best_loss: f32,

    device: B::Device,
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
            ema_loss: 10.0, // Começa alto
            best_loss: f32::MAX,
            device,
        }
    }

    /// Executa um micro-step de treino
    /// Retorna Some(stats) quando completa um step completo (após gradient accumulation)
    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Option<TrainStats> {
        // Forward
        let logits = self.model.forward(input_ids);

        // Cross-entropy loss
        let loss = self.cross_entropy_loss(logits, target_ids);
        let loss_value: f32 = loss.clone().into_scalar().elem();

        // Verifica divergência
        if !loss_value.is_finite() {
            eprintln!("❌ ERRO: Loss divergiu para NaN/Inf no step {}!", self.step);
            eprintln!("   EMA Loss anterior: {:.4}", self.ema_loss);
            eprintln!("   Último grad_norm: {:.4}", self.last_grad_norm);
            panic!("Training diverged - loss became NaN/Inf");
        }

        // Alerta se loss muito alta
        if loss_value > 20.0 && self.step > 100 {
            eprintln!(
                "⚠️  Warning: Loss muito alta ({:.4}) no step {}",
                loss_value, self.step
            );
        }

        // Acumula
        self.accumulated_loss += loss_value;
        self.micro_step += 1;

        // Backward
        let grads = loss.backward();
        let grad_params = GradientsParams::from_grads(grads, &self.model);

        // Atualiza apenas quando completou gradient accumulation
        if self.micro_step >= self.config.gradient_accumulation_steps {
            let lr = self.get_learning_rate();

            // TODO: Gradient clipping quando Burn expor API
            // Por enquanto, confiamos no Adam para estabilidade

            // Optimizer step
            self.model = self.optimizer.step(lr, self.model.clone(), grad_params);

            // Calcula métricas
            let avg_loss = self.accumulated_loss / self.micro_step as f32;

            // Atualiza EMA
            if self.step == 0 {
                self.ema_loss = avg_loss;
            } else {
                self.ema_loss = 0.99 * self.ema_loss + 0.01 * avg_loss;
            }

            // Atualiza best loss
            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
            }

            // Reset para próximo step
            let stats = TrainStats {
                loss: avg_loss,
                grad_norm: self.last_grad_norm,
                lr,
                tokens_per_sec: 0.0, // Calculado externamente
            };

            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            self.step += 1;

            return Some(stats);
        }

        None
    }

    fn cross_entropy_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        // Log softmax para estabilidade
        let log_probs = activation::log_softmax(logits_flat, 1);

        // Gather log-probs dos tokens corretos
        let targets_idx = targets_flat.unsqueeze_dim(1);
        let selected = log_probs.gather(1, targets_idx);

        // Negative log likelihood
        selected.mean().neg()
    }

    /// Cosine annealing com linear warmup
    fn get_learning_rate(&self) -> f64 {
        let warmup = self.config.warmup_steps as f64;
        let max_steps = self.config.max_steps as f64;
        let step = self.step as f64;
        let min_lr = self.config.learning_rate * self.config.min_lr_ratio;

        if step < warmup {
            // Linear warmup
            self.config.learning_rate * (step + 1.0) / warmup
        } else {
            // Cosine decay
            let progress = (step - warmup) / (max_steps - warmup).max(1.0);
            let progress = progress.min(1.0);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (self.config.learning_rate - min_lr) * cosine
        }
    }

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        std::fs::create_dir_all(path)?;
        
        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Salva metadados
        let meta = format!(
            "step={}\n\
             lr={:.6e}\n\
             ema_loss={:.6}\n\
             best_loss={:.6}\n\
             last_grad_norm={:.6}\n",
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

        // Carrega metadados
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

        println!(
            "  ✓ Checkpoint carregado: step {}, ema_loss {:.4}",
            self.step, self.ema_loss
        );
        Ok(())
    }

    /// Define learning rate manualmente (para LR finder)
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    // Getters
    pub fn step(&self) -> usize { self.step }
    pub fn config(&self) -> &TrainingConfig { &self.config }
    pub fn ema_loss(&self) -> f32 { self.ema_loss }
    #[allow(dead_code)]
    pub fn micro_step(&self) -> usize { self.micro_step }
    #[allow(dead_code)]
    pub fn current_lr(&self) -> f64 { self.get_learning_rate() }
    #[allow(dead_code)]
    pub fn best_loss(&self) -> f32 { self.best_loss }
    #[allow(dead_code)]
    pub fn last_grad_norm(&self) -> f32 { self.last_grad_norm }
    #[allow(dead_code)]
    pub fn model_config(&self) -> &RWKVConfig { &self.model_config }
}