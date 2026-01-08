#![allow(dead_code)]
use burn::{
    module::Module,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{backend::AutodiffBackend, Tensor, Int, ElementConversion, activation},
    record::CompactRecorder,
};
use super::rwkv::RWKV;
use super::config::{RWKVConfig, TrainingConfig};

pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,
    config: TrainingConfig,
    
    // Estado de treino
    step: usize,
    micro_step: usize,
    accumulated_loss: f32,
    
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(model_config: &RWKVConfig, train_config: TrainingConfig, device: B::Device) -> Self {
        let model = RWKV::new(model_config, &device);
        let optimizer = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay as f32)
            .init();
        
        println!("  Modelo inicializado: {} parâmetros", format_params(model_config.num_parameters()));
        
        Self {
            model,
            optimizer,
            config: train_config,
            step: 0,
            micro_step: 0,
            accumulated_loss: 0.0,
            device,
        }
    }

    /// Um passo de treino com gradient accumulation
    /// Retorna Some(loss) quando um step completo foi feito, None caso contrário
    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Option<f32> {
        // Forward
        let logits = self.model.forward(input_ids);
        
        // Loss
        let loss = self.cross_entropy_loss(logits, target_ids);
        let loss_value: f32 = loss.clone().into_scalar().elem();
        
        // Verifica NaN/Inf
        if !loss_value.is_finite() {
            panic!("❌ Loss divergiu (NaN/Inf)! Step {}", self.step);
        }
        
        // Acumula loss (média móvel)
        self.accumulated_loss += loss_value;
        self.micro_step += 1;
        
        // Backward (sempre faz, acumula no grafo)
        let grads = loss.backward();
        let grad_params = GradientsParams::from_grads(grads, &self.model);
        
        // Só atualiza pesos quando completou N micro-steps
        if self.micro_step >= self.config.gradient_accumulation_steps {
            let lr = self.get_learning_rate();
            
            // Aplica gradientes acumulados
            self.model = self.optimizer.step(lr, self.model.clone(), grad_params);
            
            let avg_loss = self.accumulated_loss / self.micro_step as f32;
            
            // Reset acumuladores
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            self.step += 1;
            
            return Some(avg_loss);
        }
        
        None // Ainda acumulando
    }

    fn cross_entropy_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        // Flatten: [B, S, V] -> [B*S, V]
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);
        
        // Log softmax para estabilidade numérica
        let log_probs = activation::log_softmax(logits_flat, 1);
        
        // Gather: pega o log-prob do token correto
        let targets_idx = targets_flat.unsqueeze_dim(1);
        let selected = log_probs.gather(1, targets_idx);
        
        // Negative log likelihood
        selected.mean().neg()
    }

    /// Cosine annealing com warmup
    fn get_learning_rate(&self) -> f64 {
        let warmup = self.config.warmup_steps as f64;
        let max_steps = self.config.max_steps as f64;
        let step = self.step as f64;
        let min_lr = self.config.learning_rate * 0.1;
        
        if step < warmup {
            // Linear warmup
            self.config.learning_rate * (step + 1.0) / warmup
        } else {
            // Cosine decay
            let progress = (step - warmup) / (max_steps - warmup);
            let progress = progress.min(1.0); // Clamp para não passar de 1
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            min_lr + (self.config.learning_rate - min_lr) * cosine
        }
    }

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        // Salva metadados
        let meta = format!("step={}\nlr={:.2e}\n", self.step, self.get_learning_rate());
        std::fs::write(format!("{}.meta", path), meta)?;
        
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        let recorder = CompactRecorder::new();
        self.model = self.model
            .clone()
            .load_file(path, &recorder, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        // Tenta carregar metadados
        if let Ok(meta) = std::fs::read_to_string(format!("{}.meta", path)) {
            for line in meta.lines() {
                if let Some(step_str) = line.strip_prefix("step=") {
                    self.step = step_str.parse().unwrap_or(0);
                }
            }
        }
        
        println!("  ✅ Checkpoint carregado: step {}", self.step);
        Ok(())
    }

    pub fn step(&self) -> usize { self.step }
    pub fn micro_step(&self) -> usize { self.micro_step }
    pub fn current_lr(&self) -> f64 { self.get_learning_rate() }
    pub fn config(&self) -> &TrainingConfig { &self.config }
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { n.to_string() }
}