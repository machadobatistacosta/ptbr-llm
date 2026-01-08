// src/model/trainer.rs

use burn::{
    module::Module,
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},  // ← ADICIONA SimpleOptimizer e OptimizerAdaptor
    tensor::{backend::AutodiffBackend, Tensor, Int, ElementConversion, activation},
    record::CompactRecorder,
};

use super::rwkv::RWKV;
use super::config::{RWKVConfig, TrainingConfig};

pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,  // ← CORRIGE O TIPO
    config: TrainingConfig,
    step: usize,
    device: B::Device,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(
        model_config: &RWKVConfig,
        train_config: TrainingConfig,
        device: B::Device,
    ) -> Self {
        let model = RWKV::new(model_config, &device);
        
        // Cria optimizer UMA VEZ
        let optimizer = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay as f32)  // ← CONVERTE F64 → F32
            .init();
        
        println!("Modelo criado com {} parâmetros", model_config.num_parameters());
        
        Self {
            model,
            optimizer,
            config: train_config,
            step: 0,
            device,
        }
    }

    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> f32 {
        // Forward
        let logits = self.model.forward(input_ids);
        
        // Calcula loss
        let loss = self.cross_entropy_loss(logits, target_ids);
        
        // Extrai valor escalar
        let loss_scalar = loss.clone().into_scalar();
        let loss_value: f32 = loss_scalar.elem();
        
        // Backward
        let grads = loss.backward();
        
        // Atualiza pesos COM O MESMO OPTIMIZER
        let lr = self.get_learning_rate();
        let grad_params = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(lr, self.model.clone(), grad_params);  // ← AGORA FUNCIONA
        
        self.step += 1;
        
        loss_value
    }

    fn cross_entropy_loss(
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

    fn get_learning_rate(&self) -> f64 {
        let warmup = self.config.warmup_steps.max(1) as f64;
        let step = self.step as f64;

        if step < warmup {
            self.config.learning_rate * ((step + 1.0) / warmup)
        } else {
            let decay_steps = (self.config.max_steps.saturating_sub(self.config.warmup_steps)).max(1) as f64;
            let decay_progress = (step - warmup) / decay_steps;
            (self.config.learning_rate * (1.0 - 0.9 * decay_progress)).max(self.config.learning_rate * 0.1)
        }
}

    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".bin").trim_end_matches(".mpk");
        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    pub fn load_checkpoint(&mut self, path: &str) -> std::io::Result<()> {
        let path = path.trim_end_matches(".bin").trim_end_matches(".mpk");
        let recorder = CompactRecorder::new();
        self.model = self.model
            .clone()
            .load_file(path, &recorder, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        println!("    Pesos carregados de: {}.mpk", path);
        Ok(())
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn current_lr(&self) -> f64 {
        self.get_learning_rate()
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}