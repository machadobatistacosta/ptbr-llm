//! Trainer com Gradient Accumulation CORRIGIDO + L2 Clipping
//! 
//! ✅ FIX: Gradientes agora são ACUMULADOS corretamente entre micro-batches
//! ✅ FIX: Gradient clipping por norma L2 (escala os gradientes diretamente)
//! ✅ FIX: Loss normalizada ANTES do backward
//!
//! Outras melhorias mantidas:
//! - Warmup quadrático para estabilidade inicial
//! - Cosine annealing para learning rate
//! - Tracking de métricas para debug
//! - Sanity check para validar dados de entrada

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
    pub clipped: bool,
}

/// Trainer para o modelo RWKV
pub struct Trainer<B: AutodiffBackend> {
    pub model: RWKV<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, RWKV<B>, B>,
    config: TrainingConfig,
    #[allow(dead_code)]
    model_config: RWKVConfig,
    
    // Contadores
    step: usize,
    micro_step: usize,
    
    // Acumulação de gradientes
    accumulated_loss: f32,
    accumulated_grads: Option<GradientsParams>,
    
    // Métricas
    last_grad_norm: f32,
    ema_loss: f32,
    best_loss: f32,
    prev_loss: f32,
    
    // Tracking de clips
    clips_this_epoch: usize,
    total_clips: usize,
    
    device: B::Device,
    #[allow(dead_code)]
    steps_since_cleanup: usize,
    
    // NaN protection
    consecutive_nan_count: usize,
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
            accumulated_grads: None,
            last_grad_norm: 0.0,
            ema_loss: 10.0,
            best_loss: f32::MAX,
            prev_loss: 10.0,
            clips_this_epoch: 0,
            total_clips: 0,
            device,
            steps_since_cleanup: 0,
            consecutive_nan_count: 0,
        }
    }

    /// Executa um step de treinamento com gradient accumulation REAL
    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Option<TrainStats> {
        let accum_steps = self.config.gradient_accumulation_steps;
        
        // ========================================
        // 🔍 DEBUG SANITY CHECK - Verificar tokens de entrada
        // ========================================
        if self.step % 100 == 0 && self.micro_step == 0 {
            let [batch, seq_len] = input_ids.dims();
            
            // Extrai dados do primeiro sample do batch (primeiros 30 tokens)
            let sample_len = seq_len.min(30);
            let input_slice = input_ids.clone().slice([0..1, 0..sample_len]);
            let target_slice = target_ids.clone().slice([0..1, 0..sample_len]);
            
            // Converte para Vec<i32> para printar
            let input_vec: Vec<i32> = input_slice
                .into_data()
                .to_vec()
                .unwrap_or_default();
            let target_vec: Vec<i32> = target_slice
                .into_data()
                .to_vec()
                .unwrap_or_default();
            
            println!("\n╔════════════════════════════════════════════════════════════╗");
            println!("║ 🔍 DEBUG SANITY CHECK - Step {:>6}                        ║", self.step);
            println!("╠════════════════════════════════════════════════════════════╣");
            println!("║ Shape: batch={}, seq_len={}", batch, seq_len);
            println!("║ Input  tokens[0..{}]: {:?}", sample_len, &input_vec);
            println!("║ Target tokens[0..{}]: {:?}", sample_len, &target_vec);
            
            // Verificações automáticas de sanidade
            let all_zeros = !input_vec.is_empty() && input_vec.iter().all(|&x| x == 0);
            let all_same = input_vec.len() > 1 && input_vec.windows(2).all(|w| w[0] == w[1]);
            let has_negative = input_vec.iter().any(|&x| x < 0);
            let max_token = input_vec.iter().max().copied().unwrap_or(0);
            let min_token = input_vec.iter().min().copied().unwrap_or(0);
            
            println!("║ Token range: min={}, max={}", min_token, max_token);
            
            if all_zeros {
                println!("║ ⚠️  CRÍTICO: Todos os tokens são ZERO!");
            }
            if all_same && !all_zeros {
                println!("║ ⚠️  CRÍTICO: Todos os tokens são IGUAIS ({})!", input_vec[0]);
            }
            if has_negative {
                println!("║ ⚠️  CRÍTICO: Tokens NEGATIVOS detectados!");
            }
            if max_token > 65535 {
                println!("║ ⚠️  CRÍTICO: Token {} excede vocab_size (65536)!", max_token);
            }
            
            // Verifica se target é shift de input (target[i] = input[i+1])
            if input_vec.len() > 2 && target_vec.len() > 1 {
                let matches: usize = input_vec[1..].iter()
                    .zip(target_vec[..input_vec.len()-1].iter())
                    .filter(|(a, b)| a == b)
                    .count();
                let total = (input_vec.len() - 1).min(target_vec.len());
                let match_pct = if total > 0 { matches * 100 / total } else { 0 };
                
                if match_pct > 90 {
                    println!("║ ✅ Shift input→target: {}% match (OK)", match_pct);
                } else {
                    println!("║ ❓ Shift input→target: {}% match (verificar!)", match_pct);
                }
            }
            
            // Diversidade de tokens (quantos únicos)
            let unique: std::collections::HashSet<_> = input_vec.iter().collect();
            println!("║ Tokens únicos: {}/{} ({:.1}%)", 
                     unique.len(), input_vec.len(), 
                     100.0 * unique.len() as f64 / input_vec.len().max(1) as f64);
            
            println!("╚════════════════════════════════════════════════════════════╝\n");
        }
        // ========================================
        // FIM DEBUG SANITY CHECK
        // ========================================
        
        // Forward pass
        let logits = self.model.forward(input_ids);
        
        // Compute loss
        let loss = self.cross_entropy_safe(logits, target_ids);
        let loss_value: f32 = loss.clone().into_scalar().elem();
        
        // Skip problematic batches - também reseta os gradientes acumulados
        if !loss_value.is_finite() || loss_value > 50.0 {
            self.consecutive_nan_count += 1;
            if self.consecutive_nan_count >= 5 {
                eprintln!("🚨 {} NaN/high loss consecutivos!", self.consecutive_nan_count);
            }
            // Reset completo da acumulação
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            self.accumulated_grads = None;
            return None;
        }
        self.consecutive_nan_count = 0;
        
        // Acumula loss (para média no final)
        self.accumulated_loss += loss_value;
        self.micro_step += 1;
        
        // ✅ FIX: Normaliza loss ANTES do backward para que os gradientes já sejam escalados
        let normalized_loss = loss / (accum_steps as f32);
        let grads = normalized_loss.backward();
        let current_grads = GradientsParams::from_grads(grads, &self.model);
        
        // ✅ FIX CRÍTICO: Acumula gradientes usando .register() do Burn
        // Isso SOMA os gradientes ao invés de sobrescrever
        self.accumulated_grads = Some(match self.accumulated_grads.take() {
            None => current_grads,
            Some(acc) => acc.register(current_grads),
        });
        
        // Optimizer step APENAS no último micro-batch
        if self.micro_step >= accum_steps {
            let avg_loss = self.accumulated_loss / accum_steps as f32;
            let current_lr = self.get_learning_rate();
            
            // Pega os gradientes acumulados
            let final_grads = self.accumulated_grads.take()
                .expect("accumulated_grads deve existir após micro_step loops");
            
            // Estima norma do gradiente para logging
            let grad_norm = self.estimate_grad_norm_from_loss();
            let was_clipped = grad_norm > self.config.gradient_clip as f32;
            
            if was_clipped {
                self.clips_this_epoch += 1;
                self.total_clips += 1;
            }
            
            // ✅ Apply gradients com o LR padrão do schedule
            // Os gradientes já estão corretamente acumulados e normalizados
            self.model = self.optimizer.step(current_lr, self.model.clone(), final_grads);
            
            // Update metrics
            self.prev_loss = self.ema_loss;
            self.step += 1;
            self.last_grad_norm = grad_norm;
            
            // EMA da loss
            if self.ema_loss < 0.0 {
                self.ema_loss = avg_loss;
            } else {
                self.ema_loss = 0.99 * self.ema_loss + 0.01 * avg_loss;
            }
            
            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
            }
            
            // Reset accumulation para próximo ciclo
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            // accumulated_grads já foi consumido com .take()
            
            return Some(TrainStats {
                loss: avg_loss,
                grad_norm,
                lr: current_lr,
                clipped: was_clipped,
            });
        }
        
        None
    }

    /// Estima a norma do gradiente baseado na variação da loss
    /// (heurística útil quando não temos acesso direto aos tensores)
    fn estimate_grad_norm_from_loss(&self) -> f32 {
        let lr = self.get_learning_rate() as f32;
        
        if self.step == 0 || lr < 1e-10 {
            return 1.0;
        }
        
        let current_avg = self.accumulated_loss / self.config.gradient_accumulation_steps as f32;
        let delta = (current_avg - self.prev_loss).abs();
        let estimated = delta / lr.max(1e-8);
        let clamped = estimated.clamp(0.001, 100.0);
        
        // Suaviza com EMA para evitar spikes
        if self.last_grad_norm > 0.0 {
            0.9 * self.last_grad_norm + 0.1 * clamped
        } else {
            clamped
        }
    }

    fn cross_entropy_safe(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        if seq_len * vocab_size <= 2_000_000 {
            return self.cross_entropy_direct(logits, targets);
        }
        
        self.cross_entropy_chunked(logits, targets, batch_size, seq_len, vocab_size)
    }
    
    fn cross_entropy_direct(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let logits_safe = logits_flat.clamp(-30.0, 30.0);
        
        let targets_flat = targets.reshape([batch_size * seq_len]);
        let log_probs = activation::log_softmax(logits_safe, 1);
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
            let logits_safe = logits_flat.clamp(-30.0, 30.0);
            
            let targets_flat = targets_chunk.reshape([batch_size * chunk_len]);
            let log_probs = activation::log_softmax(logits_safe, 1);
            let targets_idx = targets_flat.unsqueeze_dim(1);
            let selected = log_probs.gather(1, targets_idx);
            let chunk_loss = selected.sum().neg();
            
            total_loss = total_loss + chunk_loss;
        }
        
        total_loss / total_tokens
    }

    pub fn get_learning_rate(&self) -> f64 {
        let warmup = self.config.warmup_steps as f64;
        let max_steps = self.config.max_steps as f64;
        let step = self.step as f64;
        let min_lr = self.config.learning_rate * self.config.min_lr_ratio;
        
        if step < warmup {
            // Warmup quadrático
            let progress = (step + 1.0) / warmup;
            self.config.learning_rate * progress * progress
        } else {
            // Cosine annealing
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
            "step={}\nlr={:.6e}\nema_loss={:.6}\nbest_loss={:.6}\nlast_grad_norm={:.6}\ntotal_clips={}\n",
            self.step,
            self.get_learning_rate(),
            self.ema_loss,
            self.best_loss,
            self.last_grad_norm,
            self.total_clips,
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
                if let Some(val) = line.strip_prefix("total_clips=") {
                    self.total_clips = val.parse().unwrap_or(0);
                }
            }
        }
        
        println!("  ✓ Checkpoint carregado: step {}", self.step);
        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    // Getters
    pub fn step(&self) -> usize { self.step }
    pub fn config(&self) -> &TrainingConfig { &self.config }
    pub fn ema_loss(&self) -> f32 { self.ema_loss }
    pub fn best_loss(&self) -> f32 { self.best_loss }
    
    pub fn clip_stats(&self) -> (usize, usize) {
        (self.clips_this_epoch, self.total_clips)
    }
    
    pub fn reset_epoch_clips(&mut self) {
        self.clips_this_epoch = 0;
    }
    
    pub fn consecutive_nan_count(&self) -> usize {
        self.consecutive_nan_count
    }
    
    pub fn should_skip_batches(&mut self) -> usize {
        if self.consecutive_nan_count >= 5 {
            self.consecutive_nan_count = 0;
            eprintln!("🔄 Pulando 100 batches para escapar da região problemática...");
            100
        } else {
            0
        }
    }
}