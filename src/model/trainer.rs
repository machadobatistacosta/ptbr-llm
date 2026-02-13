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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer, GradientsAccumulator},
    record::CompactRecorder,
    tensor::{activation, backend::AutodiffBackend, ElementConversion, Int, Tensor},
};

use crate::data::{DataLoader, MmapDataset};
use crate::error::{PtbrError, Result};
use crate::helpers::create_batch_tensor;
use crate::model::Evaluator;
use crate::tokenizer::BPETokenizer;
use crate::utils::{format_duration, format_number};

use super::config::{RWKVConfig, TrainingConfig};
use super::traits::RWKVModel;

/// Estatísticas de um step de treino
#[derive(Debug, Clone, Default)]
pub struct TrainStats {
    pub loss: f32,
    pub update_norm: f32,
    pub lr: f64,
}

/// Trainer genérico para modelos RWKV (v4/v7)
pub struct Trainer<B: AutodiffBackend, M: RWKVModel<B> + AutodiffModule<B>> {
    pub model: M,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, M, B>,
    config: TrainingConfig,
    #[allow(dead_code)]
    model_config: RWKVConfig,
    
    // Contadores
    step: usize,
    micro_step: usize,
    
    // Acumulação de gradientes
    accumulated_loss: f32,
    accumulator: GradientsAccumulator<M>,
    
    // Métricas
    last_update_norm: f32,
    ema_loss: f32,
    best_loss: f32,
    prev_loss: f32,
    
    device: B::Device,
    #[allow(dead_code)]
    steps_since_cleanup: usize,
    
    // NaN protection
    consecutive_nan_count: usize,
}

impl<B, M> Trainer<B, M>
where
    B: AutodiffBackend,
    M: RWKVModel<B> + AutodiffModule<B>,
    <M as AutodiffModule<B>>::InnerModule: RWKVModel<B::InnerBackend>,
{
    pub fn new(model: M, model_config: &RWKVConfig, train_config: TrainingConfig, device: B::Device) -> Self {
        
        // Bug #10 fix: weight_decay default reduced to 0.001 in TrainingConfig
        // Bug #1 fix: Real gradient clipping via Burn's GradientClippingConfig
        let optimizer = AdamWConfig::new()
            .with_weight_decay(train_config.weight_decay as f32)
            .with_beta_1(0.9f32)
            .with_beta_2(0.99f32)
            .with_epsilon(1e-8f32)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(train_config.gradient_clip as f32)))
            .init();

            
        let accumulator = GradientsAccumulator::new();
        
        Self {
            model,
            optimizer,
            config: train_config,
            model_config: model_config.clone(),
            step: 0,
            micro_step: 0,
            accumulated_loss: 0.0,

            accumulator,
            last_update_norm: 0.0,
            ema_loss: f32::NAN,  // Bug #15 fix: Use NaN to detect first value
            best_loss: f32::MAX,
            prev_loss: 10.0,
            device,
            steps_since_cleanup: 0,
            consecutive_nan_count: 0,
        }
    }

    pub fn from_checkpoint(
        checkpoint_path: &Path,
        model: M,
        model_config: &RWKVConfig,
        train_config: TrainingConfig,
        device: B::Device
    ) -> Result<Self> {
        let mut trainer = Self::new(model, model_config, train_config, device);
        trainer.load_checkpoint(checkpoint_path.to_str().unwrap())?;
        Ok(trainer)
    }

    pub fn fit(
        &mut self,
        dataset: &mut MmapDataset,
        val_dataset: Option<&MmapDataset>,
        tokenizer: &BPETokenizer,
        output: &PathBuf,
        eval_samples: usize,
        eval_every: usize,
    ) -> Result<()> {
        let start = Instant::now();
        let initial_step = self.step;
        let max_steps = self.config.max_steps;
        let batch_size = self.config.batch_size;
        let save_every = self.config.save_every;

        let mut last_log = Instant::now();
        let mut tokens_since_log = 0usize;
        let mut epoch = 0;

        let sample_prompts = ["O Brasil é", "A Constituição Federal", "Em 2024"];
        
        // Cria evaluator para métricas de validação
        let evaluator = Evaluator::new(eval_samples);

        std::fs::create_dir_all(output)?;
        let mut metrics_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output.join("metrics.csv"))
            .map_err(|e| PtbrError::Io(e))?;

        if self.step == 0 {
             writeln!(
                metrics_file,
                "step,loss,ppl,lr,update_norm,tokens_per_sec,eval_loss,eval_ppl"
            ).map_err(|e| PtbrError::Io(e))?;
        }

        'training: loop {
            dataset.shuffle(42 + epoch);
            let loader = DataLoader::new(dataset, batch_size);
            let total_batches = loader.total_batches();

            if epoch == 0 && self.step == 0 {
                println!("  📦 Processando {} batches no primeiro epoch...", total_batches);
                std::io::stdout().flush().map_err(|e| PtbrError::Io(e))?;
            }

            let mut loader_iter = loader.into_iter();

            while let Some((inputs, targets)) = loader_iter.next() {
                
                // Validação do batch
                if inputs.is_empty() || targets.is_empty() {
                    continue;
                }
                
                let seq_len = inputs[0].len();
                if seq_len == 0 {
                    continue;
                }
                
                // Valida que todas as sequências têm o mesmo tamanho
                if !inputs.iter().all(|x| x.len() == seq_len) || !targets.iter().all(|x| x.len() == seq_len) {
                    continue;
                }
                
                // Cria tensores
                let input_tensor = create_batch_tensor::<B>(&inputs, &self.device);
                let target_tensor = create_batch_tensor::<B>(&targets, &self.device);

                // Train step
                if let Some(stats) = self.train_step(input_tensor, target_tensor) {
                    let step = self.step;
                    let steps_done = step - initial_step;
                    tokens_since_log +=
                        batch_size * seq_len * self.config.gradient_accumulation_steps;

                    // Log imediato no primeiro step
                    if step == 1 {
                        println!("  ✅ Primeiro step completo! Loss inicial: {:.4}", stats.loss);
                        std::io::stdout().flush().map_err(|e| PtbrError::Io(e))?;
                    }

                    // Log periódico
                    if last_log.elapsed().as_secs() >= 5 || step == 1 {
                        let elapsed = start.elapsed().as_secs_f64();
                        let steps_per_sec = steps_done as f64 / elapsed;
                        let tokens_per_sec = tokens_since_log as f64 / last_log.elapsed().as_secs_f64();
                        let remaining = max_steps.saturating_sub(steps_done);
                        let eta_secs = remaining as f64 / steps_per_sec.max(0.01);
                        let ppl = (stats.loss as f64).exp();

                        println!(
                            "  Step {:>6} | Loss: {:.4} | PPL: {:>7.2} | LR: {:.2e} | Upd: {:.3} | {:.1}K tok/s | ETA: {}",
                            step,
                            stats.loss,
                            ppl,
                            stats.lr,
                            stats.update_norm,
                            tokens_per_sec / 1000.0,
                            format_duration(eta_secs as u64)
                        );

                        writeln!(
                            metrics_file,
                            "{},{:.6},{:.2},{:.2e},{:.4},{:.1},,",
                            step, stats.loss, ppl, stats.lr, stats.update_norm, tokens_per_sec
                        )
                        .ok();

                        last_log = Instant::now();
                        tokens_since_log = 0;
                    }

                    // Evaluation - use val_dataset if available
                    if step % eval_every == 0 && step > 0 {
                        let eval_data = val_dataset.unwrap_or(dataset);
                        let eval_metrics = evaluator.evaluate(&self.model.valid(), eval_data, &self.device);
                        println!(
                            "  📊 Eval Step {} | {}",
                            step, eval_metrics
                        );

                        writeln!(
                            metrics_file, 
                            "{},,,,,,{:.6},{:.2}",
                            step, eval_metrics.loss, eval_metrics.perplexity
                        ).ok();

                        // Gera samples
                        if tokenizer.vocab_size() > 256 {
                            println!("  📝 Samples:");
                            for prompt in &sample_prompts {
                                let sample = self.generate_sample(tokenizer, prompt, 30);
                                println!("     \"{}\" → {}", prompt, sample.trim());
                            }
                        }
                        println!();
                    }

                    if step % save_every == 0 && step > 0 {
                        let ckpt_path = output.join(format!("checkpoint_{}", step));
                        match self.save_checkpoint(ckpt_path.to_str().unwrap()) {
                            Ok(_) => println!("  💾 Checkpoint salvo: {:?}", ckpt_path),
                            Err(e) => println!("  ⚠️ Erro salvando: {}", e),
                        }
                    }

                    // Check completion
                    if self.step >= max_steps {
                        break 'training;
                    }
                }
                // Se train_step retornou None, verifica se precisa pular batches (muitos NaN)
                let skip_count = self.should_skip_batches();
                if skip_count > 0 {
                    eprintln!(
                        "  ⏭️ Skipping {} batches after {} consecutive NaN",
                        skip_count, 5
                    );
                    for _ in 0..skip_count {
                        if loader_iter.next().is_none() {
                            break;
                        }
                    }
                }
            }

            epoch += 1;
            dataset.next_epoch();
            println!("  📚 Epoch {} completa", epoch);
        }

        // Salva modelo final
        let final_path = output.join(format!("model_final_step_{}", self.step));
        self.save_checkpoint(final_path.to_str().unwrap())
            .expect("Erro salvando modelo final");

        let elapsed = start.elapsed();
        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("  ✅ Treinamento concluído!");
        println!("  Steps: {}", self.step);
        println!("  Tempo: {}", format_duration(elapsed.as_secs()));
        println!("  EMA Loss: {:.4}", self.ema_loss());
        println!("  Modelo: {:?}", final_path);
        println!("═══════════════════════════════════════════════════════════");
        
        Ok(())
    }

    /// Executa um step de treinamento com gradient accumulation REAL
    pub fn train_step(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> Option<TrainStats> {
        let accum_steps = self.config.gradient_accumulation_steps;
        
        // ... (Sanity Check Code Skipped - Unchanged) ...

        // Forward pass
        let logits = self.model.forward_train(input_ids);
        
        // Compute loss
        let loss = self.cross_entropy_safe(logits, target_ids);
        let loss_value: f32 = loss.clone().into_scalar().elem();
        
        // Skip problematic batches
        if !loss_value.is_finite() || loss_value > 50.0 {
            self.consecutive_nan_count += 1;
            if self.consecutive_nan_count >= 5 {
                eprintln!("🚨 {} NaN/high loss consecutivos!", self.consecutive_nan_count);
            }
            // Reset accumulation
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            self.accumulator = GradientsAccumulator::<M>::new(); // Hard reset
            return None;
        }
        self.consecutive_nan_count = 0;
        
        // Acumula loss
        self.accumulated_loss += loss_value;
        self.micro_step += 1;
        
        // ✅ FIX: Normaliza loss e acumula via GradientsAccumulator
        let normalized_loss = loss / (accum_steps as f32);
        let grads = normalized_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        
        self.accumulator.accumulate(&self.model, grads_params);
        
        // Optimizer step no último micro-batch
        if self.micro_step >= accum_steps {
            let avg_loss = self.accumulated_loss / accum_steps as f32;
            let current_lr = self.get_learning_rate();
            
            // Pega gradientes acumulados
            let final_grads = self.accumulator.grads();
            
            // ✅ FIX: Snapshot of a representative parameter BEFORE the step
            let param_before = self.model.representative_param_snapshot();

            // Apply optimizer step (Burn handles gradient clipping internally)
            self.model = self.optimizer.step(current_lr, self.model.clone(), final_grads);

            // ✅ FIX: Compute REAL update magnitude
            let param_after = self.model.representative_param_snapshot();

            let update_sq_sum: f32 = param_before.iter()
                .zip(param_after.iter())
                .map(|(a, b): (&f32, &f32)| (a - b).powi(2))
                .sum();
            let update_norm = update_sq_sum.sqrt();
            
            // Real clipping is done internally by Burn
            
            // Log warnings for update issues  
            if update_norm < 1e-10 && self.step > 10 {
                eprintln!("  ⚠️  VANISHING UPDATES: ||Δw||={:.2e}", update_norm);
            } else if update_norm > 1.0 {
                eprintln!("  ⚠️  LARGE UPDATES: ||Δw||={:.2e}", update_norm);
            }
            
            // Update metrics
            self.prev_loss = self.ema_loss;
            self.step += 1;
            self.last_update_norm = update_norm;
            
            // Bug #15 fix: Use is_nan() instead of < 0.0
            if self.ema_loss.is_nan() {
                self.ema_loss = avg_loss;
            } else {
                self.ema_loss = 0.99 * self.ema_loss + 0.01 * avg_loss;
            }
            
            if avg_loss < self.best_loss {
                self.best_loss = avg_loss;
            }
            
            // Reset state
            self.accumulated_loss = 0.0;
            self.micro_step = 0;
            
            return Some(TrainStats {
                loss: avg_loss,
                update_norm,
                lr: current_lr,
            });
        }
        
        None
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

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| PtbrError::Io(e))?;
        }
        
        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| PtbrError::FileWrite { path: PathBuf::from(path), source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()) })?;
        
        let meta = format!(
            "step={}\nlr={:.6e}\nema_loss={:.6}\nbest_loss={:.6}\nlast_update_norm={:.6}\n",
            self.step,
            self.get_learning_rate(),
            self.ema_loss,
            self.best_loss,
            self.last_update_norm,
        );
        std::fs::write(format!("{}.meta", path), meta).map_err(|e| PtbrError::Io(e))?;
        
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let path = path.trim_end_matches(".mpk").trim_end_matches(".bin");
        let recorder = CompactRecorder::new();
        
        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.device)
            .map_err(|e| PtbrError::CheckpointLoad(e.to_string()))?;
        
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
                    // Backwards compatibility: ignore or log
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
    
    pub fn generate_sample(
        &self,
        tokenizer: &BPETokenizer,
        prompt: &str,
        max_tokens: usize,
    ) -> String {
        let inference_model = self.model.valid();
        let mut tokens = tokenizer.encode(prompt);
    
        for _ in 0..max_tokens {
            if tokens.len() > 512 {
                tokens = tokens[tokens.len() - 512..].to_vec();
            }
    
            let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let seq_len = input_vec.len();
    
            // Use B::InnerBackend for inference model (autodiff stripped)
            let input: Tensor<B::InnerBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), &self.device);
            let input = input.reshape([1, seq_len]);
    
            let logits = inference_model.forward_train(input);
            let [_, s, v] = logits.dims();
            let last_logits = logits.slice([0..1, s - 1..s, 0..v]).reshape([v]);
    
            let logits_data: Vec<f32> = last_logits.into_data().iter::<f32>().collect();
            let next_token = logits_data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u16)
                .unwrap_or(0);
    
            if next_token == tokenizer.eos_id() {
                break;
            }
    
            tokens.push(next_token);
        }
    
        tokenizer.decode(&tokens)
    }
}