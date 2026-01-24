#![allow(dead_code)]
// src/model/validation.rs
//! Sistema de validação durante treino

use crate::data::MmapDataset;
use crate::model::RWKV;
use crate::tokenizer::BPETokenizer;
use burn::tensor::{activation, backend::Backend, Int, Tensor};
use std::collections::VecDeque;

/// Métricas de validação
#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    pub loss: f32,
    pub perplexity: f32,
    pub tokens_evaluated: usize,
    pub accuracy_top1: f32,
    pub accuracy_top5: f32,
    pub entropy: f32,
}

impl std::fmt::Display for ValidationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Loss: {:.4} | PPL: {:.2} | Acc@1: {:.2}% | Acc@5: {:.2}% | H: {:.2}",
            self.loss,
            self.perplexity,
            self.accuracy_top1 * 100.0,
            self.accuracy_top5 * 100.0,
            self.entropy
        )
    }
}

/// Early Stopping
#[derive(Debug)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    counter: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::MAX,
            counter: 0,
            stopped: false,
        }
    }

    pub fn check(&mut self, val_loss: f32) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.stopped = true;
            }
        }
        self.stopped
    }

    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }
}

/// Validador com múltiplas métricas
pub struct Validator {
    num_samples: usize,
    sample_prompts: Vec<String>,
    history: VecDeque<ValidationMetrics>,
    max_history: usize,
}

impl Validator {
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            sample_prompts: vec![
                "O Brasil é".to_string(),
                "A Constituição Federal".to_string(),
                "Em 2024".to_string(),
                "O presidente".to_string(),
                "A cidade de São Paulo".to_string(),
            ],
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    pub fn with_prompts(mut self, prompts: Vec<String>) -> Self {
        self.sample_prompts = prompts;
        self
    }

    /// Avalia modelo e retorna métricas completas
    pub fn validate<B: Backend>(
        &mut self,
        model: &RWKV<B>,
        dataset: &MmapDataset,
        device: &B::Device,
    ) -> ValidationMetrics {
        let mut total_loss = 0.0f64;
        let mut total_correct_top1 = 0usize;
        let mut total_correct_top5 = 0usize;
        let mut total_entropy = 0.0f64;
        let mut total_tokens = 0usize;

        let start_idx = dataset.len().saturating_sub(self.num_samples);

        for idx in start_idx..dataset.len() {
            if let Some((input, target)) = dataset.get(idx) {
                let (loss, acc1, acc5, entropy, tokens) =
                    self.evaluate_sample::<B>(model, &input, &target, device);

                total_loss += loss as f64 * tokens as f64;
                total_correct_top1 += acc1;
                total_correct_top5 += acc5;
                total_entropy += entropy as f64 * tokens as f64;
                total_tokens += tokens;
            }
        }

        let avg_loss = (total_loss / total_tokens.max(1) as f64) as f32;
        let perplexity = avg_loss.exp();
        let accuracy_top1 = total_correct_top1 as f32 / total_tokens.max(1) as f32;
        let accuracy_top5 = total_correct_top5 as f32 / total_tokens.max(1) as f32;
        let avg_entropy = (total_entropy / total_tokens.max(1) as f64) as f32;

        let metrics = ValidationMetrics {
            loss: avg_loss,
            perplexity,
            tokens_evaluated: total_tokens,
            accuracy_top1,
            accuracy_top5,
            entropy: avg_entropy,
        };

        // Guarda histórico
        self.history.push_back(metrics.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        metrics
    }

    fn evaluate_sample<B: Backend>(
        &self,
        model: &RWKV<B>,
        input: &[u16],
        target: &[u16],
        device: &B::Device,
    ) -> (f32, usize, usize, f32, usize) {
        let input_i32: Vec<i32> = input.iter().map(|&x| x as i32).collect();
        let input_tensor: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::from_ints(input_i32.as_slice(), device).reshape([1, input.len()]);

        let logits = model.forward(input_tensor);
        let [_, seq_len, vocab_size] = logits.dims();

        // Flatten para cálculo
        let logits_flat = logits.reshape([seq_len, vocab_size]);
        let probs = activation::softmax(logits_flat.clone(), 1);
        let log_probs = activation::log_softmax(logits_flat, 1);

        let mut loss = 0.0f32;
        let mut correct_top1 = 0usize;
        let mut correct_top5 = 0usize;
        let mut entropy = 0.0f32;

        // Calcula métricas por posição
        for (i, &t) in target.iter().enumerate() {
            let pos_probs: Vec<f32> = probs
                .clone()
                .slice([i..i + 1, 0..vocab_size])
                .reshape([vocab_size])
                .into_data()
                .iter::<f32>()
                .collect();

            let pos_log_probs: Vec<f32> = log_probs
                .clone()
                .slice([i..i + 1, 0..vocab_size])
                .reshape([vocab_size])
                .into_data()
                .iter::<f32>()
                .collect();

            // Loss
            loss -= pos_log_probs[t as usize];

            // Top-1 accuracy
            let top1 = pos_probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if top1 == t as usize {
                correct_top1 += 1;
            }

            // Top-5 accuracy
            let mut indexed: Vec<(usize, f32)> =
                pos_probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<usize> = indexed.iter().take(5).map(|(i, _)| *i).collect();
            if top5.contains(&(t as usize)) {
                correct_top5 += 1;
            }

            // Entropy
            for &p in &pos_probs {
                if p > 1e-10 {
                    entropy -= p * p.ln();
                }
            }
        }

        let n = target.len();
        (loss / n as f32, correct_top1, correct_top5, entropy / n as f32, n)
    }

    /// Gera samples de texto para inspeção qualitativa
    pub fn generate_samples<B: Backend>(
        &self,
        model: &RWKV<B>,
        tokenizer: &BPETokenizer,
        device: &B::Device,
        max_tokens: usize,
    ) -> Vec<(String, String)> {
        self.sample_prompts
            .iter()
            .map(|prompt| {
                let generated = self.generate_one(model, tokenizer, prompt, device, max_tokens);
                (prompt.clone(), generated)
            })
            .collect()
    }

    fn generate_one<B: Backend>(
        &self,
        model: &RWKV<B>,
        tokenizer: &BPETokenizer,
        prompt: &str,
        device: &B::Device,
        max_tokens: usize,
    ) -> String {
        let mut tokens = tokenizer.encode(prompt);

        for _ in 0..max_tokens {
            if tokens.len() > 512 {
                tokens = tokens[tokens.len() - 512..].to_vec();
            }

            let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let seq_len = input_vec.len();

            let input: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_ints(input_vec.as_slice(), device).reshape([1, seq_len]);

            let logits = model.forward_inference(input);
            let [_, _v] = logits.dims();
            let logits_data: Vec<f32> = logits.into_data().iter::<f32>().collect();

            // Greedy
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

    /// Retorna tendência de melhoria
    pub fn improvement_trend(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f32> = self
            .history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.loss)
            .collect();
        let older: Vec<f32> = self
            .history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|m| m.loss)
            .collect();

        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let older_avg: f32 = older.iter().sum::<f32>() / older.len() as f32;

        (older_avg - recent_avg) / older_avg
    }
}