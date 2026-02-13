// src/model/evaluator.rs
//! Avaliador OTIMIZADO - Batch evaluation sem fragmentação de VRAM

use burn::tensor::{backend::Backend, activation, ElementConversion, Int, Tensor};
use super::traits::RWKVModel;
use crate::data::MmapDataset;

#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub loss: f32,
    pub perplexity: f32,
    pub tokens_evaluated: usize,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Loss: {:.4} | PPL: {:.2} | Tokens: {}",
            self.loss, self.perplexity, self.tokens_evaluated
        )
    }
}

pub struct Evaluator {
    num_samples: usize,
    // ✨ NOVO: Batch size para eval (reduz chamadas de kernel)
    batch_size: usize,
}

impl Evaluator {
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            batch_size: 4,  // Processa 4 samples de cada vez
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// ✨ Avaliação em batch (genérica sobre M: RWKVModel)
    pub fn evaluate<B: Backend, M: RWKVModel<B>>(
        &self,
        model: &M,
        dataset: &MmapDataset,
        device: &B::Device,
    ) -> EvalMetrics {
        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        let seq_len = dataset.seq_len();
        let start_idx = dataset.len().saturating_sub(self.num_samples);
        let end_idx = dataset.len();

        // ✨ Processa em batches
        let mut idx = start_idx;
        while idx < end_idx {
            let batch_end = (idx + self.batch_size).min(end_idx);
            let actual_batch = batch_end - idx;

            // Coleta batch
            let mut inputs_flat: Vec<i32> = Vec::with_capacity(actual_batch * seq_len);
            let mut targets_flat: Vec<i32> = Vec::with_capacity(actual_batch * seq_len);
            let mut valid_count = 0;

            for i in idx..batch_end {
                if let Some((input, target)) = dataset.get(i) {
                    inputs_flat.extend(input.iter().map(|&x| x as i32));
                    targets_flat.extend(target.iter().map(|&x| x as i32));
                    valid_count += 1;
                }
            }

            if valid_count == 0 {
                idx = batch_end;
                continue;
            }

            // Cria tensores do batch
            let input_tensor: Tensor<B, 2, Int> = {
                let data = burn::tensor::TensorData::from(inputs_flat.as_slice());
                let t: Tensor<B, 1, Int> = Tensor::from_data(data, device);
                t.reshape([valid_count, seq_len])
            };

            let target_tensor: Tensor<B, 2, Int> = {
                let data = burn::tensor::TensorData::from(targets_flat.as_slice());
                let t: Tensor<B, 1, Int> = Tensor::from_data(data, device);
                t.reshape([valid_count, seq_len])
            };

            // Forward (uses trait method)
            let logits = model.forward_train(input_tensor);
            let loss = self.cross_entropy::<B>(logits, target_tensor);

            total_loss += loss as f64 * (valid_count * seq_len) as f64;
            total_tokens += valid_count * seq_len;

            idx = batch_end;
        }

        let avg_loss = if total_tokens > 0 {
            total_loss / total_tokens as f64
        } else {
            f64::MAX
        };

        EvalMetrics {
            loss: avg_loss as f32,
            perplexity: avg_loss.exp().min(f64::MAX) as f32,
            tokens_evaluated: total_tokens,
        }
    }

    fn cross_entropy<B: Backend>(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> f32 {
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        let log_probs = activation::log_softmax(logits_flat, 1);
        let targets_idx = targets_flat.unsqueeze_dim(1);
        let selected = log_probs.gather(1, targets_idx);

        selected.mean().neg().into_scalar().elem()
    }
}