//! Avaliador para calcular métricas durante treino

use burn::tensor::{backend::Backend, activation, ElementConversion, Int, Tensor};
use crate::model::RWKV;
use crate::data::MmapDataset;

/// Métricas de avaliação
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub loss: f32,
    pub perplexity: f32,
    pub tokens_evaluated: usize,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Loss: {:.4} | PPL: {:.2} | Tokens: {}", 
            self.loss, self.perplexity, self.tokens_evaluated)
    }
}

/// Avaliador de modelo
pub struct Evaluator {
    num_samples: usize,
}

impl Evaluator {
    pub fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }

    /// Avalia modelo em subset do dataset
    pub fn evaluate<B: Backend>(
        &self,
        model: &RWKV<B>,
        dataset: &MmapDataset,
        device: &B::Device,
    ) -> EvalMetrics {
        let mut total_loss = 0.0f64;
        let mut total_tokens = 0usize;

        // Pega últimos N samples como validation
        let start_idx = dataset.len().saturating_sub(self.num_samples);
        
        for idx in start_idx..dataset.len() {
            if let Some((input, target)) = dataset.get(idx) {
                let seq_len = input.len();
                let input_tensor = self.create_tensor::<B>(&input, seq_len, device);
                let target_tensor = self.create_tensor::<B>(&target, seq_len, device);

                // Forward sem gradientes
                let logits = model.forward(input_tensor);
                let loss = self.cross_entropy::<B>(logits, target_tensor);
                
                total_loss += loss as f64 * seq_len as f64;
                total_tokens += seq_len;
            }
        }

        let avg_loss = if total_tokens > 0 {
            total_loss / total_tokens as f64
        } else {
            0.0
        };
        
        let perplexity = avg_loss.exp();

        EvalMetrics {
            loss: avg_loss as f32,
            perplexity: perplexity as f32,
            tokens_evaluated: total_tokens,
        }
    }

    fn create_tensor<B: Backend>(
        &self, 
        data: &[u16], 
        seq_len: usize,
        device: &B::Device
    ) -> Tensor<B, 2, Int> {
        let data_i32: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        // Cria tensor 1D primeiro, depois reshape
        let tensor: Tensor<B, 1, Int> = Tensor::from_ints(data_i32.as_slice(), device);
        tensor.reshape([1, seq_len])
    }

    fn cross_entropy<B: Backend>(
        &self, 
        logits: Tensor<B, 3>, 
        targets: Tensor<B, 2, Int>
    ) -> f32 {
        let [batch_size, seq_len, vocab_size] = logits.dims();
        
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);
        
        let log_probs = activation::log_softmax(logits_flat, 1);
        let targets_idx = targets_flat.unsqueeze_dim(1);
        let selected = log_probs.gather(1, targets_idx);
        
        // Usa ElementConversion para converter scalar
        let loss_scalar = selected.mean().neg().into_scalar();
        loss_scalar.elem::<f32>()
    }
}