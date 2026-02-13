// src/model/lr_finder.rs
//! Learning Rate Finder para encontrar LR ótimo

use crate::data::{DataLoader, MmapDataset};
use crate::model::{RWKVConfig, Trainer, TrainingConfig};
use super::traits::RWKVModel;
use burn::module::AutodiffModule;
use burn::tensor::{backend::AutodiffBackend, Int, Tensor};

pub struct LRFinderResult {
    pub lrs: Vec<f64>,
    pub losses: Vec<f32>,
    pub suggested_lr: f64,
}

/// Encontra learning rate ótimo usando range test
pub fn find_lr<B, M>(
    model: M,
    model_config: &RWKVConfig,
    dataset: &MmapDataset,
    device: &B::Device,
    start_lr: f64,
    end_lr: f64,
    num_steps: usize,
) -> LRFinderResult
where
    B: AutodiffBackend,
    M: RWKVModel<B> + AutodiffModule<B>,
    <M as AutodiffModule<B>>::InnerModule: RWKVModel<B::InnerBackend>,
{
    let lr_mult = (end_lr / start_lr).powf(1.0 / num_steps as f64);
    let mut current_lr = start_lr;

    let mut lrs = Vec::with_capacity(num_steps);
    let mut losses = Vec::with_capacity(num_steps);

    let train_config = TrainingConfig {
        learning_rate: start_lr,
        batch_size: 1,
        gradient_accumulation_steps: 1,
        warmup_steps: 0,
        max_steps: num_steps,
        weight_decay: 0.0,
        gradient_clip: 1.0,
        save_every: num_steps + 1,
        log_every: 1,
        min_lr_ratio: 1.0,
        ..Default::default()
    };

    let mut trainer = Trainer::new(model, model_config, train_config, device.clone());
    let loader = DataLoader::new(dataset, 1);

    for (i, (inputs, targets)) in loader.into_iter().enumerate() {
        if i >= num_steps {
            break;
        }

        // Cria tensors manualmente (compatível com main.rs)
        let batch_size = inputs.len();
        let seq_len = inputs[0].len();
        let flat: Vec<i32> = inputs.iter().flatten().map(|&x| x as i32).collect();
        let input_tensor_1d: Tensor<B, 1, Int> = Tensor::from_ints(flat.as_slice(), device);
        let input_tensor: Tensor<B, 2, Int> = input_tensor_1d.reshape([batch_size, seq_len]);
        
        let flat_target: Vec<i32> = targets.iter().flatten().map(|&x| x as i32).collect();
        let target_tensor_1d: Tensor<B, 1, Int> = Tensor::from_ints(flat_target.as_slice(), device);
        let target_tensor: Tensor<B, 2, Int> = target_tensor_1d.reshape([batch_size, seq_len]);

        // Atualiza learning rate manualmente
        trainer.set_learning_rate(current_lr);

        if let Some(stats) = trainer.train_step(input_tensor, target_tensor) {
            lrs.push(current_lr);
            losses.push(stats.loss);

            // Early stop se loss explodiu
            if stats.loss > losses.first().unwrap_or(&10.0) * 10.0 || !stats.loss.is_finite() {
                break;
            }

            current_lr *= lr_mult;
        }
    }

    // Encontra LR sugerido (menor gradiente)
    let suggested = find_steepest_descent(&lrs, &losses);

    LRFinderResult {
        lrs,
        losses,
        suggested_lr: suggested,
    }
}

fn find_steepest_descent(lrs: &[f64], losses: &[f32]) -> f64 {
    if losses.len() < 3 {
        return lrs.get(lrs.len() / 2).copied().unwrap_or(1e-4);
    }

    let mut min_grad = f32::MAX;
    let mut best_idx = 0;

    for i in 1..losses.len() - 1 {
        let grad = (losses[i + 1] - losses[i - 1]) / 2.0;
        if grad < min_grad {
            min_grad = grad;
            best_idx = i;
        }
    }

    lrs[best_idx] / 10.0 // Um pouco antes do mínimo
}
