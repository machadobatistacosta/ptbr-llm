// src/model/lr_finder.rs
//! Learning Rate Finder para encontrar LR ótimo

use crate::data::{DataLoader, MmapDataset};
use crate::model::{RWKVConfig, RWKV};
use burn::optim::{AdamW, AdamWConfig};
use burn::tensor::{backend::AutodiffBackend, Int, Tensor};

pub struct LRFinderResult {
    pub lrs: Vec<f64>,
    pub losses: Vec<f32>,
    pub suggested_lr: f64,
}

pub fn find_lr<B: AutodiffBackend>(
    model_config: &RWKVConfig,
    dataset: &MmapDataset,
    device: &B::Device,
    start_lr: f64,
    end_lr: f64,
    num_steps: usize,
) -> LRFinderResult {
    let mut model: RWKV<B> = RWKV::new(model_config, device);
    let mut optimizer = AdamWConfig::new().init::<B>();

    let lr_mult = (end_lr / start_lr).powf(1.0 / num_steps as f64);
    let mut current_lr = start_lr;

    let mut lrs = Vec::with_capacity(num_steps);
    let mut losses = Vec::with_capacity(num_steps);

    let loader = DataLoader::new(dataset, 1);
    let mut iter = loader.into_iter().cycle();

    for _ in 0..num_steps {
        if let Some((inputs, targets)) = iter.next() {
            // Forward + loss
            let input_tensor = create_tensor::<B>(&inputs[0], device);
            let target_tensor = create_tensor::<B>(&targets[0], device);

            let logits = model.forward(input_tensor);
            let loss = cross_entropy::<B>(logits, target_tensor);
            let loss_val: f32 = loss.clone().into_scalar().elem();

            // Backward
            let grads = loss.backward();
            // ... update

            lrs.push(current_lr);
            losses.push(loss_val);

            current_lr *= lr_mult;

            // Early stop if loss explodes
            if loss_val > losses.first().unwrap_or(&10.0) * 10.0 {
                break;
            }
        }
    }

    // Find suggested LR (steepest descent)
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

fn create_tensor<B: AutodiffBackend>(data: &[u16], device: &B::Device) -> Tensor<B, 2, Int> {
    let data_i32: Vec<i32> = data.iter().map(|&x| x as i32).collect();
    Tensor::from_ints(data_i32.as_slice(), device).reshape([1, data.len()])
}

fn cross_entropy<B: AutodiffBackend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
    use burn::tensor::activation;
    let [b, s, v] = logits.dims();
    let logits_flat = logits.reshape([b * s, v]);
    let targets_flat = targets.reshape([b * s]);
    let log_probs = activation::log_softmax(logits_flat, 1);
    let targets_idx = targets_flat.unsqueeze_dim(1);
    let selected = log_probs.gather(1, targets_idx);
    selected.mean().neg()
}