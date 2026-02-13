//! Common Helper Functions
//!
//! Shared utilities used across multiple commands.

use burn::tensor::{backend::Backend, Int, Tensor};
use crate::model::RWKVConfig;

/// Returns model configuration based on size string
pub fn get_model_config(model_size: &str) -> RWKVConfig {
    match model_size {
        // RWKV-4
        "140m" | "140M" | "85m" | "85M" => RWKVConfig::ptbr_140m(),
        "400m" | "400M" => RWKVConfig::ptbr_400m(),
        "800m" | "800M" => RWKVConfig::ptbr_800m(),
        "1b" | "1B" => RWKVConfig::ptbr_1b(),
        "1.5b" | "1.5B" => RWKVConfig::ptbr_1_5b(),
        // RWKV-7
        "140m-v7" | "140M-v7" => RWKVConfig::ptbr_140m_v7(),
        "400m-v7" | "400M-v7" => RWKVConfig::ptbr_400m_v7(),
        _ => {
            println!("  ⚠️ Tamanho '{}' não reconhecido, usando 140m", model_size);
            RWKVConfig::ptbr_140m()
        }
    }
}

/// Creates a batch tensor from token data
pub fn create_batch_tensor<B: Backend>(data: &[Vec<u16>], device: &B::Device) -> Tensor<B, 2, Int> {
    let batch_size = data.len();
    let seq_len = data[0].len();

    // Burn CUDA & WGPU backends typically use i32 for Int tensors
    let flat: Vec<i32> = data.iter().flatten().map(|&x| x as i32).collect();

    let tensor_data = burn::tensor::TensorData::from(flat.as_slice());
    let tensor: Tensor<B, 1, Int> = Tensor::from_data(tensor_data, device);
    
    tensor.reshape([batch_size, seq_len])
}

/// Computes softmax over logits
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}
