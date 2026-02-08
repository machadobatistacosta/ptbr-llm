//! Benchmark Command
//!
//! Benchmarks model performance.

use std::time::Instant;
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::backend::{MyBackend, get_device};
use crate::helpers::get_model_config;
use crate::model::RWKV;
use crate::utils::format_params;

pub fn execute(model_size: &str, seq_len: usize, num_iterations: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ⚡ Benchmark de Performance");
    println!("═══════════════════════════════════════════════════════════");

    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let device = get_device();
    let config = get_model_config(model_size);

    println!("  Modelo: {} ({} params)", model_size, format_params(config.num_parameters()));
    println!("  Seq len: {}", seq_len);
    println!("  Iterações: {}", num_iterations);
    println!();

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);

    // Warmup
    println!("  Warmup...");
    for _ in 0..3 {
        let dummy: Tensor<MyBackend, 2, Int> = Tensor::zeros([1, seq_len], &device);
        let _ = model.forward(dummy);
    }

    // Forward benchmark
    println!("  Benchmarking forward...");
    let mut forward_times = Vec::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        let dummy: Tensor<MyBackend, 2, Int> = Tensor::zeros([1, seq_len], &device);
        let start = Instant::now();
        let _ = model.forward(dummy);
        forward_times.push(start.elapsed().as_secs_f64());
    }

    // Inference benchmark
    println!("  Benchmarking inference...");
    let mut inference_times = Vec::with_capacity(num_iterations);
    for _ in 0..num_iterations {
        let dummy: Tensor<MyBackend, 2, Int> = Tensor::zeros([1, 32], &device);
        let start = Instant::now();
        let _ = model.forward_inference(dummy);
        inference_times.push(start.elapsed().as_secs_f64());
    }

    // Stats
    let avg_forward = forward_times.iter().sum::<f64>() / forward_times.len() as f64;
    let avg_inference = inference_times.iter().sum::<f64>() / inference_times.len() as f64;
    let tokens_per_sec = seq_len as f64 / avg_forward;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  📊 Resultados:");
    println!("  Forward (seq={}): {:.2}ms avg", seq_len, avg_forward * 1000.0);
    println!("  Inference (seq=32): {:.2}ms avg", avg_inference * 1000.0);
    println!("  Throughput: {:.1} tokens/s", tokens_per_sec);
    println!("═══════════════════════════════════════════════════════════");
}
