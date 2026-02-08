//! TestGpu Command
//!
//! Tests if GPU supports the model configuration.

use std::time::Instant;
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::backend::{MyBackend, get_device};
use crate::helpers::get_model_config;
use crate::model::RWKV;
use crate::utils::{format_bytes, format_params};

pub fn execute(model_size: &str, seq_len: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🧪 Testando GPU para modelo {}", model_size);
    println!("═══════════════════════════════════════════════════════════");

    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let device = get_device();
    let config = get_model_config(model_size);

    println!("  Parâmetros: {}", format_params(config.num_parameters()));
    println!(
        "  VRAM estimada: {}",
        format_bytes(config.estimated_vram(1, seq_len))
    );
    println!();

    println!("  [1/3] Criando modelo...");
    let start = Instant::now();
    let model: RWKV<MyBackend> = RWKV::new(&config, &device);
    println!(
        "        ✓ Modelo criado em {:.2}s",
        start.elapsed().as_secs_f64()
    );

    println!("  [2/3] Testando forward...");
    let dummy: Tensor<MyBackend, 2, Int> = Tensor::zeros([1, seq_len], &device);
    let start = Instant::now();
    let logits = model.forward(dummy);
    let [b, s, v] = logits.dims();
    println!(
        "        ✓ Forward OK: [{}, {}, {}] em {:.2}s",
        b,
        s,
        v,
        start.elapsed().as_secs_f64()
    );

    println!("  [3/3] Testando inferência...");
    let dummy2: Tensor<MyBackend, 2, Int> = Tensor::zeros([1, 32], &device);
    let start = Instant::now();
    let _ = model.forward_inference(dummy2);
    println!(
        "        ✓ Inferência OK em {:.2}s",
        start.elapsed().as_secs_f64()
    );

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  🎉 GPU suporta {} com seq_len={}!", model_size, seq_len);
    println!("═══════════════════════════════════════════════════════════");
}
