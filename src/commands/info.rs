//! Info Command
//!
//! Shows model configuration and VRAM estimates.

use crate::helpers::get_model_config;
use crate::utils::{format_bytes, format_params};

pub fn execute(model_size: &str) {
    let config = get_model_config(model_size);

    println!("═══════════════════════════════════════════════════════════");
    println!("  📊 Informações do Modelo: {}", model_size);
    println!("═══════════════════════════════════════════════════════════");
    println!("  Parâmetros: {}", format_params(config.num_parameters()));
    println!("  vocab_size: {}", config.vocab_size);
    println!("  d_model: {}", config.d_model);
    println!("  n_layers: {}", config.n_layers);
    println!("  d_ffn: {}", config.d_ffn);
    println!("  max_seq_len: {}", config.max_seq_len);
    println!();

    println!("  📦 VRAM Estimada (Treino, batch=1):");
    for seq in [256, 512, 1024] {
        let vram = config.estimated_vram(1, seq);
        let fit = if vram < 15_000_000_000 { "✅ T4" } else { "❌" };
        println!("    seq_len={}: {} {}", seq, format_bytes(vram), fit);
    }
    println!("═══════════════════════════════════════════════════════════");
}
