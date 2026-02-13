//! Find LR Command
//!
//! Finds optimal learning rate using LR range test.

use std::path::PathBuf;

use crate::backend::{TrainBackend, get_device};
use crate::data::MmapDataset;
use crate::helpers::get_model_config;
use crate::model::{find_lr as find_lr_fn, LRFinderResult, RWKV, RWKV_V7};
use crate::utils::format_number;

pub fn execute(data: &PathBuf, _tokenizer_path: &PathBuf, model_size: &str, num_steps: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔍 Learning Rate Finder");
    println!("═══════════════════════════════════════════════════════════");

    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let device = get_device();
    let config = get_model_config(model_size);

    let dataset_path = if data.join("train.bin").exists() {
        data.join("train.bin")
    } else {
        data.clone()
    };

    let dataset = MmapDataset::from_file(&dataset_path, 256).expect("Erro carregando dataset");

    println!("  Modelo: {}", model_size);
    println!("  Dataset: {} tokens", format_number(dataset.num_tokens()));
    println!("  Steps: {}", num_steps);
    println!();

    let start_lr: f64 = 1e-7;
    let end_lr: f64 = 1e-1;

    println!("  LR Range: {:.2e} → {:.2e}", start_lr, end_lr);
    println!();

    // Dispatch based on version
    let result: LRFinderResult = match config.rwkv_version {
        7 => {
            let model = RWKV_V7::new(&config, &device);
            find_lr_fn::<TrainBackend, _>(
                model, &config, &dataset, &device, start_lr, end_lr, num_steps,
            )
        }
        _ => {
            let model = RWKV::new(&config, &device);
            find_lr_fn::<TrainBackend, _>(
                model, &config, &dataset, &device, start_lr, end_lr, num_steps,
            )
        }
    };

    // Log durante execução
    for (i, (lr, loss)) in result.lrs.iter().zip(result.losses.iter()).enumerate() {
        if i % 10 == 0 {
            println!(
                "  Step {:>4} | LR: {:.2e} | Loss: {:.4}",
                i, lr, loss
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  📊 Resultado:");
    println!("  LR sugerido: {:.2e}", result.suggested_lr);
    println!("  (Use ~10x menor para estabilidade: {:.2e})", result.suggested_lr / 10.0);
    println!("═══════════════════════════════════════════════════════════");
}

