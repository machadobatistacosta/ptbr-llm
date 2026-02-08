//! Resume Command
//!
//! Resumes training from a checkpoint.

use std::path::PathBuf;

use crate::backend::{TrainBackend, get_device};
use crate::data::MmapDataset;
use crate::error::{PtbrLlmError, Result};
use crate::helpers::get_model_config;
use crate::model::{TrainingConfig, Trainer};
use crate::tokenizer::{BPETokenizer, BPEVocab};

use super::train::{get_safe_config, run_training_loop};

pub fn execute(
    checkpoint: &PathBuf,
    data: &PathBuf,
    val_data: Option<&PathBuf>,
    output: &PathBuf,
    model_size: &str,
    additional_steps: usize,
    save_every: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    seq_len: usize,
    eval_every: usize,
    eval_samples: usize,
    gradient_clip: f64,
) -> Result<()> {
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    println!("═══════════════════════════════════════════════════════════");
    println!("  🔄 Retomando Treinamento");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Checkpoint: {:?}", checkpoint);

    let device = get_device();

    let (safe_seq_len, safe_grad_accum, safe_batch) = get_safe_config(model_size, seq_len, grad_accum, batch_size);

    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = safe_seq_len;
    // Bug #5 fix: Removed hardcoded d_model/d_ffn/n_layers/vocab_size that ignored model_size CLI arg

    let train_config = TrainingConfig {
        learning_rate,
        batch_size: safe_batch,
        gradient_accumulation_steps: safe_grad_accum,
        // Bug #3 fix: Force warmup=200 after resume (optimizer state is not persisted)
        warmup_steps: 200,
        max_steps: additional_steps,
        // Bug #10: Use default 0.001 instead of hardcoded 0.01
        gradient_clip,
        save_every,
        log_every: 10,
        min_lr_ratio: 0.1,
        ..Default::default()
    };


    let mut trainer: Trainer<TrainBackend> =
        Trainer::new(&model_config, train_config, device.clone());
    trainer
        .load_checkpoint(checkpoint.to_str().unwrap())
        .map_err(|e| PtbrLlmError::CheckpointLoad(e.to_string()))?;

    let dataset_path = if data.join("train.bin").exists() {
        data.join("train.bin")
    } else if data.extension().map(|e| e == "bin").unwrap_or(false) {
        data.clone()
    } else {
        return Err(PtbrLlmError::FileNotFound(data.clone()));
    };

    let mut dataset = MmapDataset::from_file(&dataset_path, safe_seq_len)
        .map_err(|e| PtbrLlmError::DatasetCorrupted(e.to_string()))?;
    dataset.shuffle(42 + trainer.step() as u64);

    std::fs::create_dir_all(output)?;

    let mut metrics_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("metrics.csv"))?;

    let tokenizer_path = data
        .parent()
        .unwrap_or(data)
        .join("tokenizer.json");
    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .unwrap_or_else(|_| BPETokenizer::from_vocab(BPEVocab::new()));

    println!("  Continuando do step {}...", trainer.step());
    println!();

    // Bug #7 fix: Reserve 10% of data for validation
    dataset.reserve_validation(0.1);

    run_training_loop(
        &mut trainer,
        &mut dataset,
        &tokenizer,
        additional_steps,
        save_every,
        eval_every,
        eval_samples,
        safe_batch,
        output,
        &mut metrics_file,
        &device,
    );

    Ok(())
}
