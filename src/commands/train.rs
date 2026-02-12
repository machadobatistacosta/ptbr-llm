use std::path::PathBuf;
use clap::Args;

use crate::backend::{TrainBackend, get_device};
use crate::data::MmapDataset;
use crate::helpers::get_model_config;
use crate::error::{PtbrError, Result};
use crate::model::{TrainingConfig, Trainer};
use crate::tokenizer::BPETokenizer;
use crate::utils::{format_bytes, format_params};

#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(long, help = "Caminho para o dataset de treino (.bin)")]
    pub data: PathBuf,

    #[arg(long, help = "Caminho para o dataset de validação (.bin) [Opcional]")]
    pub val_data: Option<PathBuf>,

    #[arg(long, help = "Caminho para o tokenizer.json")]
    pub tokenizer: PathBuf,

    #[arg(long, help = "Diretório de saída para checkpoints e metrics")]
    pub output: PathBuf,

    #[arg(long, default_value = "400m", help = "Tamanho do modelo (400m, 800m, 1b, 1.5b)")]
    pub model_size: String,

    #[arg(long, default_value_t = 100000, help = "Número máximo de steps")]
    pub max_steps: usize,

    #[arg(long, default_value_t = 1000, help = "Salvar checkpoint a cada N steps")]
    pub save_every: usize,

    #[arg(long, default_value_t = 4, help = "Batch size por dispositivo")]
    pub batch_size: usize,

    #[arg(long, default_value_t = 4, help = "Gradient accumulation steps")]
    pub grad_accum: usize,

    #[arg(long, default_value_t = 1e-4, help = "Learning rate inicial")]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 1000, help = "Steps de warmup")]
    pub warmup_steps: usize,

    #[arg(long, default_value_t = 1.0, help = "Gradient clipping (norma L2)")]
    pub gradient_clip: f64,

    #[arg(long, default_value_t = 512, help = "Tamanho da sequência (context window)")]
    pub seq_len: usize,

    #[arg(long, default_value_t = 500, help = "Avaliar a cada N steps")]
    pub eval_every: usize,

    #[arg(long, default_value_t = 100, help = "Número de amostras para avaliação")]
    pub eval_samples: usize,
    
    #[arg(long, help = "Retomar treinamento a partir deste checkpoint")]
    pub resume_from: Option<PathBuf>,
}

/// Helper para ajustar config em T4 (mantido da versão anterior)
fn get_safe_config(
    model_size: &str,
    seq_len: usize,
    grad_accum: usize,
    batch_size: usize,
) -> (usize, usize, usize) {
    match model_size {
        "400m" | "400M" => {
            let safe_batch = batch_size.min(4);
            let safe_seq = seq_len.min(512);
            if safe_batch * safe_seq > 1024 {
                let safe_batch = 2;
                let safe_seq = seq_len.min(512);
                let scaling = batch_size as f32 / safe_batch as f32;
                let new_accum = (grad_accum as f32 * scaling).ceil() as usize;
                (safe_seq, new_accum, safe_batch)
            } else {
                (safe_seq, grad_accum, safe_batch)
            }
        }
        "800m" | "800M" => {
            let safe_batch = batch_size.min(1);
            let scaling = batch_size as f32 / safe_batch as f32;
            let new_accum = (grad_accum as f32 * scaling).ceil() as usize;
            (seq_len.min(256), new_accum, safe_batch)
        }
        "1b" | "1B" => (seq_len.min(192), grad_accum.min(2), batch_size.min(1)),
        "1.5b" | "1.5B" => (seq_len.min(128), grad_accum.min(2), batch_size.min(1)),
        _ => (seq_len.min(1024), grad_accum, batch_size),
    }
}

pub fn execute(args: TrainArgs) -> Result<()> {
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let (safe_seq_len, safe_grad_accum, safe_batch) = get_safe_config(
        &args.model_size,
        args.seq_len,
        args.grad_accum,
        args.batch_size
    );

    if safe_seq_len != args.seq_len || safe_grad_accum != args.grad_accum || safe_batch != args.batch_size {
        println!("  ⚠️ Config ajustada para T4 16GB:");
        println!("     seq_len: {} -> {}", args.seq_len, safe_seq_len);
        println!("     grad_accum: {} -> {}", args.grad_accum, safe_grad_accum);
        println!("     batch_size: {} -> {}", args.batch_size, safe_batch);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  🚀 Iniciando Treinamento RWKV");
    if let Some(ref ckpt) = args.resume_from {
        println!("  🔄 Retomando de: {:?}", ckpt);
    }
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();
    let mut model_config = get_model_config(&args.model_size);
    model_config.max_seq_len = safe_seq_len;
    model_config.dropout = 0.05;

    let tokenizer = BPETokenizer::from_file(args.tokenizer.to_str().unwrap())
        .map_err(|e| PtbrError::TokenizerLoad(e.to_string()))?;

    if tokenizer.vocab_size() != model_config.vocab_size {
        println!(
            "  ⚠️ Ajustando vocab_size: {} -> {}",
            model_config.vocab_size,
            tokenizer.vocab_size()
        );
        model_config.vocab_size = tokenizer.vocab_size();
    }

    let dataset_path = if args.data.join("train.bin").exists() {
        args.data.join("train.bin")
    } else if args.data.extension().map(|e| e == "bin").unwrap_or(false) {
        args.data.clone()
    } else {
        return Err(PtbrError::FileNotFound(args.data.clone()));
    };

    let mut dataset = MmapDataset::from_file(&dataset_path, safe_seq_len)
        .map_err(|e| PtbrError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

    if dataset.is_empty() {
        return Err(PtbrError::DatasetEmpty { path: dataset_path });
    }

    // Training config
    let train_config = TrainingConfig {
        learning_rate: args.learning_rate,
        batch_size: safe_batch,
        gradient_accumulation_steps: safe_grad_accum,
        warmup_steps: args.warmup_steps,
        max_steps: args.max_steps,
        gradient_clip: args.gradient_clip,
        save_every: args.save_every,
        log_every: 1,
        min_lr_ratio: 0.1,
        weight_decay: 0.001, // Ensuring default is set
        ..Default::default()
    };

    // Initialize Trainer
    let mut trainer = match &args.resume_from {
        Some(checkpoint_path) => {
            println!("  📥 Carregando checkpoint...");
            Trainer::<TrainBackend>::from_checkpoint(
                checkpoint_path,
                &model_config,
                train_config,
                device.clone()
            )?
        },
        None => {
            println!("  🆕 Iniciando modelo do zero...");
            Trainer::<TrainBackend>::new(&model_config, train_config, device.clone())
        }
    };

    // Load validation dataset
    let val_dataset = if let Some(val_path) = args.val_data {
        let val_path_resolved = if val_path.extension().map(|e| e == "bin").unwrap_or(false) {
            val_path.clone()
        } else if val_path.join("val.bin").exists() {
            val_path.join("val.bin")
        } else {
            val_path.clone()
        };
        
        println!("  📊 Validation dataset: {:?}", val_path_resolved);
        match MmapDataset::from_file(&val_path_resolved, safe_seq_len) {
            Ok(vd) => Some(vd),
            Err(e) => {
                eprintln!("  ⚠️ Erro carregando val dataset: {}, usando 10% do train", e);
                Some(dataset.split_validation(0.1))
            }
        }
    } else {
        println!("  📊 Sem --val-data, usando 10% do train como validação");
        Some(dataset.split_validation(0.1))
    };

    println!(
        "  Modelo: {} ({} params)",
        args.model_size,
        format_params(model_config.num_parameters())
    );
    println!(
        "  VRAM estimada: {}",
        format_bytes(model_config.estimated_vram(safe_batch, safe_seq_len))
    );

    trainer.fit(
        &mut dataset,
        val_dataset.as_ref(),
        &tokenizer,
        &args.output,
        args.eval_samples,
        args.eval_every,
    )?;

    Ok(())
}
