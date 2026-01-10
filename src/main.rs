#![allow(dead_code)]
#![allow(unused_imports)]

mod data;
mod model;
mod tokenizer;
mod logger;
mod utils;

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use std::collections::HashSet;
use rayon::prelude::*;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::{backend::Backend, ElementConversion, Int, Tensor};

// Utils
use utils::{format_bytes, format_duration, format_number, format_params};
use tokenizer::BPEVocab;

// ============ BACKEND SELECTOR ============
#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu")))]
mod backend_impl {
    pub use burn::backend::cuda_jit::{Cuda, CudaDevice};
    pub type MyBackend = Cuda;
    
    pub fn get_device() -> CudaDevice {
        CudaDevice::new(0)
    }
}

#[cfg(all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu")))]
mod backend_impl {
    pub use burn::backend::wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
    pub type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    
    pub fn get_device() -> WgpuDevice {
        WgpuDevice::BestAvailable
    }
}

#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu")))]
mod backend_impl {
    pub use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type MyBackend = NdArray;
    
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

#[cfg(not(any(
    all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu")),
    all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu")),
    all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu"))
)))]
mod backend_impl {
    pub use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type MyBackend = NdArray;
    
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }
}

use backend_impl::{MyBackend, get_device};
type TrainBackend = Autodiff<MyBackend>;

// ============ IMPORTS DO PROJETO ============
use data::{DataLoader, MmapDataset, TokenizedDatasetWriter, WikiCleaner, WikiStreamParser};
use model::{RWKVConfig, Trainer, TrainingConfig, RWKV};
use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};

use rand::distributions::WeightedIndex;
use rand::prelude::*;

// ============ CLI ============
#[derive(Parser)]
#[command(name = "ptbr-slm")]
#[command(author = "Caike Costa")]
#[command(version = "1.1.0")]
#[command(about = "RWKV Language Model para Português Brasileiro")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Processa dump Wikipedia XML.BZ2
    ProcessWiki {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "200")]
        min_chars: usize,
    },

    /// Treina tokenizer BPE
    TrainTokenizer {
        #[arg(short, long)]
        corpus: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "32000")]
        vocab_size: usize,
    },

    /// Tokeniza corpus para binário
    Tokenize {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
    },

    /// Treina modelo RWKV
    Train {
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "85m")]
        model_size: String,
        #[arg(long, default_value = "50000")]
        max_steps: usize,
        #[arg(long, default_value = "1000")]
        save_every: usize,
        #[arg(long, default_value = "1")]
        batch_size: usize,
        #[arg(long, default_value = "16")]
        grad_accum: usize,
        #[arg(long, default_value = "3e-4")]
        learning_rate: f64,
        #[arg(long, default_value = "500")]
        warmup_steps: usize,
        #[arg(long, default_value = "1024")]
        seq_len: usize,
        #[arg(long, default_value = "500")]
        eval_every: usize,
        #[arg(long, default_value = "100")]
        eval_samples: usize,
    },

    /// Retoma treino de checkpoint
    Resume {
        #[arg(short, long)]
        checkpoint: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "85m")]
        model_size: String,
        #[arg(long, default_value = "50000")]
        additional_steps: usize,
        #[arg(long, default_value = "1000")]
        save_every: usize,
        #[arg(long, default_value = "1")]
        batch_size: usize,
        #[arg(long, default_value = "16")]
        grad_accum: usize,
        #[arg(long, default_value = "1e-4")]
        learning_rate: f64,
        #[arg(long, default_value = "1024")]
        seq_len: usize,
    },

    /// Testa modelo com prompts
    TestModel {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "85m")]
        model_size: String,
    },

    /// Gera texto a partir de prompt
    Generate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        #[arg(long, default_value = "85m")]
        model_size: String,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "40")]
        top_k: usize,
    },

    /// Limpa corpus de texto
    CleanCorpus {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        verbose: bool,
    },

    /// Audita qualidade do corpus
    AuditCorpus {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Mostra informações do modelo
    Info {
        #[arg(long, default_value = "85m")]
        model_size: String,
    },

    /// Constrói dataset tokenizado
    BuildDataset {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, required = true)]
        source: Vec<String>,
        #[arg(long, default_value = "100")]
        min_chars: usize,
        #[arg(long, default_value = "true")]
        blocks: bool,
        #[arg(long, default_value = "false")]
        clean: bool,
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Testa se GPU suporta o modelo
    TestGpu {
        #[arg(long, default_value = "400m")]
        model_size: String,
        #[arg(long, default_value = "512")]
        seq_len: usize,
    },

    /// Encontra learning rate ótimo
    FindLr {
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "85m")]
        model_size: String,
        #[arg(long, default_value = "100")]
        num_steps: usize,
    },

    /// Benchmark de performance
    Benchmark {
        #[arg(long, default_value = "85m")]
        model_size: String,
        #[arg(long, default_value = "512")]
        seq_len: usize,
        #[arg(long, default_value = "10")]
        num_iterations: usize,
    },
}

// ============ MAIN ============
fn main() {
    // Inicializa logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::ProcessWiki {
            input,
            output,
            min_chars,
        } => process_wiki(&input, &output, min_chars),

        Commands::TrainTokenizer {
            corpus,
            output,
            vocab_size,
        } => train_tokenizer(&corpus, &output, vocab_size),

        Commands::Tokenize {
            input,
            output,
            tokenizer,
        } => tokenize_corpus(&input, &output, &tokenizer),

        Commands::Train {
            data,
            tokenizer,
            output,
            model_size,
            max_steps,
            save_every,
            batch_size,
            grad_accum,
            learning_rate,
            warmup_steps,
            seq_len,
            eval_every,
            eval_samples,
        } => train_model(
            &data,
            &tokenizer,
            &output,
            &model_size,
            max_steps,
            save_every,
            batch_size,
            grad_accum,
            learning_rate,
            warmup_steps,
            seq_len,
            eval_every,
            eval_samples,
        ),

        Commands::Resume {
            checkpoint,
            data,
            output,
            model_size,
            additional_steps,
            save_every,
            batch_size,
            grad_accum,
            learning_rate,
            seq_len,
        } => resume_training(
            &checkpoint,
            &data,
            &output,
            &model_size,
            additional_steps,
            save_every,
            batch_size,
            grad_accum,
            learning_rate,
            seq_len,
        ),

        Commands::TestModel {
            model,
            tokenizer,
            model_size,
        } => test_model(&model, &tokenizer, &model_size),

        Commands::Generate {
            model,
            tokenizer,
            prompt,
            max_tokens,
            model_size,
            temperature,
            top_k,
        } => generate(
            &model,
            &tokenizer,
            &prompt,
            max_tokens,
            &model_size,
            temperature,
            top_k,
        ),

        Commands::AuditCorpus { input, output } => audit_corpus_cmd(&input, &output),

        Commands::CleanCorpus {
            input,
            output,
            verbose,
        } => clean_corpus(&input, &output, verbose),

        Commands::Info { model_size } => show_info(&model_size),

        Commands::BuildDataset {
            tokenizer,
            output,
            source,
            min_chars,
            blocks,
            clean,
            seed,
        } => build_dataset(&tokenizer, &output, &source, min_chars, blocks, clean, seed),

        Commands::TestGpu { model_size, seq_len } => test_gpu(&model_size, seq_len),

        Commands::FindLr {
            data,
            tokenizer,
            model_size,
            num_steps,
        } => find_lr(&data, &tokenizer, &model_size, num_steps),

        Commands::Benchmark {
            model_size,
            seq_len,
            num_iterations,
        } => benchmark(&model_size, seq_len, num_iterations),
    }
}

// ============ TEST GPU ============
fn test_gpu(model_size: &str, seq_len: usize) {
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

// ============ BENCHMARK ============
fn benchmark(model_size: &str, seq_len: usize, num_iterations: usize) {
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

// ============ FIND LR ============
fn find_lr(data: &PathBuf, tokenizer_path: &PathBuf, model_size: &str, num_steps: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔍 Learning Rate Finder");
    println!("═══════════════════════════════════════════════════════════");

    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let device = get_device();
    let config = get_model_config(model_size);

    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

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

    let mut trainer: Trainer<TrainBackend> = Trainer::new(&config, train_config, device.clone());

    let loader = DataLoader::new(&dataset, 1);

    println!("  LR Range: {:.2e} → {:.2e}", start_lr, end_lr);
    println!();

    for (i, (inputs, targets)) in loader.into_iter().enumerate() {
        if i >= num_steps {
            break;
        }

        let input_tensor = create_batch_tensor::<TrainBackend>(&inputs, &device);
        let target_tensor = create_batch_tensor::<TrainBackend>(&targets, &device);

        if let Some(stats) = trainer.train_step(input_tensor, target_tensor) {
            lrs.push(current_lr);
            losses.push(stats.loss);

            if i % 10 == 0 {
                println!(
                    "  Step {:>4} | LR: {:.2e} | Loss: {:.4}",
                    i, current_lr, stats.loss
                );
            }

            // Early stop se loss explodiu
            if stats.loss > losses.first().unwrap_or(&10.0) * 10.0 || !stats.loss.is_finite() {
                println!("  ⚠️ Loss explodiu, parando...");
                break;
            }

            current_lr *= lr_mult;
        }
    }

    // Encontra LR sugerido (menor gradiente)
    let suggested = find_suggested_lr(&lrs, &losses);

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  📊 Resultado:");
    println!("  LR sugerido: {:.2e}", suggested);
    println!("  (Use ~10x menor para estabilidade: {:.2e})", suggested / 10.0);
    println!("═══════════════════════════════════════════════════════════");
}

fn find_suggested_lr(lrs: &[f64], losses: &[f32]) -> f64 {
    if losses.len() < 5 {
        return lrs.get(lrs.len() / 2).copied().unwrap_or(3e-4);
    }

    // Encontra ponto de maior descida
    let mut min_grad = f32::MAX;
    let mut best_idx = 0;

    for i in 2..losses.len() - 2 {
        let grad = (losses[i + 1] - losses[i - 1]) / 2.0;
        if grad < min_grad && losses[i].is_finite() {
            min_grad = grad;
            best_idx = i;
        }
    }

    lrs[best_idx]
}

// ============ PROCESS WIKI ============
fn process_wiki(input: &PathBuf, output: &PathBuf, min_chars: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  📚 Processando Wikipedia PT-BR");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Input: {:?}", input);
    println!("  Output: {:?}", output);
    println!("  Min chars: {}", min_chars);
    println!();

    let parser = WikiStreamParser::new(min_chars);
    let cleaner = WikiCleaner::new();
    let normalizer = PTBRNormalizer::new();

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let mut file_idx = 0;
    let mut current_file = create_output_file(output, file_idx);
    let mut articles_in_file = 0;
    let mut total_articles = 0;
    let mut total_chars = 0usize;

    let start = Instant::now();

    for article in parser.parse_streaming(input.to_str().unwrap()) {
        let normalized = normalizer.normalize(&article.text);
        let clean_text = cleaner.clean(&normalized);

        if clean_text.len() >= min_chars {
            writeln!(current_file, "{}", clean_text).expect("Erro escrevendo");
            writeln!(current_file).expect("Erro escrevendo newline");

            articles_in_file += 1;
            total_articles += 1;
            total_chars += clean_text.len();

            if articles_in_file >= 10_000 {
                file_idx += 1;
                current_file = create_output_file(output, file_idx);
                articles_in_file = 0;
                println!(
                    "  ✓ Arquivo {} completado ({} artigos)",
                    file_idx, total_articles
                );
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Processamento concluído!");
    println!("  Arquivos: {}", file_idx + 1);
    println!("  Artigos: {}", format_number(total_articles));
    println!("  Caracteres: {}", format_bytes(total_chars));
    println!("  Tempo: {:.1}s", elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════");
}

fn create_output_file(output: &PathBuf, idx: usize) -> std::fs::File {
    std::fs::File::create(output.join(format!("wiki_{:04}.txt", idx))).expect("Erro criando arquivo")
}

// ============ TRAIN TOKENIZER ============
fn train_tokenizer(corpus: &PathBuf, output: &PathBuf, vocab_size: usize) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔤 Treinando Tokenizer BPE");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Corpus: {:?}", corpus);
    println!("  Vocab size: {}", vocab_size);
    println!();

    let trainer = BPETrainer::new(vocab_size, 5);

    let texts: Vec<String> = if corpus.is_file() {
        println!("  Lendo arquivo único...");
        let content = std::fs::read_to_string(corpus).expect("Erro lendo arquivo");
        content.lines().map(String::from).collect()
    } else {
        println!("  Lendo diretório...");
        let mut all_texts = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(corpus)
            .expect("Erro lendo diretório")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "txt").unwrap_or(false))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            println!("    Lendo {:?}...", entry.file_name());
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                all_texts.extend(content.lines().map(String::from));
            }
        }
        all_texts
    };

    println!("  Total de linhas: {}", format_number(texts.len()));
    println!();

    let vocab = trainer.train(texts.into_iter());
    let tokenizer = BPETokenizer::from_vocab(vocab);

    std::fs::create_dir_all(output).expect("Erro criando diretório");
    let tokenizer_path = output.join("tokenizer.json");
    tokenizer
        .save(tokenizer_path.to_str().unwrap())
        .expect("Erro salvando");

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Tokenizer salvo em {:?}", tokenizer_path);
    println!("  Vocab size: {}", tokenizer.vocab_size());
    println!("═══════════════════════════════════════════════════════════");
}

// ============ TOKENIZE CORPUS ============
fn tokenize_corpus(input: &PathBuf, output: &PathBuf, tokenizer_path: &PathBuf) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔢 Tokenizando Corpus");
    println!("═══════════════════════════════════════════════════════════");

    let tokenizer =
        BPETokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Erro carregando tokenizer");

    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    println!("  BOS ID: {}", bos);
    println!("  EOS ID: {}", eos);
    println!("  Vocab: {}", tokenizer.vocab_size());
    println!();

    let normalizer = PTBRNormalizer::new();
    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let mut writer =
        TokenizedDatasetWriter::new(&output.join("train.bin")).expect("Erro criando arquivo");

    let mut total_tokens = 0usize;
    let mut total_docs = 0usize;

    let files = collect_text_files(input);

    for path in &files {
        println!(
            "  Processando {:?}...",
            path.file_name().unwrap_or_default()
        );

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                let bytes = std::fs::read(path).unwrap_or_default();
                String::from_utf8_lossy(&bytes).to_string()
            }
        };

        for doc in content.split("\n\n") {
            let doc = doc.trim();
            if doc.len() < 100 {
                continue;
            }

            let normalized = normalizer.normalize(doc);
            let mut tokens = vec![bos];
            tokens.extend(tokenizer.encode(&normalized));
            tokens.push(eos);

            writer.write_tokens(&tokens).expect("Erro escrevendo");
            total_tokens += tokens.len();
            total_docs += 1;
        }
    }

    let _final_count = writer.finish().expect("Erro finalizando");

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Tokenização concluída!");
    println!("  Documentos: {}", format_number(total_docs));
    println!("  Tokens: {}", format_number(total_tokens));
    println!("  Arquivo: {:?}", output.join("train.bin"));
    println!("═══════════════════════════════════════════════════════════");
}

fn collect_text_files(path: &PathBuf) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.clone()];
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .expect("Erro lendo diretório")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    files.sort();
    files
}

// ============ TRAIN MODEL ============
fn train_model(
    data: &PathBuf,
    tokenizer_path: &PathBuf,
    output: &PathBuf,
    model_size: &str,
    max_steps: usize,
    save_every: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    warmup_steps: usize,
    seq_len: usize,
    eval_every: usize,
    eval_samples: usize,
) {
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let (safe_seq_len, safe_grad_accum) = get_safe_config(model_size, seq_len, grad_accum);

    if safe_seq_len != seq_len || safe_grad_accum != grad_accum {
        println!("  ⚠️ Config ajustada para T4 16GB:");
        println!("     seq_len: {} -> {}", seq_len, safe_seq_len);
        println!("     grad_accum: {} -> {}", grad_accum, safe_grad_accum);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  🚀 Iniciando Treinamento RWKV");
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();

    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = safe_seq_len;
    model_config.dropout = 0.05;

    let tokenizer =
        BPETokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Erro carregando tokenizer");

    if tokenizer.vocab_size() != model_config.vocab_size {
        println!(
            "  ⚠️ Ajustando vocab_size: {} -> {}",
            model_config.vocab_size,
            tokenizer.vocab_size()
        );
        model_config.vocab_size = tokenizer.vocab_size();
    }

    let effective_batch = batch_size * safe_grad_accum;

    println!(
        "  Modelo: {} ({} params)",
        model_size,
        format_params(model_config.num_parameters())
    );
    println!(
        "  VRAM estimada: {}",
        format_bytes(model_config.estimated_vram(batch_size, safe_seq_len))
    );
    println!(
        "  Batch size: {} x {} = {} efetivo",
        batch_size, safe_grad_accum, effective_batch
    );
    println!("  Seq len: {}", safe_seq_len);
    println!("  Learning rate: {:.2e}", learning_rate);
    println!("  Warmup: {} steps", warmup_steps);
    println!("  Max steps: {}", format_number(max_steps));
    println!("  Save every: {} steps", save_every);
    println!("  Eval every: {} steps", eval_every);
    println!();

    let dataset_path = if data.join("train.bin").exists() {
        data.join("train.bin")
    } else if data.extension().map(|e| e == "bin").unwrap_or(false) {
        data.clone()
    } else {
        panic!("❌ Dataset não encontrado em {:?}", data);
    };

    let mut dataset =
        MmapDataset::from_file(&dataset_path, safe_seq_len).expect("Erro carregando dataset");

    if dataset.is_empty() {
        panic!(
            "❌ Dataset vazio! Verifique seq_len ({}) e o arquivo",
            safe_seq_len
        );
    }

    dataset.shuffle(42);
    println!(
        "  Dataset: {} tokens, {} sequências",
        format_number(dataset.num_tokens()),
        format_number(dataset.len())
    );
    println!();

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let mut metrics_file =
        std::fs::File::create(output.join("metrics.csv")).expect("Erro criando metrics.csv");
    writeln!(
        metrics_file,
        "step,loss,ppl,lr,grad_norm,tokens_per_sec,eval_loss,eval_ppl"
    )
    .expect("Erro escrevendo header");

    let train_config = TrainingConfig {
        learning_rate,
        batch_size,
        gradient_accumulation_steps: safe_grad_accum,
        warmup_steps,
        max_steps,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every,
        log_every: 10,
        min_lr_ratio: 0.1,
        ..Default::default()
    };

    let mut trainer: Trainer<TrainBackend> =
        Trainer::new(&model_config, train_config, device.clone());

    println!("═══════════════════════════════════════════════════════════");
    println!("  Iniciando loop de treino...");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    run_training_loop(
        &mut trainer,
        &mut dataset,
        &tokenizer,
        max_steps,
        save_every,
        eval_every,
        eval_samples,
        batch_size,
        output,
        &mut metrics_file,
        &device,
    );
}

fn get_safe_config(model_size: &str, seq_len: usize, grad_accum: usize) -> (usize, usize) {
    match model_size {
        "800m" | "800M" => (seq_len.min(256), grad_accum.min(4)),
        "400m" | "400M" => (seq_len.min(512), grad_accum.min(8)),
        "1b" | "1B" => (seq_len.min(192), grad_accum.min(2)),
        "1.5b" | "1.5B" => (seq_len.min(128), grad_accum.min(2)),
        _ => (seq_len.min(1024), grad_accum),
    }
}

// ============ RESUME TRAINING ============
fn resume_training(
    checkpoint: &PathBuf,
    data: &PathBuf,
    output: &PathBuf,
    model_size: &str,
    additional_steps: usize,
    save_every: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    seq_len: usize,
) {
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    println!("═══════════════════════════════════════════════════════════");
    println!("  🔄 Retomando Treinamento");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Checkpoint: {:?}", checkpoint);

    let device = get_device();

    let (safe_seq_len, safe_grad_accum) = get_safe_config(model_size, seq_len, grad_accum);

    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = safe_seq_len;
    model_config.dropout = 0.05;

    let train_config = TrainingConfig {
        learning_rate,
        batch_size,
        gradient_accumulation_steps: safe_grad_accum,
        warmup_steps: 100,
        max_steps: additional_steps,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every,
        log_every: 10,
        min_lr_ratio: 0.1,
        ..Default::default()
    };

    let mut trainer: Trainer<TrainBackend> =
        Trainer::new(&model_config, train_config, device.clone());
    trainer
        .load_checkpoint(checkpoint.to_str().unwrap())
        .expect("Erro carregando checkpoint");

    let dataset_path = if data.join("train.bin").exists() {
        data.join("train.bin")
    } else {
        data.clone()
    };

    let mut dataset =
        MmapDataset::from_file(&dataset_path, safe_seq_len).expect("Erro carregando dataset");
    dataset.shuffle(42 + trainer.step() as u64);

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let mut metrics_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("metrics.csv"))
        .expect("Erro abrindo metrics.csv");

    let tokenizer_path = data
        .parent()
        .unwrap_or(data)
        .join("tokenizer.json");
    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .unwrap_or_else(|_| BPETokenizer::from_vocab(BPEVocab::new()));

    println!("  Continuando do step {}...", trainer.step());
    println!();

    run_training_loop(
        &mut trainer,
        &mut dataset,
        &tokenizer,
        additional_steps,
        save_every,
        500,
        100,
        batch_size,
        output,
        &mut metrics_file,
        &device,
    );
}

// ============ TRAINING LOOP ============
fn run_training_loop(
    trainer: &mut Trainer<TrainBackend>,
    dataset: &mut MmapDataset,
    tokenizer: &BPETokenizer,
    max_steps: usize,
    save_every: usize,
    eval_every: usize,
    eval_samples: usize,
    batch_size: usize,
    output: &PathBuf,
    metrics_file: &mut std::fs::File,
    device: &<MyBackend as Backend>::Device,
) {
    let start = Instant::now();
    let initial_step = trainer.step();

    let mut last_log = Instant::now();
    let mut tokens_since_log = 0usize;
    let mut epoch = 0;

    let sample_prompts = ["O Brasil é", "A Constituição Federal", "Em 2024"];

    'training: loop {
        dataset.shuffle(42 + epoch);
        let loader = DataLoader::new(dataset, batch_size);

        for (inputs, targets) in loader {
            let seq_len = inputs[0].len();
            let input_tensor = create_batch_tensor::<TrainBackend>(&inputs, device);
            let target_tensor = create_batch_tensor::<TrainBackend>(&targets, device);

            if let Some(stats) = trainer.train_step(input_tensor, target_tensor) {
                let step = trainer.step();
                let steps_done = step - initial_step;
                tokens_since_log +=
                    batch_size * seq_len * trainer.config().gradient_accumulation_steps;

                // Log periódico
                if last_log.elapsed().as_secs() >= 5 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let steps_per_sec = steps_done as f64 / elapsed;
                    let tokens_per_sec = tokens_since_log as f64 / last_log.elapsed().as_secs_f64();
                    let remaining = max_steps.saturating_sub(steps_done);
                    let eta_secs = remaining as f64 / steps_per_sec.max(0.01);
                    let ppl = (stats.loss as f64).exp();

                    println!(
                        "  Step {:>6} | Loss: {:.4} | PPL: {:>7.2} | LR: {:.2e} | Grad: {:.3} | {:.1}K tok/s | ETA: {}",
                        step,
                        stats.loss,
                        ppl,
                        stats.lr,
                        stats.grad_norm,
                        tokens_per_sec / 1000.0,
                        format_duration(eta_secs as u64)
                    );

                    writeln!(
                        metrics_file,
                        "{},{:.6},{:.2},{:.2e},{:.4},{:.1},,",
                        step, stats.loss, ppl, stats.lr, stats.grad_norm, tokens_per_sec
                    )
                    .ok();

                    last_log = Instant::now();
                    tokens_since_log = 0;
                }

                // Evaluation
                if step % eval_every == 0 && step > 0 {
                    let eval_loss = evaluate_model(trainer, dataset, eval_samples, device);
                    let eval_ppl = (eval_loss as f64).exp();
                    println!(
                        "  📊 Eval Step {} | Loss: {:.4} | PPL: {:.2}",
                        step, eval_loss, eval_ppl
                    );

                    writeln!(metrics_file, "{},,,,,,{:.6},{:.2}", step, eval_loss, eval_ppl).ok();

                    // Gera samples
                    if tokenizer.vocab_size() > 256 {
                        println!("  📝 Samples:");
                        for prompt in &sample_prompts {
                            let sample = generate_sample(trainer, tokenizer, prompt, 30, device);
                            println!("     \"{}\" → {}", prompt, sample.trim());
                        }
                    }
                    println!();
                }

                // Save checkpoint
                if step % save_every == 0 && step > 0 {
                    let ckpt_path = output.join(format!("checkpoint_{}", step));
                    match trainer.save_checkpoint(ckpt_path.to_str().unwrap()) {
                        Ok(_) => println!("  💾 Checkpoint salvo: {:?}", ckpt_path),
                        Err(e) => println!("  ⚠️ Erro salvando: {}", e),
                    }
                }

                // Check completion
                if steps_done >= max_steps {
                    break 'training;
                }
            }
        }

        epoch += 1;
        dataset.next_epoch();
        println!("  📚 Epoch {} completa", epoch);
    }

    // Salva modelo final
    let final_path = output.join(format!("model_final_step_{}", trainer.step()));
    trainer
        .save_checkpoint(final_path.to_str().unwrap())
        .expect("Erro salvando modelo final");

    let elapsed = start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Treinamento concluído!");
    println!("  Steps: {}", trainer.step());
    println!("  Tempo: {}", format_duration(elapsed.as_secs()));
    println!("  EMA Loss: {:.4}", trainer.ema_loss());
    println!("  Modelo: {:?}", final_path);
    println!("═══════════════════════════════════════════════════════════");
}

fn evaluate_model(
    trainer: &Trainer<TrainBackend>,
    dataset: &MmapDataset,
    num_samples: usize,
    device: &<MyBackend as Backend>::Device,
) -> f32 {
    let mut total_loss = 0.0f64;
    let mut count = 0;

    let start_idx = dataset.len().saturating_sub(num_samples);

    for idx in start_idx..dataset.len() {
        if let Some((input, target)) = dataset.get(idx) {
            let input_tensor = create_batch_tensor::<TrainBackend>(&[input], device);
            let target_tensor = create_batch_tensor::<TrainBackend>(&[target], device);

            let logits = trainer.model.forward(input_tensor);
            let loss = compute_loss::<TrainBackend>(logits, target_tensor);

            total_loss += loss as f64;
            count += 1;
        }
    }

    (total_loss / count.max(1) as f64) as f32
}

fn compute_loss<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> f32 {
    use burn::tensor::activation;
    use burn::tensor::ElementConversion;  // ← Adicionar se não tiver no topo
    
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
    let targets_flat = targets.reshape([batch_size * seq_len]);
    
    let log_probs = activation::log_softmax(logits_flat, 1);
    let targets_idx = targets_flat.unsqueeze_dim(1);
    let selected = log_probs.gather(1, targets_idx);
    
    // Corrigido: usa elem() do trait ElementConversion
    let loss_scalar = selected.mean().neg().into_scalar();
    loss_scalar.elem::<f32>()
}

fn generate_sample(
    trainer: &Trainer<TrainBackend>,
    tokenizer: &BPETokenizer,
    prompt: &str,
    max_tokens: usize,
    device: &<MyBackend as Backend>::Device,
) -> String {
    let mut tokens = tokenizer.encode(prompt);

    for _ in 0..max_tokens {
        if tokens.len() > 512 {
            tokens = tokens[tokens.len() - 512..].to_vec();
        }

        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        let input: Tensor<TrainBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), device);
        let input = input.reshape([1, seq_len]);

        let logits = trainer.model.forward(input);
        let [_, s, v] = logits.dims();
        let last_logits = logits.slice([0..1, s - 1..s, 0..v]).reshape([v]);

        let logits_data: Vec<f32> = last_logits.into_data().iter::<f32>().collect();
        let next_token = logits_data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u16)
            .unwrap_or(0);

        if next_token == tokenizer.eos_id() {
            break;
        }

        tokens.push(next_token);
    }

    tokenizer.decode(&tokens)
}

// ============ TEST MODEL ============
fn test_model(model_path: &PathBuf, tokenizer_path: &PathBuf, model_size: &str) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🧪 Testando Modelo");
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();

    let tokenizer =
        BPETokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Erro carregando tokenizer");

    let mut config = get_model_config(model_size);
    config.dropout = 0.0;

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);
    let model = model
        .load_file(
            model_path.to_str().unwrap(),
            &CompactRecorder::new(),
            &device,
        )
        .expect("Erro carregando modelo");

    let test_prompts = vec![
        "O Brasil é",
        "A Constituição Federal",
        "Em 1500",
        "O presidente da República",
        "A cidade de São Paulo",
        "O artigo 5º garante",
    ];

    println!();
    for prompt in test_prompts {
        let tokens = tokenizer.encode(prompt);
        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), &device);
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);
        let logits_vec: Vec<f32> = logits.into_data().iter::<f32>().collect();

        let probs = softmax(&logits_vec);
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Prompt: \"{}\"", prompt);
        println!("  Top-5:");
        for (i, (token_id, prob)) in indexed.iter().take(5).enumerate() {
            let token_text = tokenizer.decode(&[*token_id as u16]);
            println!(
                "    {}. {:15} {:>6.2}%",
                i + 1,
                format!("\"{}\"", token_text.trim()),
                prob * 100.0
            );
        }
        println!();
    }
}

// ============ GENERATE ============
fn generate(
    model_path: &PathBuf,
    tokenizer_path: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    model_size: &str,
    temperature: f32,
    top_k: usize,
) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✨ Gerando Texto");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Prompt: {}", prompt);
    println!("  Temperature: {}", temperature);
    println!("  Top-K: {}", top_k);
    println!();

    let device = get_device();

    let tokenizer =
        BPETokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Erro carregando tokenizer");

    let mut config = get_model_config(model_size);
    config.dropout = 0.0;

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);
    let model = model
        .load_file(
            model_path.to_str().unwrap(),
            &CompactRecorder::new(),
            &device,
        )
        .expect("Erro carregando modelo");

    let mut tokens = tokenizer.encode(prompt);
    let mut rng = rand::thread_rng();

    print!("{}", prompt);
    std::io::stdout().flush().unwrap();

    for _ in 0..max_tokens {
        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), &device);
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);
        let logits_vec: Vec<f32> = logits.into_data().iter::<f32>().collect();

        // Temperature scaling
        let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);

        // Softmax over top-k
        let max_logit = indexed
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_logit).exp()).sum();
        let probs: Vec<f32> = indexed
            .iter()
            .map(|(_, v)| (v - max_logit).exp() / exp_sum)
            .collect();
        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        // Sample
        let dist = WeightedIndex::new(&probs).unwrap();
        let sampled_idx = dist.sample(&mut rng);
        let next_token = indices[sampled_idx] as u16;

        if next_token == tokenizer.eos_id() {
            break;
        }

        tokens.push(next_token);
        let decoded = tokenizer.decode(&[next_token]);
        print!("{}", decoded);
        std::io::stdout().flush().unwrap();
    }

    println!();
    println!();
}

// ============ CLEAN CORPUS ============
fn clean_corpus(input: &PathBuf, output: &PathBuf, verbose: bool) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🧹 Limpando Corpus");
    println!("═══════════════════════════════════════════════════════════");

    let cleaner = WikiCleaner::new();
    let normalizer = PTBRNormalizer::new();

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let files = collect_text_files(input);
    let mut total_before = 0usize;
    let mut total_after = 0usize;

    for path in &files {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        total_before += content.len();

        let normalized = normalizer.normalize(&content);
        let cleaned = cleaner.clean(&normalized);
        total_after += cleaned.len();

        let out_path = output.join(path.file_name().unwrap());
        std::fs::write(&out_path, &cleaned).expect("Erro escrevendo");

        if verbose {
            println!(
                "  {:?}: {} -> {}",
                path.file_name().unwrap(),
                format_bytes(content.len()),
                format_bytes(cleaned.len())
            );
        }
    }

    let reduction = if total_before > 0 {
        100.0 * (1.0 - total_after as f64 / total_before as f64)
    } else {
        0.0
    };

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Limpeza concluída!");
    println!("  Arquivos: {}", files.len());
    println!("  Antes: {}", format_bytes(total_before));
    println!("  Depois: {}", format_bytes(total_after));
    println!("  Redução: {:.1}%", reduction);
    println!("═══════════════════════════════════════════════════════════");
}

// ============ INFO ============
fn show_info(model_size: &str) {
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

// ============ BUILD DATASET ============
fn build_dataset(
    tokenizer_path: &PathBuf,
    output_bin: &PathBuf,
    sources: &[String],
    min_chars: usize,
    blocks: bool,
    clean: bool,
    _seed: u64,
) {
    use std::io::{BufRead, BufReader};

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧱 Build Dataset (streaming) -> train.bin");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Tokenizer: {:?}", tokenizer_path);
    println!("  Output: {:?}", output_bin);
    println!("  min_chars: {}", min_chars);
    println!(
        "  mode: {}",
        if blocks { "blocks(\\n\\n)" } else { "lines" }
    );
    println!("  clean: {}", clean);
    println!();

    let tokenizer =
        BPETokenizer::from_file(tokenizer_path.to_str().unwrap()).expect("Erro carregando tokenizer");
    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    println!("  BOS: {}", bos);
    println!("  EOS: {}", eos);
    println!("  vocab_size: {}", tokenizer.vocab_size());
    println!();

    let normalizer = PTBRNormalizer::new();
    let cleaner = WikiCleaner::new();

    if let Some(parent) = output_bin.parent() {
        std::fs::create_dir_all(parent).expect("Erro criando diretório de saída");
    }
    let mut writer =
        TokenizedDatasetWriter::new(output_bin.as_path()).expect("Erro criando train.bin");

    let mut parsed: Vec<(PathBuf, usize)> = Vec::new();
    for s in sources {
        let (path, weight) = parse_source_spec(s);
        parsed.push((path, weight));
    }

    let mut total_docs = 0usize;
    let mut total_tokens = 0usize;

    for (path, weight) in parsed {
        if !path.exists() {
            println!("  ⚠ Fonte não existe: {:?}", path);
            continue;
        }

        let files = collect_txt_files_sorted(&path);

        println!(
            "  Fonte: {:?} | weight={}x | files={}",
            path,
            weight,
            files.len()
        );

        for w in 0..weight {
            if weight > 1 {
                println!("    Pass {}/{}", w + 1, weight);
            }

            for f in &files {
                let file = std::fs::File::open(f).expect("Erro abrindo arquivo fonte");
                let mut reader = BufReader::with_capacity(1024 * 1024, file);

                let use_blocks = if blocks {
                    file_has_double_newline(f)
                } else {
                    false
                };

                if use_blocks {
                    let mut doc = String::new();
                    let mut line = String::new();

                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line).expect("Erro lendo linha");
                        if n == 0 {
                            flush_doc(
                                &mut doc,
                                min_chars,
                                clean,
                                &normalizer,
                                &cleaner,
                                &tokenizer,
                                &mut writer,
                                bos,
                                eos,
                                &mut total_docs,
                                &mut total_tokens,
                            );
                            break;
                        }

                        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
                        if trimmed.trim().is_empty() {
                            flush_doc(
                                &mut doc,
                                min_chars,
                                clean,
                                &normalizer,
                                &cleaner,
                                &tokenizer,
                                &mut writer,
                                bos,
                                eos,
                                &mut total_docs,
                                &mut total_tokens,
                            );
                            doc.clear();
                        } else {
                            doc.push_str(trimmed);
                            doc.push('\n');
                        }
                    }
                } else {
                    let mut line = String::new();
                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line).expect("Erro lendo linha");
                        if n == 0 {
                            break;
                        }
                        let d = line.trim();
                        if d.len() < min_chars {
                            continue;
                        }

                        let mut text = normalizer.normalize(d);
                        if clean {
                            text = cleaner.clean(&text);
                        }
                        if text.len() < min_chars {
                            continue;
                        }

                        let mut toks = Vec::new();
                        toks.push(bos);
                        toks.extend(tokenizer.encode(&text));
                        toks.push(eos);

                        writer.write_tokens(&toks).expect("Erro escrevendo tokens");
                        total_docs += 1;
                        total_tokens += toks.len();
                    }
                }
            }
        }
    }

    let written = writer.finish().expect("Erro finalizando writer");

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Dataset pronto!");
    println!("  Docs: {}", format_number(total_docs));
    println!("  Tokens (contado): {}", format_number(total_tokens));
    println!("  Tokens (writer): {}", format_number(written));
    println!("  Arquivo: {:?}", output_bin);
    println!("═══════════════════════════════════════════════════════════");
}

fn parse_source_spec(s: &str) -> (PathBuf, usize) {
    if let Some((a, b)) = s.rsplit_once(':') {
        if let Ok(w) = b.parse::<usize>() {
            return (PathBuf::from(a), w.max(1));
        }
    }
    (PathBuf::from(s), 1)
}

fn collect_txt_files_sorted(path: &PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.clone());
        return files;
    }

    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "txt").unwrap_or(false) {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn file_has_double_newline(path: &PathBuf) -> bool {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = vec![0u8; 1024 * 1024];
    let n = match f.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    buf.truncate(n);
    buf.windows(2).any(|w| w == b"\n\n")
}

fn flush_doc(
    doc: &mut String,
    min_chars: usize,
    clean: bool,
    normalizer: &PTBRNormalizer,
    cleaner: &WikiCleaner,
    tokenizer: &BPETokenizer,
    writer: &mut TokenizedDatasetWriter,
    bos: u16,
    eos: u16,
    total_docs: &mut usize,
    total_tokens: &mut usize,
) {
    let d = doc.trim();
    if d.len() < min_chars {
        return;
    }

    let mut text = normalizer.normalize(d);
    if clean {
        text = cleaner.clean(&text);
    }
    if text.len() < min_chars {
        return;
    }

    let mut toks = Vec::new();
    toks.push(bos);
    toks.extend(tokenizer.encode(&text));
    toks.push(eos);

    writer.write_tokens(&toks).expect("Erro escrevendo tokens");
    *total_docs += 1;
    *total_tokens += toks.len();
}

// ============ AUDIT CORPUS ============
struct FileAudit {
    path: PathBuf,
    score: f32,
    issues: Vec<String>,
    #[allow(dead_code)]
    bytes: u64,
}

fn audit_corpus_cmd(input: &PathBuf, output: &PathBuf) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🕵️  Auditoria Profunda de Corpus (Quality Score)");
    println!("═══════════════════════════════════════════════════════════");

    let files = collect_all_txt_files(input);
    println!("  Arquivos encontrados: {}", files.len());
    println!("  Analisando paralelamente...");

    let mut results: Vec<FileAudit> = files.par_iter().map(|path| analyze_file_quality(path)).collect();

    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut approved = 0;
    let mut rejected = 0;
    let mut report_file = std::fs::File::create(output).expect("Erro criando report");

    writeln!(report_file, "SCORE\tPATH\tISSUES").unwrap();

    for r in &results {
        if r.score >= 50.0 {
            approved += 1;
        } else {
            rejected += 1;
            println!(
                "❌ REJEITADO ({:.1}): {:?} -> {:?}",
                r.score,
                r.path.file_name().unwrap(),
                r.issues
            );
        }
        writeln!(report_file, "{:.1}\t{:?}\t{:?}", r.score, r.path, r.issues).unwrap();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Auditoria Finalizada");
    println!("  Aprovados: {} (Score >= 50)", approved);
    println!("  Rejeitados: {} (Lixo/Ruído)", rejected);
    println!("  Relatório salvo em: {:?}", output);
    println!("═══════════════════════════════════════════════════════════");
}

fn collect_all_txt_files(path: &PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.clone());
    } else if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    files.extend(collect_all_txt_files(&p));
                } else if p.extension().map_or(false, |e| e == "txt") {
                    files.push(p);
                }
            }
        }
    }
    files
}

fn analyze_file_quality(path: &PathBuf) -> FileAudit {
    let mut issues = Vec::new();
    let mut score: f32 = 100.0;

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => {
            return FileAudit {
                path: path.clone(),
                score: 0.0,
                issues: vec!["Erro leitura".into()],
                bytes: 0,
            }
        }
    };

    let len = content.len();
    if len < 500 {
        return FileAudit {
            path: path.clone(),
            score: 0.0,
            issues: vec!["Muito curto".into()],
            bytes: len as u64,
        };
    }

    // Encoding (Mojibake)
    if content.contains("Ã£") || content.contains("Ã©") || content.contains("Ã³") {
        return FileAudit {
            path: path.clone(),
            score: 0.0,
            issues: vec!["Encoding quebrado".into()],
            bytes: len as u64,
        };
    }

    let lower = content.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len();

    if word_count < 50 {
        score -= 20.0;
        issues.push("Poucas palavras".into());
    }

    // Stopwords Ratio
    let stopwords = [
        " o ", " a ", " de ", " que ", " e ", " do ", " da ", " em ", " um ", " para ",
    ];
    let mut stop_count = 0;
    for s in stopwords {
        stop_count += lower.matches(s).count();
    }
    let stop_ratio = stop_count as f32 / word_count.max(1) as f32;

    if stop_ratio < 0.05 {
        score -= 40.0;
        issues.push("Texto não-natural (sem conectivos)".into());
    } else if stop_ratio > 0.6 {
        score -= 30.0;
        issues.push("Repetitivo demais".into());
    }

    // Riqueza Lexical
    let unique_words: HashSet<&str> = words.iter().cloned().collect();
    let ttr = unique_words.len() as f32 / word_count.max(1) as f32;

    if ttr < 0.05 {
        score -= 30.0;
        issues.push("Vocabulário pobre".into());
    }

    // Frases
    let sentences: Vec<&str> = content.split(|c| c == '.' || c == '!' || c == '?').collect();
    let avg_sentence_len = word_count as f32 / sentences.len().max(1) as f32;

    if avg_sentence_len < 4.0 {
        score -= 20.0;
        issues.push("Frases fragmentadas".into());
    } else if avg_sentence_len > 150.0 {
        score -= 20.0;
        issues.push("Frases longas demais".into());
    }

    // Markup
    let symbols = content
        .chars()
        .filter(|c| "{[]}<>@#$%=|\\".contains(*c))
        .count();
    let symbol_ratio = symbols as f32 / len as f32;
    if symbol_ratio > 0.05 {
        score -= 30.0;
        issues.push("Markup/Código".into());
    }

    // Línguas Estrangeiras
    let en_markers = [" the ", " and ", " is ", " with ", " for "];
    let mut en_count = 0;
    for m in en_markers {
        en_count += lower.matches(m).count();
    }

    let es_markers = [" y ", " el ", " los ", " las ", " una "];
    let mut es_count = 0;
    for m in es_markers {
        es_count += lower.matches(m).count();
    }

    if en_count > stop_count {
        score = 0.0;
        issues.push("Inglês".into());
    }
    if es_count > stop_count / 2 {
        score -= 60.0;
        issues.push("Espanhol".into());
    }

    FileAudit {
        path: path.clone(),
        score: score.max(0.0),
        issues,
        bytes: len as u64,
    }
}

// ============ HELPERS ============
fn get_model_config(model_size: &str) -> RWKVConfig {
    match model_size {
        "85m" | "85M" => RWKVConfig::ptbr_85m(),
        "400m" | "400M" => RWKVConfig::ptbr_400m(),
        "800m" | "800M" => RWKVConfig::ptbr_800m(),
        "1b" | "1B" => RWKVConfig::ptbr_1b(),
        "1.5b" | "1.5B" => RWKVConfig::ptbr_1_5b(),
        _ => {
            println!("  ⚠️ Tamanho '{}' não reconhecido, usando 85m", model_size);
            RWKVConfig::ptbr_85m()
        }
    }
}

fn create_batch_tensor<B: Backend>(data: &[Vec<u16>], device: &B::Device) -> Tensor<B, 2, Int> {
    let batch_size = data.len();
    let seq_len = data[0].len();

    let flat: Vec<i32> = data.iter().flatten().map(|&x| x as i32).collect();

    let tensor: Tensor<B, 1, Int> = Tensor::from_ints(flat.as_slice(), device);
    tensor.reshape([batch_size, seq_len])
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}