//! PTBR-LLM: RWKV Language Model for Brazilian Portuguese
//!
//! A modular CLI application for training and using RWKV models.

#![allow(unused)]
#![allow(dead_code)]

mod backend;
mod commands;
mod data;
mod error;
mod helpers;
mod logger;
mod model;
mod tokenizer;
mod utils;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

// ============ CLI ============
#[derive(Parser)]
#[command(name = "ptbr-llm")]
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
        /// Special tokens (comma-separated): '[PAD]','[UNK]','[BOS]','[EOS]','[SEP]'
        /// Para ChatML: '[PAD]','[UNK]','[BOS]','[EOS]','[SEP]','<|im_start|>','<|im_end|>'
        #[arg(long)]
        special_tokens: Option<String>,
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

    /// Treina modelo RWKV (inicia do zero ou retoma)
    Train(commands::train::TrainArgs),

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
        model: Option<PathBuf>,
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
        /// Saída em formato JSON (para integração com Python)
        #[arg(long, default_value = "false")]
        json: bool,
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

    let result: Result<(), Box<dyn std::error::Error>> = match cli.command {
        Commands::ProcessWiki {
            input,
            output,
            min_chars,
        } => {
            commands::process_wiki::execute(&input, &output, min_chars);
            Ok(())
        },

        Commands::TrainTokenizer {
            corpus,
            output,
            vocab_size,
            special_tokens,
        } => {
            commands::train_tokenizer::execute(&corpus, &output, vocab_size, special_tokens);
            Ok(())
        },

        Commands::Tokenize {
            input,
            output,
            tokenizer,
        } => {
            commands::tokenize::execute(&input, &output, &tokenizer);
            Ok(())
        },

        Commands::Train(args) => commands::train::execute(args).map_err(Into::into),

        Commands::TestModel {
            model,
            tokenizer,
            model_size,
        } => {
            commands::test_model::execute(&model, &tokenizer, &model_size);
            Ok(())
        },

        Commands::Generate {
            model,
            tokenizer,
            prompt,
            max_tokens,
            model_size,
            temperature,
            top_k,
            json,
        } => {
            commands::generate::execute(
                &model,
                &tokenizer,
                &prompt,
                max_tokens,
                &model_size,
                temperature,
                top_k,
                json,
            ).map_err(Into::into)
        },

        Commands::AuditCorpus { input, output } => {
            commands::audit::execute(&input, &output);
            Ok(())
        },

        Commands::CleanCorpus {
            input,
            output,
            verbose,
        } => {
            commands::clean_corpus::execute(&input, &output, verbose);
            Ok(())
        },

        Commands::Info { model_size } => {
            commands::info::execute(&model_size);
            Ok(())
        },

        Commands::BuildDataset {
            tokenizer,
            output,
            source,
            min_chars,
            blocks,
            clean,
            seed,
        } => {
            commands::build_dataset::execute(
                &tokenizer, &output, &source, min_chars, blocks, clean, seed
            ).map_err(Into::into)
        },

        Commands::TestGpu { model_size, seq_len } => {
            commands::test_gpu::execute(&model_size, seq_len);
            Ok(())
        },

        Commands::FindLr {
            data,
            tokenizer,
            model_size,
            num_steps,
        } => {
            commands::find_lr::execute(&data, &tokenizer, &model_size, num_steps);
            Ok(())
        },

        Commands::Benchmark {
            model_size,
            seq_len,
            num_iterations,
        } => {
            commands::benchmark::execute(&model_size, seq_len, num_iterations);
            Ok(())
        },
    };

    if let Err(e) = result {
        eprintln!("❌ Erro: {}", e);
        std::process::exit(1);
    }
}