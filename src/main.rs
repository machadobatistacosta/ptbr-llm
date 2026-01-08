mod data;
mod tokenizer;
mod model;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::io::Write;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::{Tensor, Int, backend::Backend};
use burn::module::Module;
use burn::record::CompactRecorder;

// ============ BACKEND SELECTOR ============
#[cfg(feature = "cpu")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "gpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice, AutoGraphicsApi};
#[cfg(feature = "cuda")]
use burn::backend::cuda_jit::{CudaDevice, CudaRuntime};

#[cfg(feature = "cpu")]
type MyBackend = NdArray;
#[cfg(feature = "gpu")]
type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
#[cfg(feature = "cuda")]
type MyBackend = burn::backend::cuda_jit::Cuda;

type TrainBackend = Autodiff<MyBackend>;

fn get_device() -> <MyBackend as Backend>::Device {
    #[cfg(feature = "cpu")]
    { NdArrayDevice::Cpu }
    #[cfg(feature = "gpu")]
    { WgpuDevice::BestAvailable }
    #[cfg(feature = "cuda")]
    { CudaDevice::new(0) }
}

// ============ IMPORTS ============
use data::{WikiStreamParser, WikiCleaner, MmapDataset, DataLoader, TokenizedDatasetWriter};
use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};
use model::{RWKVConfig, TrainingConfig, RWKV, Trainer};

use rand::prelude::*;
use rand::distributions::WeightedIndex;

// ============ CLI ============
#[derive(Parser)]
#[command(name = "ptbr-slm")]
#[command(author = "Caike Costa")]
#[command(version = "1.0.0")]
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
    
    /// Mostra informações do modelo
    Info {
        #[arg(long, default_value = "85m")]
        model_size: String,
    },
}

fn main() {
    // Inicializa logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::ProcessWiki { input, output, min_chars } => 
            process_wiki(&input, &output, min_chars),
        Commands::TrainTokenizer { corpus, output, vocab_size } => 
            train_tokenizer(&corpus, &output, vocab_size),
        Commands::Tokenize { input, output, tokenizer } => 
            tokenize_corpus(&input, &output, &tokenizer),
        Commands::Train { 
            data, tokenizer, output, model_size, max_steps, save_every,
            batch_size, grad_accum, learning_rate, warmup_steps, seq_len 
        } => train_model(
            &data, &tokenizer, &output, &model_size, max_steps, save_every,
            batch_size, grad_accum, learning_rate, warmup_steps, seq_len
        ),
        Commands::Resume {
            checkpoint, data, output, model_size, additional_steps,
            save_every, batch_size, grad_accum, learning_rate, seq_len
        } => resume_training(
            &checkpoint, &data, &output, &model_size, additional_steps,
            save_every, batch_size, grad_accum, learning_rate, seq_len
        ),
        Commands::TestModel { model, tokenizer, model_size } => 
            test_model(&model, &tokenizer, &model_size),
        Commands::Generate { model, tokenizer, prompt, max_tokens, model_size, temperature, top_k } => 
            generate(&model, &tokenizer, &prompt, max_tokens, &model_size, temperature, top_k),
        Commands::CleanCorpus { input, output, verbose } => 
            clean_corpus(&input, &output, verbose),
        Commands::Info { model_size } => 
            show_info(&model_size),
    }
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
        // Normaliza -> Limpa
        let normalized = normalizer.normalize(&article.text);
        let clean_text = cleaner.clean(&normalized);

        if clean_text.len() >= min_chars {
            use std::io::Write;
            writeln!(current_file, "{}", clean_text).expect("Erro escrevendo");
            writeln!(current_file).expect("Erro escrevendo newline"); // Separador de docs
            
            articles_in_file += 1;
            total_articles += 1;
            total_chars += clean_text.len();

            if articles_in_file >= 10_000 {
                file_idx += 1;
                current_file = create_output_file(output, file_idx);
                articles_in_file = 0;
                println!("  ✓ Arquivo {} completado ({} artigos)", file_idx, total_articles);
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
    std::fs::File::create(output.join(format!("wiki_{:04}.txt", idx)))
        .expect("Erro criando arquivo")
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

    // Coleta textos
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
    tokenizer.save(tokenizer_path.to_str().unwrap()).expect("Erro salvando");

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

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    println!("  BOS ID: {}", bos);
    println!("  EOS ID: {}", eos);
    println!("  Vocab: {}", tokenizer.vocab_size());
    println!();

    let normalizer = PTBRNormalizer::new();
    std::fs::create_dir_all(output).expect("Erro criando diretório");

    let mut writer = TokenizedDatasetWriter::new(&output.join("train.bin"))
        .expect("Erro criando arquivo");

    let mut total_tokens = 0usize;
    let mut total_docs = 0usize;

    // Processa arquivos
    let files = collect_text_files(input);
    
    for path in &files {
        println!("  Processando {:?}...", path.file_name().unwrap_or_default());
        
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                let bytes = std::fs::read(path).unwrap_or_default();
                String::from_utf8_lossy(&bytes).to_string()
            }
        };

        // Split por parágrafo duplo (documentos)
        for doc in content.split("\n\n") {
            let doc = doc.trim();
            if doc.len() < 100 { continue; }

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
) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🚀 Iniciando Treinamento RWKV");
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();
    
    // Carrega config do modelo
    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = seq_len;
    model_config.dropout = 0.05; // Dropout só no treino

    // Valida vocab size com tokenizer
    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");
    
    if tokenizer.vocab_size() != model_config.vocab_size {
        println!("  ⚠️ Ajustando vocab_size: {} -> {}", 
            model_config.vocab_size, tokenizer.vocab_size());
        model_config.vocab_size = tokenizer.vocab_size();
    }

    let effective_batch = batch_size * grad_accum;
    
    println!("  Modelo: {} ({} params)", model_size, format_params(model_config.num_parameters()));
    println!("  Batch size: {} x {} = {} efetivo", batch_size, grad_accum, effective_batch);
    println!("  Seq len: {}", seq_len);
    println!("  Learning rate: {:.2e}", learning_rate);
    println!("  Warmup: {} steps", warmup_steps);
    println!("  Max steps: {}", format_number(max_steps));
    println!("  Save every: {} steps", save_every);
    println!();

    // Carrega dataset
    let mut dataset = MmapDataset::from_file(&data.join("train.bin"), seq_len)
        .expect("Erro carregando dataset");
    
    if dataset.is_empty() {
        panic!("❌ Dataset vazio! Verifique seq_len ({}) e o arquivo train.bin", seq_len);
    }

    dataset.shuffle(42);
    println!("  Sequências: {}", format_number(dataset.len()));
    println!();

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    // Config de treino
    let train_config = TrainingConfig {
        learning_rate,
        batch_size,
        gradient_accumulation_steps: grad_accum,
        warmup_steps,
        max_steps,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every,
        log_every: 10,
    };

    let mut trainer: Trainer<TrainBackend> = Trainer::new(&model_config, train_config, device);

    println!("═══════════════════════════════════════════════════════════");
    println!("  Iniciando loop de treino...");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    run_training_loop(&mut trainer, &mut dataset, max_steps, save_every, batch_size, output);
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
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔄 Retomando Treinamento");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Checkpoint: {:?}", checkpoint);

    let device = get_device();
    
    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = seq_len;
    model_config.dropout = 0.05;

    let train_config = TrainingConfig {
        learning_rate,
        batch_size,
        gradient_accumulation_steps: grad_accum,
        warmup_steps: 100, // Warmup curto para resume
        max_steps: additional_steps,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every,
        log_every: 10,
    };

    let mut trainer: Trainer<TrainBackend> = Trainer::new(&model_config, train_config, device);
    trainer.load_checkpoint(checkpoint.to_str().unwrap())
        .expect("Erro carregando checkpoint");

    let mut dataset = MmapDataset::from_file(&data.join("train.bin"), seq_len)
        .expect("Erro carregando dataset");
    dataset.shuffle(42 + trainer.step() as u64);

    std::fs::create_dir_all(output).expect("Erro criando diretório");

    println!("  Continuando do step {}...", trainer.step());
    println!();

    run_training_loop(&mut trainer, &mut dataset, additional_steps, save_every, batch_size, output);
}

// ============ TRAINING LOOP ============
fn run_training_loop(
    trainer: &mut Trainer<TrainBackend>,
    dataset: &mut MmapDataset,
    max_steps: usize,
    save_every: usize,
    batch_size: usize,
    output: &PathBuf,
) {
    let device = get_device();
    let start = Instant::now();
    let initial_step = trainer.step();
    
    let mut last_log = Instant::now();
    let mut epoch = 0;

    'training: loop {
        // Re-shuffle a cada epoch
        dataset.shuffle(42 + epoch);
        let loader = DataLoader::new(dataset, batch_size);

        for (inputs, targets) in loader {
            let input_tensor = create_batch_tensor::<TrainBackend>(&inputs, &device);
            let target_tensor = create_batch_tensor::<TrainBackend>(&targets, &device);

            if let Some(loss) = trainer.train_step(input_tensor, target_tensor) {
                let step = trainer.step();
                let steps_done = step - initial_step;

                // Log
                if last_log.elapsed().as_secs() >= 5 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let steps_per_sec = steps_done as f64 / elapsed;
                    let remaining = max_steps.saturating_sub(steps_done);
                    let eta_secs = remaining as f64 / steps_per_sec.max(0.01);
                    let ppl = (loss as f64).exp();

                    println!(
                        "  Step {:6} | Loss: {:.4} | PPL: {:7.2} | LR: {:.2e} | {:.2} step/s | ETA: {}",
                        step, loss, ppl, trainer.current_lr(), steps_per_sec, format_duration(eta_secs as u64)
                    );

                    last_log = Instant::now();
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
    trainer.save_checkpoint(final_path.to_str().unwrap())
        .expect("Erro salvando modelo final");

    let elapsed = start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Treinamento concluído!");
    println!("  Steps: {}", trainer.step());
    println!("  Tempo: {}", format_duration(elapsed.as_secs()));
    println!("  Modelo: {:?}", final_path);
    println!("═══════════════════════════════════════════════════════════");
}

// ============ TEST MODEL ============
fn test_model(model_path: &PathBuf, tokenizer_path: &PathBuf, model_size: &str) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🧪 Testando Modelo");
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    let mut config = get_model_config(model_size);
    config.dropout = 0.0; // CRÍTICO: Dropout OFF

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);
    let model = model.load_file(model_path.to_str().unwrap(), &CompactRecorder::new(), &device)
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

        // CORREÇÃO: Especificar tipo explicitamente
        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(
            input_vec.as_slice(), &device
        );
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);
        let logits_vec: Vec<f32> = logits.into_data().iter::<f32>().collect();

        // Top-5
        let probs = softmax(&logits_vec);
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Prompt: \"{}\"", prompt);
        println!("  Top-5:");
        for (i, (token_id, prob)) in indexed.iter().take(5).enumerate() {
            let token_text = tokenizer.decode(&[*token_id as u16]);
            println!("    {}. {:15} {:>6.2}%", 
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

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    let mut config = get_model_config(model_size);
    config.dropout = 0.0; // CRÍTICO

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);
    let model = model.load_file(model_path.to_str().unwrap(), &CompactRecorder::new(), &device)
        .expect("Erro carregando modelo");

    let mut tokens = tokenizer.encode(prompt);
    let mut rng = rand::thread_rng();

    print!("{}", prompt);
    std::io::stdout().flush().unwrap();

    for _ in 0..max_tokens {
        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        // CORREÇÃO: Especificar tipo explicitamente
        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(
            input_vec.as_slice(), &device
        );
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);
        let logits_vec: Vec<f32> = logits.into_data().iter::<f32>().collect();

        // Apply temperature
        let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();
        
        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);
        
        // Softmax over top-k
        let max_logit = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_logit).exp()).sum();
        let probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_logit).exp() / exp_sum).collect();
        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        // Sample
        let dist = WeightedIndex::new(&probs).unwrap();
        let sampled_idx = dist.sample(&mut rng);
        let next_token = indices[sampled_idx] as u16;

        // Check EOS
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
            println!("  {:?}: {} -> {}", 
                path.file_name().unwrap(),
                format_bytes(content.len()),
                format_bytes(cleaned.len())
            );
        }
    }

    let reduction = if total_before > 0 {
        100.0 * (1.0 - total_after as f64 / total_before as f64)
    } else { 0.0 };

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
    
    // Estimativa de memória
    let params = config.num_parameters();
    let mem_fp32 = params * 4;
    let mem_fp16 = params * 2;
    let mem_train_fp32 = mem_fp32 * 4; // pesos + grads + adam states
    let mem_train_fp16 = mem_fp16 + mem_fp32 * 3; // mixed precision
    
    println!("  Memória (Inferência):");
    println!("    FP32: {}", format_bytes(mem_fp32));
    println!("    FP16: {}", format_bytes(mem_fp16));
    println!();
    println!("  Memória (Treino, estimativa):");
    println!("    FP32: {}", format_bytes(mem_train_fp32));
    println!("    Mixed: {}", format_bytes(mem_train_fp16));
    println!("═══════════════════════════════════════════════════════════");
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

fn create_batch_tensor<B: Backend>(
    data: &[Vec<u16>],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch_size = data.len();
    let seq_len = data[0].len();

    let flat: Vec<i32> = data.iter()
        .flatten()
        .map(|&x| x as i32)
        .collect();

    // CORREÇÃO: Criar tensor 1D primeiro, depois reshape
    let tensor: Tensor<B, 1, Int> = Tensor::from_ints(flat.as_slice(), device);
    tensor.reshape([batch_size, seq_len])
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { n.to_string() }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { n.to_string() }
}

fn format_bytes(n: usize) -> String {
    if n >= 1_000_000_000 { format!("{:.2}GB", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}MB", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}KB", n as f64 / 1e3) }
    else { format!("{}B", n) }
}

fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 { format!("{}h{}m{}s", h, m, s) }
    else if m > 0 { format!("{}m{}s", m, s) }
    else { format!("{}s", s) }
}