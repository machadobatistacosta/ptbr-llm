// src/main.rs

mod data;
mod tokenizer;
mod model;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use data::{WikiStreamParser, WikiParserConfig, WikiCleaner, MmapDataset, DataLoader, TokenizedDatasetWriter};
use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};
use model::{RWKVConfig, TrainingConfig, RWKV, Trainer};

use rand::prelude::*;
use rand::distributions::WeightedIndex;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::tensor::{Tensor, Int, backend::Backend};
use burn::module::Module;
use burn::record::CompactRecorder;

type MyBackend = NdArray;
type TrainBackend = Autodiff<MyBackend>;

#[derive(Parser)]
#[command(name = "ptbr-slm")]
#[command(about = "Small Language Model focado em Portugues do Brasil")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    ProcessWiki {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    TrainTokenizer {
        #[arg(short, long)]
        corpus: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "32000")]
        vocab_size: usize,
    },
    Tokenize {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
    },
    Train {
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "mini")]
        model_size: String,
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(long)]
        save_every: Option<usize>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        grad_accum: Option<usize>,
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long)]
        warmup_steps: Option<usize>,
        #[arg(long)]
        seq_len: Option<usize>,
    },
    Resume {
        #[arg(short, long)]
        checkpoint: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "150000")]
        additional_steps: usize,
        #[arg(long, default_value = "micro")]
        model_size: String,
        #[arg(long)]
        save_every: Option<usize>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        grad_accum: Option<usize>,
        #[arg(long)]
        learning_rate: Option<f64>,
        #[arg(long)]
        seq_len: Option<usize>,
    },
    TestModel {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "micro")]
        model_size: String,
    },
    Generate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        #[arg(long, default_value = "micro")]
        model_size: String,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
    },
    CleanCorpus {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "false")]
        verbose: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::ProcessWiki { input, output } => process_wiki(&input, &output),
        Commands::TrainTokenizer { corpus, output, vocab_size } => train_tokenizer(&corpus, &output, vocab_size),
        Commands::Tokenize { input, output, tokenizer } => tokenize_corpus(&input, &output, &tokenizer),
        Commands::Train {
            data, tokenizer, output, model_size,
            max_steps, save_every, batch_size, grad_accum,
            learning_rate, warmup_steps, seq_len
        } => {
            train_model(
                &data, &tokenizer, &output, &model_size,
                max_steps, save_every, batch_size, grad_accum,
                learning_rate, warmup_steps, seq_len
            );
        }
        Commands::Resume {
            checkpoint, data, output, additional_steps, model_size,
            save_every, batch_size, grad_accum, learning_rate, seq_len
        } => {
            resume_training(
                &checkpoint, &data, &output, additional_steps, &model_size,
                save_every, batch_size, grad_accum, learning_rate, seq_len
            );
        }
        Commands::TestModel { model, tokenizer, model_size } => test_model(&model, &tokenizer, &model_size),
        Commands::Generate { model, tokenizer, prompt, max_tokens, model_size, temperature } => {
            generate(&model, &tokenizer, &prompt, max_tokens, &model_size, temperature)
        }
        Commands::CleanCorpus { input, output, verbose } => clean_corpus(&input, &output, verbose),
    }
}

fn process_wiki(input: &PathBuf, output: &PathBuf) {
    println!("Processando Wikipedia PT-BR...");

    let config = WikiParserConfig::default();
    let parser = WikiStreamParser::new(config);
    let cleaner = WikiCleaner::new();

    std::fs::create_dir_all(output).expect("Erro criando diretorio");

    let mut file_idx = 0;
    let mut current_file = std::fs::File::create(output.join(format!("wiki_{:03}.txt", file_idx)))
        .expect("Erro criando arquivo");
    let mut articles_in_file = 0;
    let mut total_articles = 0;

    for article in parser.parse_streaming(input.to_str().unwrap()) {
        let clean_text = cleaner.clean(&article.text);

        if clean_text.len() >= 100 {
            use std::io::Write;
            writeln!(current_file, "{}", clean_text).expect("Erro escrevendo");
            articles_in_file += 1;
            total_articles += 1;

            if articles_in_file >= 10_000 {
                file_idx += 1;
                current_file = std::fs::File::create(output.join(format!("wiki_{:03}.txt", file_idx)))
                    .expect("Erro criando arquivo");
                articles_in_file = 0;
                println!("Arquivo {} completado ({} artigos total)", file_idx, total_articles);
            }
        }
    }

    println!("Processamento concluido! {} arquivos, {} artigos total.", file_idx + 1, total_articles);
}

fn train_tokenizer(corpus: &PathBuf, output: &PathBuf, vocab_size: usize) {
    println!("Treinando tokenizer BPE com {} tokens...", vocab_size);

    let trainer = BPETrainer::new(vocab_size, 2);

    let texts: Box<dyn Iterator<Item = String>> = if corpus.is_file() {
        println!("Lendo arquivo {:?}...", corpus);
        let content = std::fs::read_to_string(corpus).expect("Erro lendo arquivo");
        Box::new(content.lines().map(String::from).collect::<Vec<_>>().into_iter())
    } else {
        Box::new(
            std::fs::read_dir(corpus)
                .expect("Erro lendo diretorio")
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().extension().map(|e| e == "txt").unwrap_or(false))
                .flat_map(|entry| {
                    println!("Lendo {:?}...", entry.path());
                    std::fs::read_to_string(entry.path())
                        .ok()
                        .into_iter()
                        .flat_map(|content| content.lines().map(String::from).collect::<Vec<_>>())
                })
        )
    };

    let vocab = trainer.train(texts);
    let tokenizer = BPETokenizer::from_vocab(vocab);

    std::fs::create_dir_all(output).expect("Erro criando diretorio de saida");

    let tokenizer_path = output.join("tokenizer.json");
    tokenizer.save(tokenizer_path.to_str().unwrap()).expect("Erro salvando tokenizer");

    // ✅ usa vocab_size() (tira warning) e imprime telemetria útil
    println!("Tokenizer salvo em {:?} (vocab_size={})", tokenizer_path, tokenizer.vocab_size());
}

fn tokenize_corpus(input: &PathBuf, output: &PathBuf, tokenizer_path: &PathBuf) {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    println!("Tokenizando corpus...");

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    // IDs corretos do tokenizer (no v15: BOS=258, EOS=259)
    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    println!("  BOS id: {}", bos);
    println!("  EOS id: {}", eos);
    println!("  vocab_size: {}", tokenizer.vocab_size());

    let normalizer = PTBRNormalizer::new();

    std::fs::create_dir_all(output).expect("Erro criando diretorio");

    let mut writer = TokenizedDatasetWriter::new(&output.join("train.bin"))
        .expect("Erro criando arquivo de saida");

    // Helper: decide se arquivo tem \n\n nos primeiros ~1MB
    fn has_double_newline(path: &PathBuf) -> bool {
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

    // Processa 1 arquivo (streaming)
    fn tokenize_file(
        path: &PathBuf,
        tokenizer: &mut BPETokenizer,
        normalizer: &PTBRNormalizer,
        writer: &mut TokenizedDatasetWriter,
        bos: u16,
        eos: u16,
        split_mode_blocks: bool,
    ) -> usize {
        let file = File::open(path).expect("Erro abrindo arquivo");
        let mut reader = BufReader::with_capacity(1024 * 1024, file);

        let mut total_tokens = 0usize;

        if split_mode_blocks {
            // Modo blocos: acumula até linha vazia
            let mut doc = String::new();
            let mut line = String::new();

            loop {
                line.clear();
                let n = reader.read_line(&mut line).expect("Erro lendo linha");
                if n == 0 {
                    // EOF: flush do doc pendente
                    let d = doc.trim();
                    if !d.is_empty() {
                        let norm = normalizer.normalize(d);
                        if norm.len() >= 80 {
                            let mut toks = Vec::new();
                            toks.push(bos);
                            toks.extend(tokenizer.encode(&norm));
                            toks.push(eos);
                            writer.write_tokens(&toks).expect("Erro escrevendo tokens");
                            total_tokens += toks.len();
                        }
                    }
                    break;
                }

                if line.trim().is_empty() {
                    // fim do doc
                    let d = doc.trim();
                    if !d.is_empty() {
                        let norm = normalizer.normalize(d);
                        if norm.len() >= 80 {
                            let mut toks = Vec::new();
                            toks.push(bos);
                            toks.extend(tokenizer.encode(&norm));
                            toks.push(eos);
                            writer.write_tokens(&toks).expect("Erro escrevendo tokens");
                            total_tokens += toks.len();
                        }
                    }
                    doc.clear();
                } else {
                    doc.push_str(&line);
                }
            }
        } else {
            // Modo linhas: 1 doc por linha (ideal pro wiki_clean gerado com writeln por artigo)
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).expect("Erro lendo linha");
                if n == 0 {
                    break;
                }
                let d = line.trim();
                if d.len() < 80 {
                    continue;
                }
                let norm = normalizer.normalize(d);
                let mut toks = Vec::new();
                toks.push(bos);
                toks.extend(tokenizer.encode(&norm));
                toks.push(eos);
                writer.write_tokens(&toks).expect("Erro escrevendo tokens");
                total_tokens += toks.len();
            }
        }

        total_tokens
    }

    // Coleta arquivos
    let mut files: Vec<PathBuf> = Vec::new();
    if input.is_file() {
        files.push(input.clone());
    } else {
        for entry in std::fs::read_dir(input).expect("Erro lendo diretorio") {
            let entry = entry.expect("Erro lendo entrada");
            if entry.path().extension().map(|e| e == "txt").unwrap_or(false) {
                files.push(entry.path());
            }
        }
        files.sort();
    }

    let mut grand_total = 0usize;
    for path in files {
        let blocks = has_double_newline(&path);
        let tokens = tokenize_file(
            &path,
            &mut tokenizer,
            &normalizer,
            &mut writer,
            bos,
            eos,
            blocks,
        );
        grand_total += tokens;
        println!(
            "Tokenizado: {:?} ({} tokens) [mode={}]",
            path,
            format_number(tokens),
            if blocks { "blocks" } else { "lines" }
        );
    }

    let total = writer.finish().expect("Erro finalizando");
    println!("Total de tokens (writer): {}", format_number(total));
    println!("Total de tokens (contado): {}", format_number(grand_total));
}

fn run_training_loop(
    trainer: &mut Trainer<TrainBackend>,
    dataset: &MmapDataset,
    train_config: &TrainingConfig,
    output: &PathBuf,
) {
    let device = NdArrayDevice::Cpu;
    let data_loader = DataLoader::new(dataset, train_config.batch_size);

    let start = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    let mut total_loss = 0.0f32;
    let mut loss_count = 0;
    let initial_step = trainer.step();

    println!("  Iniciando do step {}...\n", initial_step);

    for (inputs, targets) in data_loader {
        let input_tensor = create_batch_tensor::<TrainBackend>(&inputs, &device);
        let target_tensor = create_batch_tensor::<TrainBackend>(&targets, &device);

        let loss = trainer.train_step(input_tensor, target_tensor);
        total_loss += loss;
        loss_count += 1;

        if last_print.elapsed().as_secs() >= 5 {
            let step = trainer.step();
            let steps_done = step - initial_step;
            let elapsed = start.elapsed().as_secs();
            let avg_loss = total_loss / loss_count as f32;
            let steps_per_sec = steps_done as f64 / elapsed.max(1) as f64;
            let remaining = train_config.max_steps.saturating_sub(steps_done);
            let eta = (remaining as f64 / steps_per_sec.max(0.01)) as u64;

            println!(
                "Step {:6} | Loss: {:.4} | LR: {:.2e} | {:.2} steps/s | ETA: {}h{}m",
                step,
                avg_loss,
                trainer.current_lr(),
                steps_per_sec,
                eta / 3600,
                (eta % 3600) / 60
            );

            total_loss = 0.0;
            loss_count = 0;
            last_print = std::time::Instant::now();
        }

        let steps_done = trainer.step() - initial_step;
        if steps_done % train_config.save_every == 0 && steps_done > 0 {
            let checkpoint_path = output.join(format!("checkpoint_{}.bin", trainer.step()));
            match trainer.save_checkpoint(checkpoint_path.to_str().unwrap()) {
                Ok(_) => println!("  Checkpoint salvo: {:?}", checkpoint_path),
                Err(e) => println!("  Erro salvando checkpoint: {}", e),
            }
        }

        if steps_done >= train_config.max_steps {
            break;
        }
    }

    let final_path = output.join(format!("model_step_{}.bin", trainer.step()));
    trainer.save_checkpoint(final_path.to_str().unwrap())
        .expect("Erro salvando modelo final");

    let elapsed = start.elapsed();
    println!("\n===================================================");
    println!("  Treinamento concluido!");
    println!("  Steps totais: {}", trainer.step());
    println!(
        "  Tempo: {}h{}m",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60
    );
    println!("  Modelo salvo em: {:?}", final_path);
    println!("===================================================");
}

fn train_model(
    data: &PathBuf,
    tokenizer_path: &PathBuf,
    output: &PathBuf,
    model_size: &str,
    max_steps: Option<usize>,
    save_every: Option<usize>,
    batch_size: Option<usize>,
    grad_accum: Option<usize>,
    learning_rate: Option<f64>,
    warmup_steps: Option<usize>,
    seq_len: Option<usize>,
) {
    println!("===================================================");
    println!("  Iniciando treinamento (CPU)");
    println!("===================================================");

    let device = NdArrayDevice::Cpu;

    let mut model_config = match model_size {
        "85m" => RWKVConfig::ptbr_85m(),
        "mini" => RWKVConfig::ptbr_mini(),
        "micro" => RWKVConfig::ptbr_micro(),
        _ => RWKVConfig::ptbr_micro(),
    };

    if let Some(sl) = seq_len {
        model_config.max_seq_len = sl;
    }

    println!("  Modelo: {} ({} parametros)", model_size, format_params(model_config.num_parameters()));

    // ✅ Sanity check: tokenizer vocab vs model vocab
    let tok = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer para sanity check");
    let tok_vocab = tok.vocab_size();
    if tok_vocab != model_config.vocab_size {
        panic!(
            "Vocab mismatch: tokenizer={} model_config={}",
            tok_vocab, model_config.vocab_size
        );
    }

    let mut train_config = TrainingConfig {
        learning_rate: 3e-4,
        batch_size: 2,
        gradient_accumulation_steps: 16,
        warmup_steps: 500,
        max_steps: 50_000,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every: 2500,
        eval_every: 500,
    };

    if let Some(v) = max_steps { train_config.max_steps = v; }
    if let Some(v) = save_every { train_config.save_every = v; }
    if let Some(v) = batch_size { train_config.batch_size = v; }
    if let Some(v) = grad_accum { train_config.gradient_accumulation_steps = v; }
    if let Some(v) = learning_rate { train_config.learning_rate = v; }
    if let Some(v) = warmup_steps { train_config.warmup_steps = v; }

    std::fs::create_dir_all(output).expect("Erro criando diretorio de checkpoints");

    let mut dataset = MmapDataset::from_file(
        &data.join("train.bin"),
        model_config.max_seq_len,
    ).expect("Erro carregando dataset");

    if dataset.is_empty() {
        panic!(
            "Dataset vazio: seq_len={} pode estar alto demais, ou train.bin está errado",
            model_config.max_seq_len
        );
    }

    // ✅ Shuffle real
    dataset.shuffle(42);

    println!("  Sequencias: {}", format_number(dataset.len()));
    println!(
        "  Batch size: {} (efetivo: {})",
        train_config.batch_size,
        train_config.batch_size * train_config.gradient_accumulation_steps
    );
    println!("  Seq len: {}", model_config.max_seq_len);
    println!("  Max steps: {}", format_number(train_config.max_steps));
    println!("  Learning rate: {:.2e}", train_config.learning_rate);
    println!("  Warmup steps: {}", train_config.warmup_steps);
    println!("  Save every: {} steps", train_config.save_every);
    println!("===================================================\n");

    let mut trainer: Trainer<TrainBackend> = Trainer::new(&model_config, train_config.clone(), device.clone());

    // ✅ usa getters do modelo + trainer.config()
    println!(
        "  Runtime model: vocab={} d_model={} layers={}",
        trainer.model.vocab_size(),
        trainer.model.d_model(),
        trainer.model.n_layers(),
    );
    println!(
        "  Trainer config: lr={:.2e} wd={:.2e} clip={}",
        trainer.config().learning_rate,
        trainer.config().weight_decay,
        trainer.config().gradient_clip
    );

    run_training_loop(&mut trainer, &dataset, &train_config, output);
}

fn resume_training(
    checkpoint_path: &PathBuf,
    data: &PathBuf,
    output: &PathBuf,
    additional_steps: usize,
    model_size: &str,
    save_every: Option<usize>,
    batch_size: Option<usize>,
    grad_accum: Option<usize>,
    learning_rate: Option<f64>,
    seq_len: Option<usize>,
) {
    println!("===================================================");
    println!("  Retomando treinamento de checkpoint");
    println!("===================================================");

    let device = NdArrayDevice::Cpu;

    let mut model_config = match model_size {
        "85m" => RWKVConfig::ptbr_85m(),
        "mini" => RWKVConfig::ptbr_mini(),
        "micro" => RWKVConfig::ptbr_micro(),
        _ => RWKVConfig::ptbr_micro(),
    };

    if let Some(sl) = seq_len {
        model_config.max_seq_len = sl;
    }

    println!("  Modelo: {} ({} parametros)", model_size, format_params(model_config.num_parameters()));
    println!("  Checkpoint: {:?}", checkpoint_path);

    let mut train_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 2,
        gradient_accumulation_steps: 16,
        warmup_steps: 100,
        max_steps: additional_steps,
        weight_decay: 0.01,
        gradient_clip: 1.0,
        save_every: 5000,
        eval_every: 500,
    };

    if let Some(v) = save_every { train_config.save_every = v; }
    if let Some(v) = batch_size { train_config.batch_size = v; }
    if let Some(v) = grad_accum { train_config.gradient_accumulation_steps = v; }
    if let Some(v) = learning_rate { train_config.learning_rate = v; }

    std::fs::create_dir_all(output).expect("Erro criando diretorio de checkpoints");

    let mut dataset = MmapDataset::from_file(
        &data.join("train.bin"),
        model_config.max_seq_len,
    ).expect("Erro carregando dataset");

    if dataset.is_empty() {
        panic!(
            "Dataset vazio: seq_len={} pode estar alto demais, ou train.bin está errado",
            model_config.max_seq_len
        );
    }

    // ✅ Shuffle real
    dataset.shuffle(42);

    println!("  Sequencias: {}", format_number(dataset.len()));
    println!("  Steps adicionais: {}", format_number(additional_steps));
    println!("  Learning rate: {:.2e}", train_config.learning_rate);
    println!(
        "  Batch size: {} (efetivo: {})",
        train_config.batch_size,
        train_config.batch_size * train_config.gradient_accumulation_steps
    );

    let mut trainer: Trainer<TrainBackend> = Trainer::new(&model_config, train_config.clone(), device.clone());

    // ✅ usa getters do modelo + trainer.config()
    println!(
        "  Runtime model: vocab={} d_model={} layers={}",
        trainer.model.vocab_size(),
        trainer.model.d_model(),
        trainer.model.n_layers(),
    );
    println!(
        "  Trainer config: lr={:.2e} wd={:.2e} clip={}",
        trainer.config().learning_rate,
        trainer.config().weight_decay,
        trainer.config().gradient_clip
    );

    println!("\n  Carregando pesos do checkpoint...");
    trainer.load_checkpoint(checkpoint_path.to_str().unwrap())
        .expect("Erro ao carregar checkpoint");
    println!("  Checkpoint carregado com sucesso!");

    println!("===================================================\n");

    run_training_loop(&mut trainer, &dataset, &train_config, output);
}

fn test_model(model_path: &PathBuf, tokenizer_path: &PathBuf, model_size: &str) {
    println!("===================================================");
    println!("  Teste de Modelo - Top-5 Predicoes");
    println!("===================================================\n");

    let device = NdArrayDevice::Cpu;

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    // ✅ Dropout OFF na inferência
    let mut config = match model_size {
        "85m" => RWKVConfig::ptbr_85m(),
        "mini" => RWKVConfig::ptbr_mini(),
        "micro" => RWKVConfig::ptbr_micro(),
        _ => RWKVConfig::ptbr_micro(),
    };
    config.dropout = 0.0;

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);

    let recorder = CompactRecorder::new();
    let model = model.load_file(model_path.to_str().unwrap(), &recorder, &device)
        .expect("Erro carregando modelo");

    let test_prompts = vec![
        "O Brasil",
        "A cidade de Sao Paulo",
        "Em 1500",
        "O presidente",
        "A lingua portuguesa",
    ];

    for prompt in test_prompts {
        let tokens = tokenizer.encode(prompt);
        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), &device);
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);

        let dims = logits.dims();
        let vocab_size = dims[dims.len() - 1];
        let seq_len_out = dims[dims.len() - 2];

        let last_token_logits = if dims.len() == 3 {
            logits.slice([0..1, seq_len_out-1..seq_len_out, 0..vocab_size])
        } else {
            logits.slice([seq_len_out-1..seq_len_out, 0..vocab_size])
        };

        let logits_vec: Vec<f32> = last_token_logits.into_data().iter::<f32>().collect();

        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits_vec.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Prompt: \"{}\"", prompt);
        println!("Top-5 proximos tokens:");
        for (i, (token_id, prob)) in indexed.iter().take(5).enumerate() {
            let token_text = tokenizer.decode(&[*token_id as u16]);
            println!(
                "  {}. {:15} {:>6.2}%",
                i + 1,
                format!("\"{}\"", token_text.trim()),
                prob * 100.0
            );
        }
        println!();
    }
}

fn generate(
    model_path: &PathBuf,
    tokenizer_path: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    model_size: &str,
    temperature: f32,
) {
    println!("Gerando texto...");

    let device = NdArrayDevice::Cpu;

    let mut tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .expect("Erro carregando tokenizer");

    // ✅ Dropout OFF na inferência
    let mut config = match model_size {
        "85m" => RWKVConfig::ptbr_85m(),
        "mini" => RWKVConfig::ptbr_mini(),
        "micro" => RWKVConfig::ptbr_micro(),
        _ => RWKVConfig::ptbr_micro(),
    };
    config.dropout = 0.0;

    let model: RWKV<MyBackend> = RWKV::new(&config, &device);

    let recorder = CompactRecorder::new();
    let model = model.load_file(model_path.to_str().unwrap(), &recorder, &device)
        .expect("Erro carregando modelo");

    let mut tokens = tokenizer.encode(prompt);

    println!("Prompt: {}", prompt);
    println!("Temperatura: {}", temperature);
    print!("Gerado: ");

    let mut rng = rand::thread_rng();

    for _ in 0..max_tokens {
        let input_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let seq_len = input_vec.len();

        let input: Tensor<MyBackend, 1, Int> = Tensor::from_ints(input_vec.as_slice(), &device);
        let input = input.reshape([1, seq_len]);

        let logits = model.forward_inference(input);

        let dims = logits.dims();
        let vocab_size = dims[dims.len() - 1];
        let seq_len_out = dims[dims.len() - 2];

        let last_token_logits = if dims.len() == 3 {
            logits.slice([0..1, seq_len_out-1..seq_len_out, 0..vocab_size])
        } else {
            logits.slice([seq_len_out-1..seq_len_out, 0..vocab_size])
        };

        let logits_vec: Vec<f32> = last_token_logits.into_data().iter::<f32>().collect();
        let temp_logits: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();
        let max_logit = temp_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = temp_logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

        let dist = WeightedIndex::new(&probs).unwrap();
        let next_token = dist.sample(&mut rng) as u16;

        if next_token == tokenizer.eos_id() {
            break;
        }

        tokens.push(next_token);

        let decoded = tokenizer.decode(&[next_token]);
        print!("{}", decoded);
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    println!();
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

    let tensor: Tensor<B, 1, Int> = Tensor::from_ints(flat.as_slice(), device);
    tensor.reshape([batch_size, seq_len])
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}GB", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}MB", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}KB", n as f64 / 1_000.0)
    } else {
        format!("{}B", n)
    }
}

fn clean_corpus(input: &PathBuf, output: &PathBuf, verbose: bool) {
    println!("===================================================");
    println!("  Limpando corpus");
    println!("===================================================\n");

    std::fs::create_dir_all(output).expect("Erro criando diretorio");

    let cleaner = WikiCleaner::new();

    let mut total_chars_before = 0usize;
    let mut total_chars_after = 0usize;
    let mut file_count = 0usize;

    let mut entries: Vec<_> = std::fs::read_dir(input)
        .expect("Erro lendo diretorio")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "txt").unwrap_or(false))
        .collect();

    entries.sort_by_key(|e| e.path());
    println!("  Encontrados {} arquivos\n", entries.len());

    for entry in entries {
        let content = match std::fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => {
                match std::fs::read(entry.path()) {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Err(e) => {
                        println!("  Erro lendo {:?}: {}", entry.file_name(), e);
                        continue;
                    }
                }
            }
        };

        total_chars_before += content.len();

        let mut cleaned_blocks = Vec::new();

        for block in content.split("\n\n") {
            if block.trim().is_empty() {
                continue;
            }

            let clean_block = cleaner.clean(block);

            if clean_block.len() > 100
                && clean_block.lines().any(|l| l.len() > 50)
                && !has_garbage(&clean_block)
            {
                cleaned_blocks.push(clean_block);
            }
        }

        let clean_content = cleaned_blocks.join("\n\n");
        total_chars_after += clean_content.len();

        if !clean_content.is_empty() {
            let output_path = output.join(entry.file_name());
            std::fs::write(&output_path, &clean_content)
                .expect("Erro escrevendo arquivo");
            file_count += 1;

            if verbose {
                println!(
                    "  {:?}: {}KB -> {}KB",
                    entry.file_name(),
                    content.len() / 1024,
                    clean_content.len() / 1024
                );
            }
        }
    }

    let pct = if total_chars_before > 0 {
        100.0 * (1.0 - total_chars_after as f64 / total_chars_before as f64)
    } else {
        0.0
    };

    println!("\n===================================================");
    println!("  Arquivos salvos: {}", file_count);
    println!(
        "  Tamanho: {} -> {} ({:.1}% removido)",
        format_bytes(total_chars_before),
        format_bytes(total_chars_after),
        pct
    );
    println!("===================================================");
}

fn has_garbage(text: &str) -> bool {
    let garbage = [
        "|", "{|", "|-", "align=", "width=", "colspan=",
        "latM=", "lonM=", "{{", "}}", "[[Categoria:",
        "Ficheiro:", "<ref", "<!--",
    ];
    garbage.iter().any(|g| text.contains(g))
}