//! Train Command
//!
//! Main training command for RWKV model.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use burn::module::AutodiffModule;
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::backend::{MyBackend, TrainBackend, get_device};
use crate::data::{DataLoader, MmapDataset};
use crate::helpers::{create_batch_tensor, get_model_config};
use crate::error::{PtbrLlmError, Result};
use crate::model::{Evaluator, TrainingConfig, Trainer, RWKV};
use crate::tokenizer::BPETokenizer;
use crate::utils::{format_bytes, format_duration, format_number, format_params};

/// Returns (safe_seq_len, safe_grad_accum, safe_batch_size)
pub(crate) fn get_safe_config(
    model_size: &str,
    seq_len: usize,
    grad_accum: usize,
    batch_size: usize,
) -> (usize, usize, usize) {
    match model_size {
        "400m" | "400M" => {
            // T4 16GB: 400M cabe com batch=2, seq=512
            // Com batch=4, seq=256 também funciona
            let safe_batch = batch_size.min(4);
            let safe_seq = seq_len.min(512);

            // Se batch*seq muito grande, reduz
            if safe_batch * safe_seq > 1024 {
                // batch=2, seq=512 ou batch=4, seq=256
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

pub(crate) fn run_training_loop(
    trainer: &mut Trainer<TrainBackend>,
    dataset: &mut MmapDataset,
    val_dataset: Option<&MmapDataset>,
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
    
    // Cria evaluator para métricas de validação
    let evaluator = Evaluator::new(eval_samples);

    'training: loop {
        dataset.shuffle(42 + epoch);
        let loader = DataLoader::new(dataset, batch_size);
        let total_batches = loader.total_batches();

        if epoch == 0 && trainer.step() == 0 {
            println!("  📦 Processando {} batches no primeiro epoch...", total_batches);
            std::io::stdout().flush().unwrap();
        }

        for (inputs, targets) in loader {
            
            // Validação do batch
            if inputs.is_empty() || targets.is_empty() {
                continue;
            }
            
            let seq_len = inputs[0].len();
            if seq_len == 0 {
                continue;
            }
            
            // Valida que todas as sequências têm o mesmo tamanho
            if !inputs.iter().all(|x| x.len() == seq_len) || !targets.iter().all(|x| x.len() == seq_len) {
                continue;
            }
            
            // Cria tensores
            let input_tensor = create_batch_tensor::<TrainBackend>(&inputs, device);
            let target_tensor = create_batch_tensor::<TrainBackend>(&targets, device);

            // Train step
            if let Some(stats) = trainer.train_step(input_tensor, target_tensor) {
                let step = trainer.step();
                let steps_done = step - initial_step;
                tokens_since_log +=
                    batch_size * seq_len * trainer.config().gradient_accumulation_steps;

                // Log imediato no primeiro step
                if step == 1 {
                    println!("  ✅ Primeiro step completo! Loss inicial: {:.4}", stats.loss);
                    std::io::stdout().flush().unwrap();
                }

                // Log periódico
                if last_log.elapsed().as_secs() >= 5 || step == 1 {
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

                // Evaluation - use val_dataset if available
                if step % eval_every == 0 && step > 0 {
                    let eval_data = val_dataset.unwrap_or(dataset);
                    let eval_metrics = evaluator.evaluate(&trainer.model.valid(), eval_data, device);
                    println!(
                        "  📊 Eval Step {} | {}",
                        step, eval_metrics
                    );

                    writeln!(
                        metrics_file, 
                        "{},,,,,,{:.6},{:.2}",
                        step, eval_metrics.loss, eval_metrics.perplexity
                    ).ok();

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

                // Save checkpoint in its own scope to drop temporaries
                if step % save_every == 0 && step > 0 {
                    {
                        let ckpt_path = output.join(format!("checkpoint_{}", step));
                        match trainer.save_checkpoint(ckpt_path.to_str().unwrap()) {
                            Ok(_) => println!("  💾 Checkpoint salvo: {:?}", ckpt_path),
                            Err(e) => println!("  ⚠️ Erro salvando: {}", e),
                        }
                    }
                    // Tenta liberar memória do sistema operacional
                    // Não temos clear_cache explícito no backend padrão, mas isso dropa vars locais
                }

                // Check completion
                if steps_done >= max_steps {
                    break 'training;
                }
            }
            // Se train_step retornou None, verifica se precisa pular batches (muitos NaN)
            let skip_count = trainer.should_skip_batches();
            if skip_count > 0 {
                // O loop naturalmente avança - só logamos a mensagem
                // O importante é que o estado NaN foi resetado e memória liberada
                eprintln!("  ✅ Estado resetado após {} NaN, continuando...", skip_count);
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

pub(crate) fn generate_sample(
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

pub fn execute(
    data: &PathBuf,
    val_data: Option<&PathBuf>,
    tokenizer_path: &PathBuf,
    output: &PathBuf,
    model_size: &str,
    max_steps: usize,
    save_every: usize,
    batch_size: usize,
    grad_accum: usize,
    learning_rate: f64,
    warmup_steps: usize,
    gradient_clip: f64,
    seq_len: usize,
    eval_every: usize,
    eval_samples: usize,
) -> Result<()> {
    std::env::set_var("CUDA_VISIBLE_DEVICES", "0");

    let (safe_seq_len, safe_grad_accum, safe_batch) = get_safe_config(model_size, seq_len, grad_accum, batch_size);

    if safe_seq_len != seq_len || safe_grad_accum != grad_accum || safe_batch != batch_size {
        println!("  ⚠️ Config ajustada para T4 16GB:");
        println!("     seq_len: {} -> {}", seq_len, safe_seq_len);
        println!("     grad_accum: {} -> {}", grad_accum, safe_grad_accum);
        println!("     batch_size: {} -> {}", batch_size, safe_batch);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  🚀 Iniciando Treinamento RWKV");
    println!("═══════════════════════════════════════════════════════════");

    let device = get_device();

    let mut model_config = get_model_config(model_size);
    model_config.max_seq_len = safe_seq_len;
    model_config.dropout = 0.05;

    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| PtbrLlmError::TokenizerLoad(e.to_string()))?;

    if tokenizer.vocab_size() != model_config.vocab_size {
        println!(
            "  ⚠️ Ajustando vocab_size: {} -> {}",
            model_config.vocab_size,
            tokenizer.vocab_size()
        );
        model_config.vocab_size = tokenizer.vocab_size();
    }

    let effective_batch = safe_batch * safe_grad_accum;

    println!(
        "  Modelo: {} ({} params)",
        model_size,
        format_params(model_config.num_parameters())
    );
    println!(
        "  VRAM estimada: {}",
        format_bytes(model_config.estimated_vram(safe_batch, safe_seq_len))
    );
    println!(
        "  Batch size: {} x {} = {} efetivo",
        safe_batch, safe_grad_accum, effective_batch
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
        return Err(PtbrLlmError::FileNotFound(data.clone()));
    };

    let mut dataset = MmapDataset::from_file(&dataset_path, safe_seq_len)
        .map_err(|e| PtbrLlmError::DatasetCorrupted(e.to_string()))?;

    if dataset.is_empty() {
        return Err(PtbrLlmError::DatasetEmpty {
            path: dataset_path,
            seq_len: safe_seq_len,
        });
    }

    dataset.shuffle(42);
    println!(
        "  Dataset: {} tokens, {} sequências",
        format_number(dataset.num_tokens()),
        format_number(dataset.len())
    );
    println!();

    std::fs::create_dir_all(output)?;

    let mut metrics_file = std::fs::File::create(output.join("metrics.csv"))?;
    writeln!(
        metrics_file,
        "step,loss,ppl,lr,grad_norm,tokens_per_sec,eval_loss,eval_ppl"
    )?;

    // Training config
    let train_config = TrainingConfig {
        learning_rate,
        batch_size: safe_batch,
        gradient_accumulation_steps: safe_grad_accum,
        warmup_steps,
        max_steps,
        // Bug #10 fix: Use default (0.001) instead of hardcoded 0.01
        gradient_clip,
        save_every,
        log_every: 1, // Log every step for debug
        min_lr_ratio: 0.1,
        ..Default::default()
    };

    let mut trainer: Trainer<TrainBackend> =
        Trainer::new(&model_config, train_config, device.clone());

    // Load validation dataset: use --val-data if provided, else split 10%
    let val_dataset = if let Some(val_path) = val_data {
        let val_path_resolved = if val_path.extension().map(|e| e == "bin").unwrap_or(false) {
            val_path.clone()
        } else if val_path.join("val.bin").exists() {
            val_path.join("val.bin")
        } else {
            val_path.clone()
        };
        
        println!("  📊 Validation dataset: {:?}", val_path_resolved);
        match MmapDataset::from_file(&val_path_resolved, safe_seq_len) {
            Ok(vd) => {
                println!("  Val tokens: {}", format_number(vd.num_tokens()));
                Some(vd)
            }
            Err(e) => {
                eprintln!("  ⚠️ Erro carregando val dataset: {}, usando 10% do train", e);
                dataset.reserve_validation(0.1);
                None
            }
        }
    } else {
        println!("  📊 Sem --val-data, usando 10% do train como validação");
        dataset.reserve_validation(0.1);
        None
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  Iniciando loop de treino...");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    run_training_loop(
        &mut trainer,
        &mut dataset,
        val_dataset.as_ref(),
        &tokenizer,
        max_steps,
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
