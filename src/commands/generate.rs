//! Generate Command
//!
//! Generates text from a prompt using the trained model.

use std::io::Write;
use std::path::PathBuf;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::backend::{MyBackend, get_device};
use crate::error::{PtbrError, Result};
use crate::helpers::get_model_config;
use crate::model::{RWKV, RWKVState};
use crate::tokenizer::BPETokenizer;

pub fn execute(
    model_path: &Option<PathBuf>,
    tokenizer_path: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    model_size: &str,
    temperature: f32,
    top_k: usize,
    json_output: bool,
) -> Result<()> {
    if !json_output {
        println!("═══════════════════════════════════════════════════════════");
        println!("  ✨ Gerando Texto");
        println!("═══════════════════════════════════════════════════════════");
        println!("  Prompt: {}", prompt);
        println!("  Temperature: {}", temperature);
        println!("  Top-K: {}", top_k);
        if model_path.is_none() {
            println!("  ⚠️ AVISO: Nenhum modelo fornecido, usando pesos ALEATÓRIOS!");
        }
        println!();
    }

    let device = get_device();

    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| PtbrError::TokenizerLoad(e.to_string()))?;

    let mut config = get_model_config(model_size);
    config.dropout = 0.0;

    let model_raw: RWKV<MyBackend> = RWKV::new(&config, &device);
    let model = if let Some(path) = model_path {
        model_raw
            .load_file(
                path.to_str().unwrap(),
                &CompactRecorder::new(),
                &device,
            )
            .map_err(|e: burn::record::RecorderError| PtbrError::CheckpointLoad(e.to_string()))?
    } else {
        model_raw
    };

    let mut tokens = tokenizer.encode(prompt);
    let initial_token_count = tokens.len();
    let mut rng = rand::thread_rng();
    let mut generated_text = String::new();

    if !json_output {
        print!("{}", prompt);
        std::io::stdout().flush().unwrap();
    }

    // Inicializa estado para inferência incremental
    let mut state = RWKVState::new(config.n_layers, config.d_model, 1, &device);

    // Bug #6 fix: Process all tokens EXCEPT the last one during state init
    // The generation loop will process the last token to get the first prediction
    // Before fix: all tokens processed here, then last token processed AGAIN in generation loop
    if tokens.len() > 1 {
        for &token in &tokens[..tokens.len() - 1] {
            let token_array = [token as i32];
            let token_tensor_1d: Tensor<MyBackend, 1, Int> = 
                Tensor::from_ints(token_array.as_slice(), &device);
            let token_tensor: Tensor<MyBackend, 2, Int> = token_tensor_1d.reshape([1, 1]);
            let _ = model.forward_step(token_tensor, &mut state);
        }
    }


    // Gera tokens incrementalmente
    for _ in 0..max_tokens {
        // Usa último token para gerar próximo
        let last_token = tokens[tokens.len() - 1];
        let token_array = [last_token as i32];
        let token_tensor_1d: Tensor<MyBackend, 1, Int> = 
            Tensor::from_ints(token_array.as_slice(), &device);
        let token_tensor: Tensor<MyBackend, 2, Int> = token_tensor_1d.reshape([1, 1]);

        let logits = model.forward_step(token_tensor, &mut state);
        let logits_vec: Vec<f32> = logits.into_data().iter::<f32>().collect();

        // Temperature scaling
        let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> = scaled
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, v): &(usize, f32)| v.is_finite()) // ✨ Proteção contra NaNs/Infs
            .collect();
            
        // Fallback seguro se tudo for NaN (evita vetor vazio se top_k > 0)
        if indexed.is_empty() {
             eprintln!("⚠️ AVISO: Todos os logits são NaN/Inf! Usando token 0 como fallback.");
             indexed.push((0, 0.0));
        }

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        let next_token = if let Ok(dist) = WeightedIndex::new(&probs) {
            // Normal sampling
            let sampled_idx = dist.sample(&mut rng);
            indices[sampled_idx] as u16
        } else {
            // Fallback: probs inválidas (todas zero ou NaN)
            // Usa determinístico: token com maior logit (primeiro em indexed, pois está sorted)
            if !json_output {
                eprintln!("⚠️ Probabilidades inválidas no sampling, usando greedy fallback");
            }
            if !indices.is_empty() {
                indices[0] as u16
            } else {
                // Último recurso: EOS
                tokenizer.eos_id()
            }
        };

        if next_token == tokenizer.eos_id() {
            break;
        }

        tokens.push(next_token);
        let decoded = tokenizer.decode(&[next_token]);
        generated_text.push_str(&decoded);
        
        if !json_output {
            print!("{}", decoded);
            std::io::stdout().flush().unwrap();
        }
    }

    if json_output {
        // Output JSON estruturado para integração Python
        let json_response = serde_json::json!({
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": format!("{}{}", prompt, generated_text),
            "tokens_generated": tokens.len() - initial_token_count,
            "model_size": model_size,
            "temperature": temperature,
            "top_k": top_k
        });
        println!("{}", json_response);
    } else {
        println!();
        println!();
    }

    Ok(())
}
