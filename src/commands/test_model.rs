//! Test Model Command
//!
//! Tests a trained model with sample prompts.

use std::path::PathBuf;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::backend::{MyBackend, get_device};
use crate::helpers::{get_model_config, softmax};
use crate::model::RWKV;
use crate::tokenizer::BPETokenizer;

pub fn execute(model_path: &PathBuf, tokenizer_path: &PathBuf, model_size: &str) {
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
