//! Train Tokenizer Command
//!
//! Trains a BPE tokenizer on corpus.

use std::path::PathBuf;
use rayon::prelude::*;

use crate::tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};
use crate::utils::format_number;

pub fn execute(corpus: &PathBuf, output: &PathBuf, vocab_size: usize, special_tokens: Option<String>) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔤 Treinando Tokenizer BPE");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Corpus: {:?}", corpus);
    println!("  Vocab size: {}", vocab_size);
    println!();

    let mut trainer = BPETrainer::new(vocab_size, 5);

    // ✨ Injeta custom special tokens se fornecidos
    if let Some(tokens_str) = special_tokens {
        let custom_tokens: Vec<&str> = tokens_str
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        if !custom_tokens.is_empty() {
            println!("  🎯 Special tokens customizados:");
            for token in &custom_tokens {
                println!("     - {}", token);
            }
            trainer = trainer.with_special_tokens(custom_tokens);
        }
    }

    println!();

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

    println!("  Total de linhas raw: {}", format_number(texts.len()));
    println!("  🔥 MODO TOTAL: Normalizando TODAS as linhas...");
    
    let normalizer = PTBRNormalizer::new();
    let normalized_texts: Vec<String> = texts
        .par_iter()
        .map(|text| normalizer.normalize(text))
        .collect();
    println!("  Normalização concluída ({} linhas).", normalized_texts.len());
    println!();

    let vocab = trainer.train(normalized_texts.into_iter());
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
    println!("  Special tokens: {}", tokenizer.get_all_special_tokens().len());
    println!("═══════════════════════════════════════════════════════════");
}
