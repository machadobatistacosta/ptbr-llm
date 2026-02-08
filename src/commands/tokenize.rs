//! Tokenize Command
//!
//! Tokenizes corpus to binary format.

use std::path::PathBuf;

use crate::data::TokenizedDatasetWriter;
use crate::tokenizer::{BPETokenizer, PTBRNormalizer};
use crate::utils::format_number;

/// Collect .txt files from path (file or directory)
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

pub fn execute(input: &PathBuf, output: &PathBuf, tokenizer_path: &PathBuf) {
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
