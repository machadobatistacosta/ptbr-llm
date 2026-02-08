//! Clean Corpus Command
//!
//! Cleans and normalizes text corpus.

use std::path::PathBuf;

use crate::data::WikiCleaner;
use crate::tokenizer::PTBRNormalizer;
use crate::utils::format_bytes;

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

pub fn execute(input: &PathBuf, output: &PathBuf, verbose: bool) {
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
