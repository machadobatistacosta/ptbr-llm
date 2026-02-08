//! Process Wiki Command
//!
//! Processes Wikipedia XML.BZ2 dump into clean text files.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use crate::data::{WikiCleaner, WikiStreamParser};
use crate::tokenizer::PTBRNormalizer;
use crate::utils::{format_bytes, format_number};

fn create_output_file(output: &PathBuf, idx: usize) -> std::fs::File {
    std::fs::File::create(output.join(format!("wiki_{:04}.txt", idx))).expect("Erro criando arquivo")
}

pub fn execute(input: &PathBuf, output: &PathBuf, min_chars: usize) {
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
