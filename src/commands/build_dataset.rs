//! Build Dataset Command
//!
//! Builds tokenized binary dataset from text sources.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;

use crate::data::{TokenizedDatasetWriter, WikiCleaner};
use crate::error::{PtbrLlmError, Result};
use crate::tokenizer::{BPETokenizer, PTBRNormalizer};
use crate::utils::format_number;

fn parse_source_spec(s: &str) -> (PathBuf, usize) {
    if let Some((a, b)) = s.rsplit_once(':') {
        if let Ok(w) = b.parse::<usize>() {
            return (PathBuf::from(a), w.max(1));
        }
    }
    (PathBuf::from(s), 1)
}

/// Collects .txt files from a path, sorted alphabetically.
/// Used for dataset building - similar to clean_corpus/tokenize but sorted.
fn collect_txt_files_sorted(path: &PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.clone());
        return files;
    }

    if let Ok(rd) = std::fs::read_dir(path) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|x| x == "txt").unwrap_or(false) {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn file_has_double_newline(path: &PathBuf) -> bool {
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

fn flush_doc(
    doc: &mut String,
    min_chars: usize,
    clean: bool,
    normalizer: &PTBRNormalizer,
    cleaner: &WikiCleaner,
    tokenizer: &BPETokenizer,
    writer: &mut TokenizedDatasetWriter,
    bos: u16,
    eos: u16,
    total_docs: &mut usize,
    total_tokens: &mut usize,
) -> Result<()> {
    let d = doc.trim();
    if d.len() < min_chars {
        return Ok(());
    }

    let mut text = normalizer.normalize(d);
    if clean {
        text = cleaner.clean(&text);
    }
    if text.len() < min_chars {
        return Ok(());
    }

    let mut toks = Vec::new();
    toks.push(bos);
    toks.extend(tokenizer.encode(&text));
    toks.push(eos);

    writer.write_tokens(&toks)?;
    *total_docs += 1;
    *total_tokens += toks.len();
    Ok(())
}

pub fn execute(
    tokenizer_path: &PathBuf,
    output_bin: &PathBuf,
    sources: &[String],
    min_chars: usize,
    blocks: bool,
    clean: bool,
    _seed: u64,
) -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🧱 Build Dataset (streaming) -> train.bin");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Tokenizer: {:?}", tokenizer_path);
    println!("  Output: {:?}", output_bin);
    println!("  min_chars: {}", min_chars);
    println!(
        "  mode: {}",
        if blocks { "blocks(\\n\\n)" } else { "lines" }
    );
    println!("  clean: {}", clean);
    println!();

    let tokenizer = BPETokenizer::from_file(tokenizer_path.to_str().unwrap())
        .map_err(|e| PtbrLlmError::TokenizerLoad(e.to_string()))?;
    let bos = tokenizer.bos_id();
    let eos = tokenizer.eos_id();

    println!("  BOS: {}", bos);
    println!("  EOS: {}", eos);
    println!("  vocab_size: {}", tokenizer.vocab_size());
    println!();

    let normalizer = PTBRNormalizer::new();
    let cleaner = WikiCleaner::new();

    if let Some(parent) = output_bin.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = TokenizedDatasetWriter::new(output_bin.as_path())
        .map_err(|e| PtbrLlmError::DatasetCorrupted(format!("Erro criando train.bin: {}", e)))?;

    let mut parsed: Vec<(PathBuf, usize)> = Vec::new();
    for s in sources {
        let (path, weight) = parse_source_spec(s);
        parsed.push((path, weight));
    }

    let mut total_docs = 0usize;
    let mut total_tokens = 0usize;

    for (path, weight) in parsed {
        if !path.exists() {
            println!("  ⚠ Fonte não existe: {:?}", path);
            continue;
        }

        let files = collect_txt_files_sorted(&path);

        println!(
            "  Fonte: {:?} | weight={}x | files={}",
            path,
            weight,
            files.len()
        );

        for w in 0..weight {
            if weight > 1 {
                println!("    Pass {}/{}", w + 1, weight);
            }

            for f in &files {
                let file = std::fs::File::open(f)?;
                let mut reader = BufReader::with_capacity(1024 * 1024, file);

                let use_blocks = if blocks {
                    file_has_double_newline(f)
                } else {
                    false
                };

                if use_blocks {
                    let mut doc = String::new();
                    let mut line = String::new();

                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line)?;
                        if n == 0 {
                            flush_doc(
                                &mut doc,
                                min_chars,
                                clean,
                                &normalizer,
                                &cleaner,
                                &tokenizer,
                                &mut writer,
                                bos,
                                eos,
                                &mut total_docs,
                                &mut total_tokens,
                            )?;
                            break;
                        }

                        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
                        if trimmed.trim().is_empty() {
                            flush_doc(
                                &mut doc,
                                min_chars,
                                clean,
                                &normalizer,
                                &cleaner,
                                &tokenizer,
                                &mut writer,
                                bos,
                                eos,
                                &mut total_docs,
                                &mut total_tokens,
                            )?;
                            doc.clear();
                        } else {
                            doc.push_str(trimmed);
                            doc.push('\n');
                        }
                    }
                } else {
                    let mut line = String::new();
                    loop {
                        line.clear();
                        let n = reader.read_line(&mut line)?;
                        if n == 0 {
                            break;
                        }
                        let d = line.trim();
                        if d.len() < min_chars {
                            continue;
                        }

                        let mut text = normalizer.normalize(d);
                        if clean {
                            text = cleaner.clean(&text);
                        }
                        if text.len() < min_chars {
                            continue;
                        }

                        let mut toks = Vec::new();
                        toks.push(bos);
                        toks.extend(tokenizer.encode(&text));
                        toks.push(eos);

                        writer.write_tokens(&toks)?;
                        total_docs += 1;
                        total_tokens += toks.len();
                    }
                }
            }
        }
    }

    let written = writer.finish()
        .map_err(|e| PtbrLlmError::DatasetCorrupted(format!("Erro finalizando writer: {}", e)))?;

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Dataset pronto!");
    println!("  Docs: {}", format_number(total_docs));
    println!("  Tokens (contado): {}", format_number(total_tokens));
    println!("  Tokens (writer): {}", format_number(written));
    println!("  Arquivo: {:?}", output_bin);
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
