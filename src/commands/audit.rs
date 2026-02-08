//! Audit Command
//!
//! Audits corpus quality and identifies problematic files.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use rayon::prelude::*;

struct FileAudit {
    path: PathBuf,
    score: f32,
    issues: Vec<String>,
    #[allow(dead_code)]
    bytes: u64,
}

fn collect_all_txt_files(path: &PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        files.push(path.clone());
    } else if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    files.extend(collect_all_txt_files(&p));
                } else if p.extension().map_or(false, |e| e == "txt") {
                    files.push(p);
                }
            }
        }
    }
    files
}

fn analyze_file_quality(path: &PathBuf) -> FileAudit {
    let mut issues = Vec::new();
    let mut score: f32 = 100.0;

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => {
            return FileAudit {
                path: path.clone(),
                score: 0.0,
                issues: vec!["Erro leitura".into()],
                bytes: 0,
            }
        }
    };

    let len = content.len();
    if len < 500 {
        return FileAudit {
            path: path.clone(),
            score: 0.0,
            issues: vec!["Muito curto".into()],
            bytes: len as u64,
        };
    }

    // Encoding (Mojibake)
    if content.contains("Ã£") || content.contains("Ã©") || content.contains("Ã³") {
        return FileAudit {
            path: path.clone(),
            score: 0.0,
            issues: vec!["Encoding quebrado".into()],
            bytes: len as u64,
        };
    }

    let lower = content.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len();

    if word_count < 50 {
        score -= 20.0;
        issues.push("Poucas palavras".into());
    }

    // Stopwords Ratio
    let stopwords = [
        " o ", " a ", " de ", " que ", " e ", " do ", " da ", " em ", " um ", " para ",
    ];
    let mut stop_count = 0;
    for s in stopwords {
        stop_count += lower.matches(s).count();
    }
    let stop_ratio = stop_count as f32 / word_count.max(1) as f32;

    if stop_ratio < 0.05 {
        score -= 40.0;
        issues.push("Texto não-natural (sem conectivos)".into());
    } else if stop_ratio > 0.6 {
        score -= 30.0;
        issues.push("Repetitivo demais".into());
    }

    // Riqueza Lexical
    let unique_words: HashSet<&str> = words.iter().cloned().collect();
    let ttr = unique_words.len() as f32 / word_count.max(1) as f32;

    if ttr < 0.05 {
        score -= 30.0;
        issues.push("Vocabulário pobre".into());
    }

    // Frases
    let sentences: Vec<&str> = content.split(|c| c == '.' || c == '!' || c == '?').collect();
    let avg_sentence_len = word_count as f32 / sentences.len().max(1) as f32;

    if avg_sentence_len < 4.0 {
        score -= 20.0;
        issues.push("Frases fragmentadas".into());
    } else if avg_sentence_len > 150.0 {
        score -= 20.0;
        issues.push("Frases longas demais".into());
    }

    // Markup
    let symbols = content
        .chars()
        .filter(|c| "{[]}<>@#$%=|\\".contains(*c))
        .count();
    let symbol_ratio = symbols as f32 / len as f32;
    if symbol_ratio > 0.05 {
        score -= 30.0;
        issues.push("Markup/Código".into());
    }

    // Línguas Estrangeiras
    let en_markers = [" the ", " and ", " is ", " with ", " for "];
    let mut en_count = 0;
    for m in en_markers {
        en_count += lower.matches(m).count();
    }

    let es_markers = [" y ", " el ", " los ", " las ", " una "];
    let mut es_count = 0;
    for m in es_markers {
        es_count += lower.matches(m).count();
    }

    if en_count > stop_count {
        score = 0.0;
        issues.push("Inglês".into());
    }
    if es_count > stop_count / 2 {
        score -= 60.0;
        issues.push("Espanhol".into());
    }

    FileAudit {
        path: path.clone(),
        score: score.max(0.0),
        issues,
        bytes: len as u64,
    }
}

pub fn execute(input: &PathBuf, output: &PathBuf) {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🕵️  Auditoria Profunda de Corpus (Quality Score)");
    println!("═══════════════════════════════════════════════════════════");

    let files = collect_all_txt_files(input);
    println!("  Arquivos encontrados: {}", files.len());
    println!("  Analisando paralelamente...");

    let mut results: Vec<FileAudit> = files.par_iter().map(|path| analyze_file_quality(path)).collect();

    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let mut approved = 0;
    let mut rejected = 0;
    let mut report_file = std::fs::File::create(output).expect("Erro criando report");

    writeln!(report_file, "SCORE\tPATH\tISSUES").unwrap();

    for r in &results {
        if r.score >= 50.0 {
            approved += 1;
        } else {
            rejected += 1;
            println!(
                "❌ REJEITADO ({:.1}): {:?} -> {:?}",
                r.score,
                r.path.file_name().unwrap(),
                r.issues
            );
        }
        writeln!(report_file, "{:.1}\t{:?}\t{:?}", r.score, r.path, r.issues).unwrap();
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ Auditoria Finalizada");
    println!("  Aprovados: {} (Score >= 50)", approved);
    println!("  Rejeitados: {} (Lixo/Ruído)", rejected);
    println!("  Relatório salvo em: {:?}", output);
    println!("═══════════════════════════════════════════════════════════");
}
