use ptbr_llm::{WikiCleaner, DirtySample};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Mutex;
use std::time::Instant;

fn main() {
    let start_time = Instant::now();
    println!("🔍 Surgical Cleaner Audit Tool v1.0");
    println!("===================================");

    let data_dir = "data/tokenizer_full_input_cleaned";
    println!("📂 Scanning directory: {}", data_dir);

    // Colecionador thread-safe de sujeira
    let dirty_samples: Mutex<Vec<DirtySample>> = Mutex::new(Vec::new());
    let total_lines = Mutex::new(0usize);
    let total_files = Mutex::new(0usize);

    // Listar arquivos usando std::fs (sem walkdir)
    // Assumindo estrutura plana conforme verificado
    let paths: Vec<_> = std::fs::read_dir(data_dir)
        .expect("Could not read directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().map_or(false, |ext| ext == "txt"))
        .collect();

    let file_count = paths.len();
    println!("📜 Found {} text files to audit.", file_count);

    // Processamento Paralelo com Rayon
    paths.par_iter().for_each(|path| {
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            let mut file_line_count = 0;

            for line in reader.lines() {
                if let Ok(content) = line {
                    file_line_count += 1;
                    let trimmed = content.trim();
                    
                    if !trimmed.is_empty() {
                         // A mágica acontece aqui: Sensor de Entropia
                         if let Some(sample) = WikiCleaner::audit_line(trimmed) {
                             if let Ok(mut lock) = dirty_samples.lock() {
                                 lock.push(sample);
                             }
                         }
                    }
                }
            }
            
            let mut t_lock = total_lines.lock().unwrap();
            *t_lock += file_line_count;
        }
        let mut f_lock = total_files.lock().unwrap();
        *f_lock += 1;
        if *f_lock % 20 == 0 {
             print!(".");
             std::io::stdout().flush().unwrap();
        }
    });

    println!("\n✅ Scan Complete in {:.2?}s", start_time.elapsed());
    
    // Processar resultados
    let mut samples = dirty_samples.into_inner().unwrap();
    let total_detected = samples.len();
    
    // Sort by score descending (worst first)
    samples.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Top 200
    let top_200 = samples.into_iter().take(200).collect::<Vec<_>>();

    // Report Generation
    let report_path = "DIRTY_SAMPLES.log";
    let mut file = File::create(report_path).expect("Could not create log file");
    
    writeln!(file, "DIRTY SAMPLES LOG - TOP 200 WORST OFFENDERS").unwrap();
    writeln!(file, "Generated at: {:?}", std::time::SystemTime::now()).unwrap();
    writeln!(file, "Total Files Scanned: {}", total_files.lock().unwrap()).unwrap();
    writeln!(file, "Total Lines Scanned: {}", *total_lines.lock().unwrap()).unwrap();
    
    let t_lines = *total_lines.lock().unwrap();
    let percentage = if t_lines > 0 { (total_detected as f64 / t_lines as f64) * 100.0 } else { 0.0 };

    writeln!(file, "Total Flags Raised: {} ({:.4}%)", total_detected, percentage).unwrap();
    writeln!(file, "==================================================").unwrap();

    for (i, sample) in top_200.iter().enumerate() {
        writeln!(file, "[#{}] Score: {:.2} | Reason: {}", i + 1, sample.score, sample.reason).unwrap();
        writeln!(file, "Line: {}", sample.line).unwrap();
        writeln!(file, "--------------------------------------------------").unwrap();
    }

    println!("📊 Report generated: {}", report_path);
    println!("   Total Flags: {}", total_detected);
    println!("   Top 1 offender score: {:.2}", top_200.first().map_or(0.0, |s| s.score));
}
