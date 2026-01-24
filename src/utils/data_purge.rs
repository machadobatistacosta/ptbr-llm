use ptbr_llm::WikiCleaner;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use rand::Rng;

fn main() {
    let start_time = Instant::now();
    println!("🧹 Balanced Soberania Purge Tool v1.0");
    println!("========================================");

    let data_dir = "data/tokenizer_full_input_cleaned";
    println!("📂 Target directory: {}", data_dir);

    let total_preserved = AtomicUsize::new(0);
    let total_deleted_entropy = AtomicUsize::new(0);
    let total_deleted_garbage = AtomicUsize::new(0);

    // Listar arquivos
    let paths: Vec<_> = std::fs::read_dir(data_dir)
        .expect("Could not read directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().map_or(false, |ext| ext == "txt"))
        .collect();

    println!("📜 Found {} text files to process.", paths.len());

    paths.par_iter().for_each(|path| {
        // Read Phase
        let file = File::open(path).expect("Cannot open file");
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
        
        // Process Phase
        let mut new_content = Vec::with_capacity(lines.len());
        let mut local_garbage = 0;
        let mut local_entropy = 0;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }

            // Audit Logic
            if let Some(sample) = WikiCleaner::audit_line(trimmed) {
                // Rule 1: CRITICAL GARBAGE (100% DELETE)
                let is_garbage = sample.reason.contains("Ghost BOM") 
                              || sample.reason.contains("Malformed")
                              || sample.reason.contains("Caps Lock Abuse"); // Caps lock usually noise too

                if is_garbage {
                    local_garbage += 1;
                    continue; 
                }

                // Rule 2: HIGH ENTROPY / FOREIGN (95% DELETE)
                if sample.score > 7.0 {
                     let mut rng = rand::thread_rng();
                     if rng.gen_bool(0.05) {
                         // Keep 5% for Diversity (The Model needs to know what Japanese looks like, just not speak it)
                         new_content.push(line); 
                     } else {
                         local_entropy += 1;
                     }
                     continue;
                }
                
                // Rule 3: MILD DIRT (KEEP)
                // Score between 0.5 and 7.0 (e.g. lots of symbols but valid PT)
                new_content.push(line);

            } else {
                // CLEAN LINE (KEEP)
                new_content.push(line);
            }
        }

        // Write Phase (Atomic replace simulation)
        // We overwrite the file directly for simplicity as we have memory copy
        if let Ok(mut file) = File::create(path) {
            for line in &new_content {
                writeln!(file, "{}", line).unwrap();
            }
        }

        total_preserved.fetch_add(new_content.len(), Ordering::Relaxed);
        total_deleted_garbage.fetch_add(local_garbage, Ordering::Relaxed);
        total_deleted_entropy.fetch_add(local_entropy, Ordering::Relaxed);
        
        print!(".");
        std::io::stdout().flush().unwrap();
    });

    println!("\n\n✅ Purge Complete in {:.2?}s", start_time.elapsed());
    println!("--------------------------------------------------");
    println!("Total Lines Preserved: {}", total_preserved.load(Ordering::Relaxed));
    println!("Deleted (Garbage):     {}", total_deleted_garbage.load(Ordering::Relaxed));
    println!("Deleted (Foreign/Hi):  {}", total_deleted_entropy.load(Ordering::Relaxed));
    println!("--------------------------------------------------");
}
