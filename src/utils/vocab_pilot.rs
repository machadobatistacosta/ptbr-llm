use ptbr_llm::{tokenizer::{BPETrainer, BPETokenizer, PTBRNormalizer}, WikiCleaner};
use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::prelude::*;

fn main() {
    println!("🧪 VOCAB PILOT: 100k Line Sample Test");
    println!("=======================================");

    let data_dir = "data/tokenizer_full_input_cleaned";
    let sample_size = 20_000;
    
    // 1. Gather files
    let paths: Vec<_> = std::fs::read_dir(data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "txt"))
        .collect();
    
    // 2. Sample lines
    println!("  Sampling {} lines from {} files...", sample_size, paths.len());
    let mut rng = rand::thread_rng();
    let mut sampled_lines = Vec::with_capacity(sample_size);
    // total_files_used removed

    let cleaner = WikiCleaner::new();
    let normalizer = PTBRNormalizer::new();

    // Shuffle files to be random
    let mut shuffled_paths = paths.clone();
    shuffled_paths.shuffle(&mut rng);

    for path in shuffled_paths {
        if sampled_lines.len() >= sample_size { break; }
        
        let file = File::open(path).expect("Cannot open file");
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            if let Ok(l) = line {
                if l.trim().len() > 20 {
                    // Apply Pipeline: Planalto Fix -> Normalizer
                    let fixed = cleaner.fix_planalto(&l);
                    let normalized = normalizer.normalize(&fixed);
                    
                    if rng.gen_bool(0.1) { // 10% chance to take line to spread across files
                        sampled_lines.push(normalized);
                    }
                }
            }
            if sampled_lines.len() >= sample_size { break; }
        }
        // total_files_used increment removed
    }

    println!("  Collected {} lines.", sampled_lines.len());

    // 3. Train Tokenizer
    println!("  Training BPE (20000 tokens)... [This may take a minute]");
    let trainer = BPETrainer::new(20000, 5);
    let vocab = trainer.train(sampled_lines.into_iter());
    let tokenizer = BPETokenizer::from_vocab(vocab);

    // 4. Stats
    println!("\n  📊 STATS:");
    println!("  - Vocab Size: {}", tokenizer.vocab_size());
    
    println!("\n  🔥 TOP 50 Frequent Tokens (Sample):");
    // Since we don't have frequency sorted in tokenizer struct directly (it's lost after map build usually),
    // we iterate id 0..50 assuming low IDs are special/frequent or check BPE merge order.
    // Actually, common tokens usually end up with lower IDs in some BPE impls, or we can just print random ones.
    // Better: Decode ID 256 to 300 (ASCII usually 0-255).
    for i in 256..306 {
        if i < tokenizer.vocab_size() {
             print!("|{}| ", tokenizer.decode(&[i as u16]));
        }
    }
    println!("\n");

    // 5. Long Token Check
    println!("  🦒 Long Token Samples (>10 chars):");
    let mut long_tokens_found = 0;
    // Iterate from end (more complex tokens usually later)
    for i in (tokenizer.vocab_size()-5000..tokenizer.vocab_size()).rev() {
        let text = tokenizer.decode(&[i as u16]);
        if text.len() > 10 {
            println!("  - [{}] '{}'", i, text);
            long_tokens_found += 1;
            if long_tokens_found >= 10 { break; }
        }
    }

    // 6. Validation Test
    println!("\n  ⚖️  VALIDATION TEST:");
    let text = "De acordo com o Art. 1º da Constituição Federal, a soberania é inalienável.";
    
    // Manual fix/norm mostly happens in dataset loader, but verify if tokenizer handles raw:
    let fixed = cleaner.fix_planalto(text);
    let normalized = normalizer.normalize(&fixed);
    
    let tokens = tokenizer.encode(&normalized);
    let decoded = tokenizer.decode(&tokens);
    
    println!("  Input:    {}", text);
    println!("  Norm:     {}", normalized);
    println!("  Encoded:  {:?}", tokens);
    println!("  Decoded:  {}", decoded); // Should match normalized exactly

    let unk_id = tokenizer.unk_id();
    if tokens.contains(&unk_id) {
        println!("  ❌ FAILED: Contains [UNK]!");
        std::process::exit(1);
    }
    if decoded != normalized {
         println!("  ❌ FAILED: Roundtrip mismatch!");
         std::process::exit(1);
    }

    println!("  ✅ TEST PASSED!");
}
