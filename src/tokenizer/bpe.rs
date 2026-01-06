// src/tokenizer/bpe.rs

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

/// Estrutura serializável para JSON
#[derive(Serialize, Deserialize, Clone)]
pub struct BPEVocab {
    /// Lista de tokens (índice = id)
    pub id_to_token: Vec<Vec<u8>>,
    /// Merges na ordem em que foram feitos
    pub merges: Vec<(u16, u16)>,
    /// Special tokens
    pub special_tokens: HashMap<String, u16>,
}

impl BPEVocab {
    pub fn new() -> Self {
        Self {
            id_to_token: Vec::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
        }
    }
    
    /// Reconstrói token_to_id a partir de id_to_token
    pub fn build_token_to_id(&self) -> HashMap<Vec<u8>, u16> {
        self.id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u16))
            .collect()
    }
}

impl Default for BPEVocab {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    special_tokens: HashMap<String, u16>,
    cache: HashMap<String, Vec<u16>>,
}

impl BPETokenizer {
    pub const PAD_TOKEN: &'static str = "[PAD]";
    pub const UNK_TOKEN: &'static str = "[UNK]";
    pub const BOS_TOKEN: &'static str = "[BOS]";
    pub const EOS_TOKEN: &'static str = "[EOS]";
    pub const SEP_TOKEN: &'static str = "[SEP]";
    
    pub fn pad_id(&self) -> u16 {
        *self.special_tokens.get(Self::PAD_TOKEN).unwrap_or(&0)
    }
    
    pub fn unk_id(&self) -> u16 {
        *self.special_tokens.get(Self::UNK_TOKEN).unwrap_or(&1)
    }
    
    pub fn bos_id(&self) -> u16 {
        *self.special_tokens.get(Self::BOS_TOKEN).unwrap_or(&2)
    }
    
    pub fn eos_id(&self) -> u16 {
        *self.special_tokens.get(Self::EOS_TOKEN).unwrap_or(&3)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn from_vocab(vocab: BPEVocab) -> Self {
        let token_to_id = vocab.build_token_to_id();
        Self {
            id_to_token: vocab.id_to_token,
            token_to_id,
            merges: vocab.merges,
            special_tokens: vocab.special_tokens,
            cache: HashMap::with_capacity(10_000),
        }
    }

    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let vocab: BPEVocab = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        Ok(Self::from_vocab(vocab))
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let vocab = BPEVocab {
            id_to_token: self.id_to_token.clone(),
            merges: self.merges.clone(),
            special_tokens: self.special_tokens.clone(),
        };
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &vocab)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    pub fn encode(&mut self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();
        
        for word in self.pre_tokenize(text) {
            if let Some(cached) = self.cache.get(&word) {
                result.extend(cached.clone());
                continue;
            }
            
            let tokens = self.encode_word(&word);
            
            if word.len() < 20 {
                self.cache.insert(word.clone(), tokens.clone());
            }
            
            result.extend(tokens);
        }
        
        result
    }

    pub fn decode(&self, ids: &[u16]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize))
            .flatten()
            .copied()
            .collect();
        
        String::from_utf8_lossy(&bytes)
            .replace("Ġ", " ")
            .to_string()
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                ' ' | '\t' | '\n' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    if let Some(&next) = chars.peek() {
                        if next.is_alphabetic() {
                            current.push('Ġ');
                        }
                    }
                }
                '.' | ',' | '!' | '?' | ':' | ';' | '(' | ')' | '"' | '\'' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(c.to_string());
                }
                _ => {
                    current.push(c);
                }
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    fn encode_word(&self, word: &str) -> Vec<u16> {
        let bytes: Vec<u8> = word.bytes().collect();
        
        if bytes.is_empty() {
            return Vec::new();
        }

        let mut tokens: Vec<u16> = bytes
            .iter()
            .map(|&b| {
                self.token_to_id
                    .get(&vec![b])
                    .copied()
                    .unwrap_or(self.unk_id())
            })
            .collect();

        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_merge: Option<(usize, usize)> = None;
            let mut best_priority = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                
                if let Some(pos) = self.merges.iter().position(|m| *m == pair) {
                    if pos < best_priority {
                        best_priority = pos;
                        best_merge = Some((i, pos));
                    }
                }
            }

            match best_merge {
                Some((idx, merge_idx)) => {
                    let (a, b) = self.merges[merge_idx];
                    let mut merged_bytes = Vec::new();
                    if let Some(bytes_a) = self.id_to_token.get(a as usize) {
                        merged_bytes.extend(bytes_a);
                    }
                    if let Some(bytes_b) = self.id_to_token.get(b as usize) {
                        merged_bytes.extend(bytes_b);
                    }
                    
                    if let Some(&new_id) = self.token_to_id.get(&merged_bytes) {
                        tokens[idx] = new_id;
                        tokens.remove(idx + 1);
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }

        tokens
    }
}

/// Trainer BPE OTIMIZADO com paralelismo
pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: usize,
}

impl BPETrainer {
    pub fn new(vocab_size: usize, min_frequency: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: min_frequency.max(5),
        }
    }

    pub fn train<I>(&self, texts: I) -> BPEVocab
    where
        I: Iterator<Item = String>,
    {
        let start_total = std::time::Instant::now();
        
        // ============ FASE 1: Contar palavras ============
        println!("═══════════════════════════════════════════════════════════");
        println!("  FASE 1/3: Contando palavras");
        println!("═══════════════════════════════════════════════════════════");
        
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0usize;
        let mut last_print = std::time::Instant::now();
        
        for text in texts {
            for word in text.split_whitespace() {
                if word.len() <= 50 {
                    *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                    total_words += 1;
                }
                
                if last_print.elapsed().as_secs() >= 5 {
                    println!("  {} milhões de palavras...", total_words / 1_000_000);
                    last_print = std::time::Instant::now();
                }
            }
        }

        println!("  ✓ Total: {} palavras", format_number(total_words));
        println!("  ✓ Únicas: {}", format_number(word_freqs.len()));
        
        // ============ FASE 2: Filtrar por frequência ============
        println!("\n═══════════════════════════════════════════════════════════");
        println!("  FASE 2/3: Filtrando palavras (freq >= {})", self.min_frequency);
        println!("═══════════════════════════════════════════════════════════");
        
        let filtered: HashMap<Vec<u8>, usize> = word_freqs
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_frequency)
            .map(|(word, freq)| {
                let mut bytes = vec![0xC4, 0xA0];  // 'Ġ'
                bytes.extend(word.bytes());
                (bytes, freq)
            })
            .collect();
        
        let total_freq: usize = filtered.values().sum();
        println!("  ✓ Palavras após filtro: {}", format_number(filtered.len()));
        println!("  ✓ Cobertura: {:.2}% do corpus", 
            (total_freq as f64 / total_words as f64) * 100.0);
        
        // ============ FASE 3: BPE Merges ============
        println!("\n═══════════════════════════════════════════════════════════");
        println!("  FASE 3/3: Treinando BPE ({} tokens)", self.vocab_size);
        println!("═══════════════════════════════════════════════════════════");
        
        let vocab = self.train_bpe(filtered);
        
        let elapsed = start_total.elapsed();
        println!("\n═══════════════════════════════════════════════════════════");
        println!("  ✓ CONCLUÍDO em {}m {}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60);
        println!("  ✓ Vocabulário final: {} tokens", vocab.id_to_token.len());
        println!("  ✓ Merges realizados: {}", vocab.merges.len());
        println!("═══════════════════════════════════════════════════════════");
        
        vocab
    }

    fn train_bpe(&self, word_freqs: HashMap<Vec<u8>, usize>) -> BPEVocab {
        // Inicializa vocabulário com bytes (0-255)
        let mut id_to_token: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut token_to_id: HashMap<Vec<u8>, u16> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u16))
            .collect();

        // Special tokens
        let special_tokens_list = [
            BPETokenizer::PAD_TOKEN,
            BPETokenizer::UNK_TOKEN,
            BPETokenizer::BOS_TOKEN,
            BPETokenizer::EOS_TOKEN,
            BPETokenizer::SEP_TOKEN,
        ];

        let mut special_map = HashMap::new();
        for token in special_tokens_list {
            let id = id_to_token.len() as u16;
            id_to_token.push(token.as_bytes().to_vec());
            token_to_id.insert(token.as_bytes().to_vec(), id);
            special_map.insert(token.to_string(), id);
        }

        // Converte palavras para sequências de IDs
        let mut word_splits: Vec<(Vec<u16>, usize)> = word_freqs
            .iter()
            .map(|(word, &freq)| {
                let splits: Vec<u16> = word
                    .iter()
                    .map(|&b| token_to_id[&vec![b]])
                    .collect();
                (splits, freq)
            })
            .collect();

        let mut merges: Vec<(u16, u16)> = Vec::new();
        let start = std::time::Instant::now();
        let base_vocab_size = id_to_token.len();
        let target_merges = self.vocab_size - base_vocab_size;
        
        println!("  Base vocab: {} | Target: {} | Merges needed: {}", 
            base_vocab_size, self.vocab_size, target_merges);
        println!();

        let mut last_print = std::time::Instant::now();
        
        while id_to_token.len() < self.vocab_size {
            // Conta pares em paralelo
            let pair_counts = self.count_pairs_parallel(&word_splits);
            
            if pair_counts.is_empty() {
                println!("\n  ⚠ Sem mais pares válidos. Parando.");
                break;
            }

            // Encontra melhor par
            let (best_pair, best_count) = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&pair, &count)| (pair, count))
                .unwrap();

            if best_count < self.min_frequency {
                println!("\n  ⚠ Melhor par abaixo da frequência mínima. Parando.");
                break;
            }

            let (a, b) = best_pair;

            // Cria novo token
            let mut new_token = Vec::new();
            new_token.extend(&id_to_token[a as usize]);
            new_token.extend(&id_to_token[b as usize]);
            
            let new_id = id_to_token.len() as u16;
            token_to_id.insert(new_token.clone(), new_id);
            id_to_token.push(new_token);
            merges.push(best_pair);

            // Atualiza splits em paralelo
            word_splits.par_iter_mut().for_each(|(splits, _)| {
                let mut i = 0;
                while i < splits.len().saturating_sub(1) {
                    if splits[i] == a && splits[i + 1] == b {
                        splits[i] = new_id;
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            });

            // Progress a cada 2 segundos
            let current_merges = merges.len();
            if last_print.elapsed().as_secs() >= 2 {
                let elapsed = start.elapsed().as_secs();
                let rate = current_merges as f64 / elapsed.max(1) as f64;
                let remaining = target_merges.saturating_sub(current_merges);
                let eta_secs = (remaining as f64 / rate.max(0.1)) as u64;
                
                // Preview seguro para UTF-8
                let token_bytes = &id_to_token[new_id as usize];
                let token_preview = String::from_utf8_lossy(token_bytes);
                let preview: String = token_preview
                    .chars()
                    .take(12)
                    .collect::<String>()
                    .replace('\n', "\\n");
                
                print!("\r  Vocab: {}/{} | Freq: {} | Token: {:15} | {:.1}/s | ETA: {}m{}s    ",
                    id_to_token.len(),
                    self.vocab_size,
                    format_number(best_count),
                    format!("\"{}\"", preview),
                    rate,
                    eta_secs / 60,
                    eta_secs % 60
                );
                std::io::Write::flush(&mut std::io::stdout()).ok();
                
                last_print = std::time::Instant::now();
            }
        }

        println!(); // Nova linha após progress

        BPEVocab {
            id_to_token,
            merges,
            special_tokens: special_map,
        }
    }

    fn count_pairs_parallel(&self, word_splits: &[(Vec<u16>, usize)]) -> HashMap<(u16, u16), usize> {
        let chunk_counts: Vec<HashMap<(u16, u16), usize>> = word_splits
            .par_chunks(10_000)
            .map(|chunk| {
                let mut counts = HashMap::new();
                for (splits, freq) in chunk {
                    for window in splits.windows(2) {
                        let pair = (window[0], window[1]);
                        *counts.entry(pair).or_insert(0) += freq;
                    }
                }
                counts
            })
            .collect();

        let mut total_counts = HashMap::new();
        for chunk_count in chunk_counts {
            for (pair, count) in chunk_count {
                *total_counts.entry(pair).or_insert(0) += count;
            }
        }

        total_counts
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}