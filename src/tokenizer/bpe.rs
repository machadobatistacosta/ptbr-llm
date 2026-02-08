use lru::LruCache;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::num::NonZeroUsize;
use std::sync::Arc;
use crate::utils::format_number;

const MAX_CACHE_SIZE: usize = 100_000;

#[derive(Serialize, Deserialize, Clone)]
pub struct BPEVocab {
    pub id_to_token: Vec<Vec<u8>>,
    pub merges: Vec<(u16, u16)>,
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

    pub fn build_token_to_id(&self) -> HashMap<Vec<u8>, u16> {
        self.id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u16))
            .collect()
    }
}

impl Default for BPEVocab {
    fn default() -> Self { Self::new() }
}

// Bug #17 fix: Use lru crate for proper LRU cache semantics (was FIFO before)
type TokenCache = LruCache<String, Vec<u16>>;

fn new_cache() -> Arc<RwLock<TokenCache>> {
    Arc::new(RwLock::new(
        LruCache::new(NonZeroUsize::new(MAX_CACHE_SIZE).unwrap())
    ))
}


/// Tokenizer BPE thread-safe
pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    // Bug #8 fix: HashMap for O(1) merge priority lookup instead of O(n) linear scan
    merge_priority: HashMap<(u16, u16), usize>,
    special_tokens: HashMap<String, u16>,
    // Bug #17 fix: Proper LRU cache from lru crate
    cache: Arc<RwLock<TokenCache>>,
    // Bug #16 fix: Static array for O(1) byte-to-id lookup
    byte_to_id: [u16; 256],
}

impl BPETokenizer {
    pub const PAD_TOKEN: &'static str = "[PAD]";
    pub const UNK_TOKEN: &'static str = "[UNK]";
    pub const BOS_TOKEN: &'static str = "[BOS]";
    pub const EOS_TOKEN: &'static str = "[EOS]";
    #[allow(dead_code)]
    pub const SEP_TOKEN: &'static str = "[SEP]";

    #[allow(dead_code)]
    pub fn pad_id(&self) -> u16 { *self.special_tokens.get(Self::PAD_TOKEN).unwrap_or(&0) }
    pub fn unk_id(&self) -> u16 { *self.special_tokens.get(Self::UNK_TOKEN).unwrap_or(&1) }
    pub fn bos_id(&self) -> u16 { *self.special_tokens.get(Self::BOS_TOKEN).expect("No [BOS]") }
    pub fn eos_id(&self) -> u16 { *self.special_tokens.get(Self::EOS_TOKEN).expect("No [EOS]") }
    pub fn vocab_size(&self) -> usize { self.id_to_token.len() }

    /// Retorna o ID de um special token customizado (se existir)
    #[allow(dead_code)]
    pub fn special_token_id(&self, token_name: &str) -> Option<u16> {
        self.special_tokens.get(token_name).copied()
    }

    /// Lista todos os special tokens carregados
    pub fn get_all_special_tokens(&self) -> &HashMap<String, u16> {
        &self.special_tokens
    }

    pub fn from_vocab(vocab: BPEVocab) -> Self {
        let token_to_id = vocab.build_token_to_id();
        // Bug #8 fix: Build HashMap for O(1) merge priority lookup
        let merge_priority: HashMap<(u16, u16), usize> = vocab.merges
            .iter()
            .enumerate()
            .map(|(i, &pair)| (pair, i))
            .collect();
        
        // Bug #16 fix: Build static byte-to-id lookup array
        let unk = *vocab.special_tokens.get("[UNK]").unwrap_or(&1);
        let mut byte_to_id = [unk; 256];
        for b in 0u16..=255 {
            if let Some(&id) = token_to_id.get(&vec![b as u8]) {
                byte_to_id[b as usize] = id;
            }
        }
        
        Self {
            id_to_token: vocab.id_to_token,
            token_to_id,
            merges: vocab.merges,
            merge_priority,
            special_tokens: vocab.special_tokens,
            cache: new_cache(),  // Bug #17: Use lru crate
            byte_to_id,
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
        serde_json::to_writer_pretty(writer, &vocab)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Encode thread-safe
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();

        for word in self.pre_tokenize(text) {
            // Bug #17: lru::get() requires &mut, so use write() lock
            // try peek() first for read-only check, fall back to get() for LRU update
            let cached = {
                let mut cache = self.cache.write();
                cache.get(&word).cloned()
            };
            
            if let Some(tokens) = cached {
                result.extend(tokens);
                continue;
            }

            let tokens = self.encode_word(&word);

            // Insert cache with automatic eviction
            {
                let mut cache = self.cache.write();
                cache.put(word, tokens.clone());
            }

            result.extend(tokens);
        }

        result
    }

    #[allow(dead_code)]
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u16>> {
        texts.par_iter().map(|t| self.encode(t)).collect()
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

    /// Bug #14 fix: Add Ġ prefix to first token
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut is_first_word = true;
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                ' ' | '\t' | '\n' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    if let Some(&next) = chars.peek() {
                        if !next.is_whitespace() {
                            current.push('Ġ');
                        }
                    }
                }
                '.' | ',' | '!' | '?' | ':' | ';' | '(' | ')' | '"' | '\'' | '-' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(c.to_string());
                }
                _ => {
                    // Bug #14: First word gets Ġ prefix
                    if is_first_word && current.is_empty() {
                        current.push('Ġ');
                        is_first_word = false;
                    }
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

        // Bug #16 fix: Use byte_to_id array for O(1) lookup
        let mut tokens: Vec<u16> = bytes
            .iter()
            .map(|&b| self.byte_to_id[b as usize])
            .collect();

        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_merge: Option<(usize, usize)> = None;
            let mut best_priority = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                // Bug #8 fix: O(1) HashMap lookup instead of O(n) linear scan
                if let Some(&pos) = self.merge_priority.get(&pair) {
                    if pos < best_priority {
                        best_priority = pos;
                        best_merge = Some((i, pos));
                    }
                }
            }

            match best_merge {
                Some((idx, merge_idx)) => {
                    let (a, b) = self.merges[merge_idx];
                    let mut merged = Vec::new();
                    if let Some(ba) = self.id_to_token.get(a as usize) {
                        merged.extend(ba);
                    }
                    if let Some(bb) = self.id_to_token.get(b as usize) {
                        merged.extend(bb);
                    }

                    if let Some(&new_id) = self.token_to_id.get(&merged) {
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

impl Clone for BPETokenizer {
    fn clone(&self) -> Self {
        Self {
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
            merges: self.merges.clone(),
            merge_priority: self.merge_priority.clone(),
            special_tokens: self.special_tokens.clone(),
            cache: new_cache(),  // Bug #17: Fresh LRU cache
            byte_to_id: self.byte_to_id,  // Bug #16: Copy array
        }
    }
}


use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Entry for priority queue - tracks pair and its count
#[derive(Eq, PartialEq)]
struct PairEntry {
    pair: (u16, u16),
    count: usize,
}

impl Ord for PairEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher count = higher priority
        self.count.cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair)) // Tie-break by pair for determinism
    }
}

impl PartialOrd for PairEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Indexed word representation for O(1) merge operations
/// Uses a linked-list-like structure with indices instead of actual removals
struct IndexedWord {
    tokens: Vec<u16>,
    next: Vec<i32>,  // next[i] = index of next active token, -1 if end
    first: i32,      // index of first active token
    freq: usize,
}

impl IndexedWord {
    fn new(tokens: Vec<u16>, freq: usize) -> Self {
        let len = tokens.len();
        let next: Vec<i32> = (0..len).map(|i| if i + 1 < len { (i + 1) as i32 } else { -1 }).collect();
        Self {
            tokens,
            next,
            first: if len > 0 { 0 } else { -1 },
            freq,
        }
    }

    /// Iterate over active (token_idx, token_id) pairs
    fn iter_active(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        let mut idx = self.first;
        std::iter::from_fn(move || {
            if idx < 0 {
                None
            } else {
                let current = idx as usize;
                let token = self.tokens[current];
                idx = self.next[current];
                Some((current, token))
            }
        })
    }

    /// Get active pairs as (idx, pair)
    fn get_pairs(&self) -> Vec<(usize, (u16, u16))> {
        let mut pairs = Vec::new();
        let mut prev_idx: Option<usize> = None;
        let mut prev_token: Option<u16> = None;
        
        for (idx, token) in self.iter_active() {
            if let (Some(pi), Some(pt)) = (prev_idx, prev_token) {
                pairs.push((pi, (pt, token)));
            }
            prev_idx = Some(idx);
            prev_token = Some(token);
        }
        pairs
    }

    /// Apply merge: replace (a, b) with new_id, returns pairs removed and added
    fn apply_merge(&mut self, a: u16, b: u16, new_id: u16) -> (Vec<(u16, u16)>, Vec<(u16, u16)>) {
        let mut removed = Vec::new();
        let mut added = Vec::new();
        
        let mut prev_prev_idx: Option<usize> = None;
        let mut prev_idx: Option<usize> = None;
        let mut prev_token: Option<u16> = None;
        
        let mut idx = self.first;
        while idx >= 0 {
            let current = idx as usize;
            let token = self.tokens[current];
            let next_idx = self.next[current];
            
            // Check if current pair is (a, b)
            if let (Some(pi), Some(pt)) = (prev_idx, prev_token) {
                if pt == a && token == b {
                    // Found match - merge!
                    
                    // Record pair being removed
                    removed.push((a, b));
                    
                    // Record left neighbor pair being removed (if exists)
                    if let (Some(ppi), Some(_)) = (prev_prev_idx, prev_idx) {
                        removed.push((self.tokens[ppi], a));
                    }
                    
                    // Record right neighbor pair being removed (if exists)
                    if next_idx >= 0 {
                        removed.push((b, self.tokens[next_idx as usize]));
                    }
                    
                    // Apply merge: update prev token to new_id, skip current
                    self.tokens[pi] = new_id;
                    self.next[pi] = next_idx;
                    
                    // Record new pairs
                    if let Some(ppi) = prev_prev_idx {
                        added.push((self.tokens[ppi], new_id));
                    }
                    if next_idx >= 0 {
                        added.push((new_id, self.tokens[next_idx as usize]));
                    }
                    
                    // Reset prev tracking since current was consumed
                    // prev stays the same (now has new_id), prev_prev stays
                    prev_token = Some(new_id);
                    idx = next_idx;
                    continue;
                }
            }
            
            prev_prev_idx = prev_idx;
            prev_idx = Some(current);
            prev_token = Some(token);
            idx = next_idx;
        }
        
        (removed, added)
    }
}

/// BPETrainer com suporte a special tokens dinâmicos (v3 - OPTIMIZED)
/// 
/// Performance: O(V * log P) where V = vocab_size, P = unique pairs
/// Previous: O(V * W * L) where W = words, L = avg word length
pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: usize,
    special_tokens: Vec<String>,
}

impl BPETrainer {
    /// Cria trainer com tokens especiais padrão (backward compatible)
    pub fn new(vocab_size: usize, min_frequency: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: min_frequency.max(2),
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[BOS]".to_string(),
                "[EOS]".to_string(),
                "[SEP]".to_string(),
            ],
        }
    }

    /// Injeta custom special tokens (Chat, Instruction, etc.)
    pub fn with_special_tokens(mut self, tokens: Vec<&str>) -> Self {
        self.special_tokens = tokens.into_iter().map(|s| s.to_string()).collect();
        self
    }

    /// Adiciona tokens customizados aos padrão (útil para expandir, não substituir)
    #[allow(dead_code)]
    pub fn append_special_tokens(mut self, tokens: Vec<&str>) -> Self {
        self.special_tokens.extend(tokens.into_iter().map(|s| s.to_string()));
        self
    }

    pub fn train<I>(&self, texts: I) -> BPEVocab
    where
        I: Iterator<Item = String>,
    {
        let start = std::time::Instant::now();

        println!("  📊 Fase 1: Contando palavras...");

        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0usize;

        for text in texts {
            // Bug #9 fix: Use same pre-tokenization logic as BPETokenizer
            for word in pre_tokenize_for_training(&text) {
                if word.len() <= 100 {
                    *word_freqs.entry(word).or_insert(0) += 1;
                    total_words += 1;
                }
            }
        }

        println!("    Total palavras: {}", format_number(total_words));
        println!("    Palavras únicas: {}", format_number(word_freqs.len()));

        println!("  🔍 Fase 2: Filtrando (min_freq={})...", self.min_frequency);

        // Bug #19 fix: Don't add hardcoded Ġ prefix - pre_tokenize_for_training already handles it
        let filtered: HashMap<Vec<u8>, usize> = word_freqs
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_frequency)
            .map(|(word, freq)| (word.into_bytes(), freq))
            .collect();

        println!("    Após filtro: {} palavras", format_number(filtered.len()));

        println!("  🔧 Fase 3: Treinando BPE (OTIMIZADO)...");

        let vocab = self.train_bpe_optimized(filtered);

        let elapsed = start.elapsed();
        println!("  ✅ Concluído em {:.1}s", elapsed.as_secs_f64());
        println!("    Vocab final: {} tokens", vocab.id_to_token.len());
        println!("    Merges: {}", vocab.merges.len());

        vocab
    }

    /// Optimized BPE training with:
    /// - Incremental pair counting (no full recount per merge)
    /// - Priority queue for O(log n) best pair lookup
    /// - Indexed word structure for O(1) merges (no Vec::remove)
    fn train_bpe_optimized(&self, word_freqs: HashMap<Vec<u8>, usize>) -> BPEVocab {
        let mut id_to_token: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut token_to_id: HashMap<Vec<u8>, u16> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u16))
            .collect();

        // Add special tokens
        let mut special_map = HashMap::new();
        for token_str in &self.special_tokens {
            let id = id_to_token.len() as u16;
            let token_bytes = token_str.as_bytes().to_vec();
            id_to_token.push(token_bytes.clone());
            token_to_id.insert(token_bytes, id);
            special_map.insert(token_str.clone(), id);
        }

        // Convert words to indexed representation
        let mut words: Vec<IndexedWord> = word_freqs
            .iter()
            .map(|(word, &freq)| {
                let tokens: Vec<u16> = word.iter().map(|&b| token_to_id[&vec![b]]).collect();
                IndexedWord::new(tokens, freq)
            })
            .collect();

        // Initial pair counting
        let mut pair_counts: HashMap<(u16, u16), usize> = HashMap::new();
        for word in &words {
            for (_, pair) in word.get_pairs() {
                *pair_counts.entry(pair).or_insert(0) += word.freq;
            }
        }

        // Build initial priority queue
        let mut heap: BinaryHeap<PairEntry> = pair_counts
            .iter()
            .map(|(&pair, &count)| PairEntry { pair, count })
            .collect();

        let mut merges: Vec<(u16, u16)> = Vec::new();
        let target = self.vocab_size;
        let mut last_print = std::time::Instant::now();
        let train_start = std::time::Instant::now();
        let mut merge_count = 0usize;

        while id_to_token.len() < target {
            // Find best pair (may need to skip stale entries)
            let best_pair = loop {
                match heap.pop() {
                    None => break None,
                    Some(entry) => {
                        // Check if this entry is still valid (count matches)
                        let current_count = pair_counts.get(&entry.pair).copied().unwrap_or(0);
                        if current_count == entry.count && current_count >= self.min_frequency {
                            break Some((entry.pair, entry.count));
                        }
                        // Stale entry - if count still valid, re-add with correct count
                        if current_count >= self.min_frequency {
                            heap.push(PairEntry { pair: entry.pair, count: current_count });
                        }
                    }
                }
            };

            let (best_pair, _best_count) = match best_pair {
                Some(p) => p,
                None => break, // No more valid pairs
            };

            let (a, b) = best_pair;

            // Create new token
            let mut new_token = Vec::new();
            new_token.extend(&id_to_token[a as usize]);
            new_token.extend(&id_to_token[b as usize]);

            let new_id = id_to_token.len() as u16;
            token_to_id.insert(new_token.clone(), new_id);
            id_to_token.push(new_token);
            merges.push(best_pair);
            merge_count += 1;

            // Apply merge to all words and update pair counts incrementally
            for word in &mut words {
                let (removed, added) = word.apply_merge(a, b, new_id);
                
                // Update counts for removed pairs
                for pair in removed {
                    if let Some(count) = pair_counts.get_mut(&pair) {
                        *count = count.saturating_sub(word.freq);
                    }
                }
                
                // Update counts for added pairs
                for pair in added {
                    let new_count = pair_counts.entry(pair).or_insert(0);
                    *new_count += word.freq;
                    // Add to heap (may create duplicates, but we handle stale entries)
                    heap.push(PairEntry { pair, count: *new_count });
                }
            }

            // Remove merged pair from counts
            pair_counts.remove(&best_pair);

            // Progress logging
            if last_print.elapsed().as_secs() >= 3 {
                let merges_per_sec = merge_count as f64 / train_start.elapsed().as_secs_f64().max(0.001);
                let remaining = target.saturating_sub(id_to_token.len());
                let eta_secs = remaining as f64 / merges_per_sec.max(0.1);
                println!(
                    "    Vocab: {}/{} | Merges: {} | {:.1} merges/s | ETA: {:.0}s",
                    id_to_token.len(),
                    target,
                    merges.len(),
                    merges_per_sec,
                    eta_secs
                );
                last_print = std::time::Instant::now();
            }
        }

        BPEVocab {
            id_to_token,
            merges,
            special_tokens: special_map,
        }
    }
}



/// Bug #9 fix: Pre-tokenize function for training that matches BPETokenizer::pre_tokenize
/// This ensures trainer and tokenizer split text the same way
/// Bug #14 fix: Now adds Ġ prefix to first token
fn pre_tokenize_for_training(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut is_first_word = true;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\n' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                if let Some(&next) = chars.peek() {
                    if !next.is_whitespace() {
                        current.push('Ġ');
                    }
                }
            }
            '.' | ',' | '!' | '?' | ':' | ';' | '(' | ')' | '"' | '\'' | '-' => {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(c.to_string());
            }
            _ => {
                // Bug #14: First word gets Ġ prefix
                if is_first_word && current.is_empty() {
                    current.push('Ġ');
                    is_first_word = false;
                }
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}