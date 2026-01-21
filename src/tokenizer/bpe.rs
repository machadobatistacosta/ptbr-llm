use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;

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

struct LRUCache {
    map: HashMap<String, Vec<u16>>,
    order: VecDeque<String>,
    max_size: usize,
}

impl LRUCache {
    fn new(max_size: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn get(&self, key: &str) -> Option<Vec<u16>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: Vec<u16>) {
        if self.map.len() >= self.max_size {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }
}

/// Tokenizer BPE thread-safe
pub struct BPETokenizer {
    id_to_token: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u16>,
    merges: Vec<(u16, u16)>,
    special_tokens: HashMap<String, u16>,
    cache: Arc<RwLock<LRUCache>>,
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
        Self {
            id_to_token: vocab.id_to_token,
            token_to_id,
            merges: vocab.merges,
            special_tokens: vocab.special_tokens,
            cache: Arc::new(RwLock::new(LRUCache::new(MAX_CACHE_SIZE))),
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

    /// Encode thread-safe (não precisa de &mut self)
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();

        for word in self.pre_tokenize(text) {
            // Check cache
            {
                let cache = self.cache.read();
                if let Some(cached) = cache.get(&word) {
                    result.extend(cached);
                    continue;
                }
            }

            let tokens = self.encode_word(&word);

            // Insert cache
            {
                let mut cache = self.cache.write();
                cache.insert(word, tokens.clone());
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
                '.' | ',' | '!' | '?' | ':' | ';' | '(' | ')' | '"' | '\'' | '-' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(c.to_string());
                }
                _ => current.push(c),
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
            .map(|&b| self.token_to_id.get(&vec![b]).copied().unwrap_or(self.unk_id()))
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
            special_tokens: self.special_tokens.clone(),
            cache: Arc::new(RwLock::new(LRUCache::new(MAX_CACHE_SIZE))),
        }
    }
}

/// BPETrainer com suporte a special tokens dinâmicos (v2 - ChatML Ready)
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
            for word in text.split_whitespace() {
                if word.len() <= 100 {
                    *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                    total_words += 1;
                }
            }
        }

        println!("    Total palavras: {}", format_number(total_words));
        println!("    Palavras únicas: {}", format_number(word_freqs.len()));

        println!("  🔍 Fase 2: Filtrando (min_freq={})...", self.min_frequency);

        let filtered: HashMap<Vec<u8>, usize> = word_freqs
            .into_iter()
            .filter(|(_, freq)| *freq >= self.min_frequency)
            .map(|(word, freq)| {
                let mut bytes = vec![0xC4, 0xA0];
                bytes.extend(word.bytes());
                (bytes, freq)
            })
            .collect();

        println!("    Após filtro: {} palavras", format_number(filtered.len()));

        println!("  🔧 Fase 3: Treinando BPE...");

        let vocab = self.train_bpe(filtered);

        let elapsed = start.elapsed();
        println!("  ✅ Concluído em {:.1}s", elapsed.as_secs_f64());
        println!("    Vocab final: {} tokens", vocab.id_to_token.len());
        println!("    Merges: {}", vocab.merges.len());

        vocab
    }

    fn train_bpe(&self, word_freqs: HashMap<Vec<u8>, usize>) -> BPEVocab {
        let mut id_to_token: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut token_to_id: HashMap<Vec<u8>, u16> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u16))
            .collect();

        // ✨ Usa special_tokens dinâmicos ao invés de constantes hardcoded
        let mut special_map = HashMap::new();
        for token_str in &self.special_tokens {
            let id = id_to_token.len() as u16;
            let token_bytes = token_str.as_bytes().to_vec();
            id_to_token.push(token_bytes.clone());
            token_to_id.insert(token_bytes, id);
            special_map.insert(token_str.clone(), id);
        }

        let mut word_splits: Vec<(Vec<u16>, usize)> = word_freqs
            .iter()
            .map(|(word, &freq)| {
                let splits: Vec<u16> = word.iter().map(|&b| token_to_id[&vec![b]]).collect();
                (splits, freq)
            })
            .collect();

        let mut merges: Vec<(u16, u16)> = Vec::new();
        let target = self.vocab_size;
        let mut last_print = std::time::Instant::now();

        while id_to_token.len() < target {
            let pair_counts: HashMap<(u16, u16), usize> = word_splits
                .par_chunks(5000)
                .map(|chunk| {
                    let mut counts = HashMap::new();
                    for (splits, freq) in chunk {
                        for w in splits.windows(2) {
                            *counts.entry((w[0], w[1])).or_insert(0) += freq;
                        }
                    }
                    counts
                })
                .reduce(HashMap::new, |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                });

            if pair_counts.is_empty() {
                break;
            }

            let (best_pair, best_count) = pair_counts
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&p, &c)| (p, c))
                .unwrap();

            if best_count < self.min_frequency {
                break;
            }

            let (a, b) = best_pair;

            let mut new_token = Vec::new();
            new_token.extend(&id_to_token[a as usize]);
            new_token.extend(&id_to_token[b as usize]);

            let new_id = id_to_token.len() as u16;
            token_to_id.insert(new_token.clone(), new_id);
            id_to_token.push(new_token);
            merges.push(best_pair);

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

            if last_print.elapsed().as_secs() >= 3 {
                println!(
                    "    Vocab: {}/{} | Merges: {}",
                    id_to_token.len(),
                    target,
                    merges.len()
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

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}