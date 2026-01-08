// src/data/dataset.rs

use memmap2::Mmap;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Dataset que usa memory-mapping
pub struct MmapDataset {
    data: Mmap,
    indices: Vec<usize>,
    seq_len: usize,
}

impl MmapDataset {
    pub fn from_file(path: &Path, seq_len: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let data = unsafe { Mmap::map(&file)? };
        
        let num_tokens = data.len() / 2;
        let num_sequences = num_tokens.saturating_sub(seq_len) / seq_len;
        
        let indices: Vec<usize> = (0..num_sequences)
            .map(|i| i * seq_len * 2)
            .collect();
        
        Ok(Self {
            data,
            indices,
            seq_len,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<(Vec<u16>, Vec<u16>)> {
        if idx >= self.indices.len() {
            return None;
        }

        let start = self.indices[idx];
        let end = start + (self.seq_len + 1) * 2;
        
        if end > self.data.len() {
            return None;
        }

        let bytes = &self.data[start..end];
        let tokens: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let input = tokens[..self.seq_len].to_vec();
        let target = tokens[1..=self.seq_len].to_vec();

        Some((input, target))
    }

    pub fn shuffle(&mut self, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.indices.shuffle(&mut rng);
    }
}

/// Batched iterator
pub struct DataLoader<'a> {
    dataset: &'a MmapDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a MmapDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Vec<Vec<u16>>, Vec<Vec<u16>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for idx in self.current_idx..end_idx {
            if let Some((input, target)) = self.dataset.get(idx) {
                inputs.push(input);
                targets.push(target);
            }
        }

        self.current_idx = end_idx;

        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }
}

/// Utilitário para criar arquivos binários
pub struct TokenizedDatasetWriter {
    writer: BufWriter<File>,
    tokens_written: usize,
}

impl TokenizedDatasetWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::with_capacity(1024 * 1024, file);
        
        Ok(Self {
            writer,
            tokens_written: 0,
        })
    }

    pub fn write_tokens(&mut self, tokens: &[u16]) -> std::io::Result<()> {
        for &token in tokens {
            self.writer.write_all(&token.to_le_bytes())?;
            self.tokens_written += 1;
        }
        Ok(())
    }

    pub fn finish(mut self) -> std::io::Result<usize> {
        self.writer.flush()?;
        Ok(self.tokens_written)
    }
}