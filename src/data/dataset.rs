use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Erros do dataset
#[derive(Debug)]
pub enum DatasetError {
    IoError(std::io::Error),
    InvalidFormat(String),
    TooSmall { tokens: usize, required: usize },
}

impl From<std::io::Error> for DatasetError {
    fn from(e: std::io::Error) -> Self {
        DatasetError::IoError(e)
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::IoError(e) => write!(f, "IO Error: {}", e),
            DatasetError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            DatasetError::TooSmall { tokens, required } => {
                write!(f, "Dataset too small: {} tokens, need at least {}", tokens, required)
            }
        }
    }
}

impl std::error::Error for DatasetError {}

pub struct MmapDataset {
    data: Mmap,
    indices: Vec<usize>,
    seq_len: usize,
    epoch: usize,
    num_tokens: usize,
}

impl MmapDataset {
    pub fn from_file(path: &Path, seq_len: usize) -> Result<Self, DatasetError> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // CRITICAL FIX: Dataset format includes 8-byte header
        const HEADER_SIZE: usize = 8;
        
        if file_size < HEADER_SIZE {
            return Err(DatasetError::InvalidFormat(
                "File too small to contain header".into(),
            ));
        }

        let data = unsafe { Mmap::map(&file)? };
        
        // CRITICAL FIX: Skip header, only count actual token data
        let data_size = file_size - HEADER_SIZE;
        
        if data_size % 2 != 0 {
            return Err(DatasetError::InvalidFormat(
                "Data size not multiple of 2 (expected u16 tokens)".into(),
            ));
        }

        let num_tokens = data_size / 2;
        let required_tokens = seq_len + 2;

        if num_tokens < required_tokens {
            return Err(DatasetError::TooSmall {
                tokens: num_tokens,
                required: required_tokens,
            });
        }

        let num_sequences = (num_tokens - seq_len - 1) / seq_len;

        if num_sequences == 0 {
            return Err(DatasetError::TooSmall {
                tokens: num_tokens,
                required: seq_len * 2 + 1,
            });
        }

        // CRITICAL FIX: Indices start AFTER the 8-byte header
        let indices: Vec<usize> = (0..num_sequences)
            .map(|i| HEADER_SIZE + i * seq_len * 2)
            .collect();

        println!(
            "  ✓ Dataset: {} tokens, {} sequências (seq_len={})",
            format_number(num_tokens),
            format_number(num_sequences),
            seq_len
        );

        Ok(Self {
            data,
            indices,
            seq_len,
            epoch: 0,
            num_tokens,
        })
    }

    pub fn len(&self) -> usize { self.indices.len() }
    pub fn is_empty(&self) -> bool { self.indices.is_empty() }
    pub fn num_tokens(&self) -> usize { self.num_tokens }
    #[allow(dead_code)]
    pub fn epoch(&self) -> usize { self.epoch }
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize { self.seq_len }

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

    pub fn shuffle(&mut self, base_seed: u64) {
        let seed = base_seed.wrapping_add(self.epoch as u64);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.indices.shuffle(&mut rng);
    }

    pub fn next_epoch(&mut self) {
        self.epoch += 1;
    }
}

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
    pub fn reset(&mut self) { self.current_idx = 0; }

    #[allow(dead_code)]
    pub fn remaining(&self) -> usize {
        self.dataset.len().saturating_sub(self.current_idx)
    }

    #[allow(dead_code)]
    pub fn total_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Vec<Vec<u16>>, Vec<Vec<u16>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let indices: Vec<usize> = (self.current_idx..end_idx).collect();
        
        // Parallel batch loading using Rayon
        let results: Vec<_> = indices.par_iter()
            .filter_map(|&idx| self.dataset.get(idx))
            .collect();

        self.current_idx = end_idx;

        if results.is_empty() { 
            None 
        } else { 
            let (inputs, targets): (Vec<_>, Vec<_>) = results.into_iter().unzip();
            Some((inputs, targets))
        }
    }
}

pub struct TokenizedDatasetWriter {
    writer: BufWriter<File>,
    tokens_written: usize,
    #[allow(dead_code)]
    header_written: bool,
}

impl TokenizedDatasetWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
        
        // CRITICAL FIX: Write placeholder header (will be updated in finish())
        // Header format: u64 (8 bytes) containing total token count
        writer.write_all(&0u64.to_le_bytes())?;
        
        Ok(Self {
            writer,
            tokens_written: 0,
            header_written: true,
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
        use std::io::Seek;
        
        self.writer.flush()?;
        
        // CRITICAL FIX: Update header with actual token count
        let mut file = self.writer.into_inner()?;
        file.seek(std::io::SeekFrom::Start(0))?;
        file.write_all(&(self.tokens_written as u64).to_le_bytes())?;
        file.flush()?;
        
        Ok(self.tokens_written)
    }

    #[allow(dead_code)]
    pub fn tokens_written(&self) -> usize { self.tokens_written }
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