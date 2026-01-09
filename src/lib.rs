//! PTBR-SLM: Small Language Model para Português Brasileiro
//!
//! Implementação RWKV com suporte a múltiplos backends (CPU/GPU/CUDA)

pub mod data;
pub mod model;
pub mod tokenizer;

pub use data::{DataLoader, MmapDataset, TokenizedDatasetWriter, WikiCleaner, WikiStreamParser};
pub use model::{RWKVConfig, Trainer, TrainingConfig, RWKV};
pub use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};
