//! PTBR-SLM: Small Language Model para Português Brasileiro
//! 
//! Implementação RWKV com suporte a múltiplos backends (CPU/GPU/CUDA)

pub mod model;
pub mod data;
pub mod tokenizer;

pub use model::{RWKV, RWKVConfig, TrainingConfig, Trainer};
pub use data::{MmapDataset, DataLoader, TokenizedDatasetWriter, WikiStreamParser, WikiCleaner};
pub use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer};