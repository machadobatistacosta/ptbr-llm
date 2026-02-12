//! PTBR-LLM: Language Model para Português Brasileiro

pub mod data;
pub mod error;
pub use error::{PtbrError, Result};
pub mod model;
pub mod tokenizer;
pub mod helpers;
mod logger;
pub mod utils;

// Re-exports principais
pub use data::{DataLoader, MmapDataset, TokenizedDatasetWriter, WikiCleaner, DirtySample, WikiStreamParser};
pub use model::{RWKVConfig, Trainer, TrainingConfig, RWKV, RWKVState, TrainStats};
pub use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer, BPEVocab};

/// Retorna nome do backend ativo
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    { "CUDA" }

    #[cfg(all(feature = "gpu", not(feature = "cuda")))]
    { "WGPU" }

    #[cfg(not(any(feature = "cuda", feature = "gpu")))]
    { "CPU (NdArray)" }
}

#[cfg(all(feature = "cuda", feature = "cpu"))]
compile_error!("Features 'cuda' and 'cpu' are mutually exclusive. Pick one.");

#[cfg(all(feature = "cuda", feature = "gpu"))]
compile_error!("Features 'cuda' and 'gpu' are mutually exclusive. Pick one.");

#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("Features 'cpu' and 'gpu' are mutually exclusive. Pick one.");