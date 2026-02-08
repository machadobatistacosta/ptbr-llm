//! PTBR-LLM: Language Model para Português Brasileiro

pub mod data;
pub mod error;
pub mod model;
pub mod tokenizer;
pub mod logger;
pub mod utils;

// Re-exports principais
pub use data::{DataLoader, MmapDataset, TokenizedDatasetWriter, WikiCleaner, DirtySample, WikiStreamParser};
pub use model::{RWKVConfig, Trainer, TrainingConfig, RWKV, RWKVState, TrainStats};
pub use tokenizer::{BPETokenizer, BPETrainer, PTBRNormalizer, BPEVocab};

/// Retorna nome do backend ativo
pub fn backend_name() -> &'static str {
    #[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu")))]
    {
        return "CUDA";
    }
    
    #[cfg(all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu")))]
    {
        return "WGPU";
    }
    
    #[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu")))]
    {
        return "CPU (NdArray)";
    }
    
    // Fallback
    #[cfg(not(any(
        all(feature = "cuda", not(feature = "cpu"), not(feature = "gpu")),
        all(feature = "gpu", not(feature = "cuda"), not(feature = "cpu")),
        all(feature = "cpu", not(feature = "cuda"), not(feature = "gpu"))
    )))]
    {
        return "CPU (NdArray) [fallback]";
    }
}