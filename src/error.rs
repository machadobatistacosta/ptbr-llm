use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PtbrError {
    // --- I/O ---
    #[error("Failed to read file {path}: {source}")]
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Failed to write file {path}: {source}")]
    FileWrite {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    // --- Data ---
    #[error("Dataset corrupt at byte offset {offset}: {reason}")]
    DataCorrupt { offset: usize, reason: String },

    #[error("Mmap failed for {path}: {source}")]
    MmapError {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Dataset empty: {path}")]
    DatasetEmpty { path: PathBuf },

    // --- Model ---
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Checkpoint load failed: {0}")]
    CheckpointLoad(String),

    // --- CUDA ---
    #[error("CUDA kernel `{kernel}` failed: {reason}")]
    CudaError { kernel: String, reason: String },

    #[error("CUDA unavailable — compile with --features cuda")]
    CudaUnavailable,

    // --- Tokenizer ---
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Tokenizer load failed: {0}")]
    TokenizerLoad(String),

    #[error("Unknown token ID: {0}")]
    UnknownToken(u32),

    // --- Config ---
    #[error("Invalid config: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, PtbrError>;
