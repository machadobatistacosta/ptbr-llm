//! Centralized Error Types
//!
//! Provides a unified error system for the ptbr-llm application.

use std::fmt;
use std::path::PathBuf;

/// Unified error type for the ptbr-llm application
#[derive(Debug)]
pub enum PtbrLlmError {
    // I/O Errors
    /// Generic I/O error
    Io(std::io::Error),
    /// File not found
    FileNotFound(PathBuf),

    // Dataset Errors
    /// Dataset is empty (no valid sequences)
    DatasetEmpty { path: PathBuf, seq_len: usize },
    /// Dataset file is corrupted or invalid
    DatasetCorrupted(String),

    // Tokenizer Errors
    /// Failed to load tokenizer
    TokenizerLoad(String),
    /// Failed to save tokenizer
    TokenizerSave(String),

    // Model Errors
    /// Failed to load checkpoint
    CheckpointLoad(String),
    /// Failed to save checkpoint
    CheckpointSave(String),
    /// Invalid model configuration
    ModelConfigInvalid(String),

    // Training Errors
    /// Training produced NaN loss
    TrainingNaN { step: usize },

    // General Errors
    /// Other error with message
    Other(String),
}

impl fmt::Display for PtbrLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // I/O
            PtbrLlmError::Io(e) => write!(f, "Erro de I/O: {}", e),
            PtbrLlmError::FileNotFound(path) => {
                write!(f, "Arquivo não encontrado: {:?}", path)
            }

            // Dataset
            PtbrLlmError::DatasetEmpty { path, seq_len } => {
                write!(
                    f,
                    "Dataset vazio em {:?}. Verifique se seq_len={} é compatível com o arquivo",
                    path, seq_len
                )
            }
            PtbrLlmError::DatasetCorrupted(msg) => {
                write!(f, "Dataset corrompido: {}", msg)
            }

            // Tokenizer
            PtbrLlmError::TokenizerLoad(msg) => {
                write!(f, "Erro carregando tokenizer: {}", msg)
            }
            PtbrLlmError::TokenizerSave(msg) => {
                write!(f, "Erro salvando tokenizer: {}", msg)
            }

            // Model
            PtbrLlmError::CheckpointLoad(msg) => {
                write!(f, "Erro carregando checkpoint: {}", msg)
            }
            PtbrLlmError::CheckpointSave(msg) => {
                write!(f, "Erro salvando checkpoint: {}", msg)
            }
            PtbrLlmError::ModelConfigInvalid(msg) => {
                write!(f, "Configuração de modelo inválida: {}", msg)
            }

            // Training
            PtbrLlmError::TrainingNaN { step } => {
                write!(f, "Loss NaN detectada no step {}. Treinamento interrompido", step)
            }

            // General
            PtbrLlmError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for PtbrLlmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PtbrLlmError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ============ From implementations ============

impl From<std::io::Error> for PtbrLlmError {
    fn from(err: std::io::Error) -> Self {
        PtbrLlmError::Io(err)
    }
}

impl From<String> for PtbrLlmError {
    fn from(msg: String) -> Self {
        PtbrLlmError::Other(msg)
    }
}

impl From<&str> for PtbrLlmError {
    fn from(msg: &str) -> Self {
        PtbrLlmError::Other(msg.to_string())
    }
}

/// Result type alias using PtbrLlmError
pub type Result<T> = std::result::Result<T, PtbrLlmError>;
