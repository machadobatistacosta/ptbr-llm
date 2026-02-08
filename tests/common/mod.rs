//! Common test utilities and helpers
//! 
//! Shared helpers for integration tests.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use std::path::{Path, PathBuf};
use std::io::Write;

pub type TestBackend = NdArray;

pub fn test_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

/// Creates a minimal test model configuration for fast tests
pub fn test_model_config() -> ptbr_llm::RWKVConfig {
    ptbr_llm::RWKVConfig::new()
        .with_vocab_size(256)
        .with_d_model(64)
        .with_n_layers(2)
        .with_d_ffn(128)
        .with_max_seq_len(32)
        .with_dropout(0.0)
        .with_layer_norm_eps(1e-5)
        .with_weight_tying(true)
}

/// Creates a temporary tokenized dataset file
/// Returns the path to the created .bin file
pub fn create_temp_dataset(tokens: &[u16], dir: &Path) -> PathBuf {
    let path = dir.join("test_train.bin");
    let mut writer = ptbr_llm::TokenizedDatasetWriter::new(&path)
        .expect("Failed to create dataset writer");
    
    // Write tokens in chunks to simulate documents
    let chunk_size = 64.min(tokens.len());
    for chunk in tokens.chunks(chunk_size) {
        writer.write_tokens(chunk).expect("Failed to write tokens");
    }
    
    writer.finish().expect("Failed to finish writing");
    path
}

/// Creates a minimal test tokenizer with byte-level vocabulary (256 tokens)
pub fn create_test_tokenizer() -> ptbr_llm::BPETokenizer {
    // Create a basic BPEVocab with 256 byte tokens + special tokens
    let mut id_to_token: Vec<Vec<u8>> = Vec::with_capacity(260);
    
    // Tokens 0-255: single bytes
    for i in 0u8..=255 {
        id_to_token.push(vec![i]);
    }
    
    let mut special_tokens = std::collections::HashMap::new();
    // Use existing IDs for special tokens (we'll use common byte values)
    // PAD = 0, EOS = 1 (typically), BOS = 2, UNK = 3
    special_tokens.insert("[PAD]".to_string(), 0);
    special_tokens.insert("[EOS]".to_string(), 1);
    special_tokens.insert("[BOS]".to_string(), 2);
    special_tokens.insert("[UNK]".to_string(), 3);
    
    let vocab = ptbr_llm::BPEVocab {
        id_to_token,
        merges: Vec::new(),  // No merges for simple byte tokenizer
        special_tokens,
    };
    
    ptbr_llm::BPETokenizer::from_vocab(vocab)
}

/// Generates random token data for testing
pub fn generate_random_tokens(count: usize, max_val: u16) -> Vec<u16> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen_range(4..max_val)).collect()  // Skip 0-3 (special tokens)
}
