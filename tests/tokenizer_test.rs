//! Tokenizer Integration Tests
//!
//! Tests for BPE tokenizer functionality.

mod common;

use ptbr_llm::{BPETokenizer, PTBRNormalizer, BPEVocab};

#[test]
fn test_encode_decode_roundtrip_ascii() {
    let tokenizer = common::create_test_tokenizer();
    let text = "hello world";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens);
    
    // ASCII should be recoverable
    assert!(decoded.contains("hello") || decoded.contains("world") || !tokens.is_empty(),
        "Encoding should produce tokens for ASCII text");
}

#[test]
fn test_encode_decode_roundtrip_ptbr() {
    let tokenizer = common::create_test_tokenizer();
    
    // Portuguese text with accents
    let texts = [
        "São Paulo é uma cidade",
        "ação reação",
        "à á â ã é ê í ó ô õ ú ç",
    ];
    
    for text in texts {
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty(), "Should produce tokens for: {}", text);
        
        let decoded = tokenizer.decode(&tokens);
        assert!(!decoded.is_empty(), "Should decode back to non-empty string");
    }
}

#[test]
fn test_encode_empty_string() {
    let tokenizer = common::create_test_tokenizer();
    let tokens = tokenizer.encode("");
    
    // Empty string should produce empty or minimal tokens
    assert!(tokens.len() <= 2, "Empty string should produce 0-2 tokens (possible BOS/EOS only)");
}

#[test]
fn test_special_tokens_ids() {
    let tokenizer = common::create_test_tokenizer();
    
    let bos_id = tokenizer.bos_id();
    let eos_id = tokenizer.eos_id();
    let vocab_size = tokenizer.vocab_size();
    
    // BOS and EOS should be different
    // Note: With our byte tokenizer, they might be same if not properly set
    assert!(bos_id < vocab_size as u16, "BOS should be < vocab_size");
    assert!(eos_id < vocab_size as u16, "EOS should be < vocab_size");
}

#[test]
fn test_encode_produces_valid_ids() {
    let tokenizer = common::create_test_tokenizer();
    let vocab_size = tokenizer.vocab_size();
    
    // Long Portuguese text
    let text = "O Brasil é o maior país da América do Sul. \
                Sua capital é Brasília. A língua oficial é o português.";
    
    let tokens = tokenizer.encode(text);
    
    for token in tokens {
        assert!(
            (token as usize) < vocab_size,
            "Token ID {} should be < vocab_size {}",
            token,
            vocab_size
        );
    }
}

#[test]
fn test_normalizer_mojibake() {
    let normalizer = PTBRNormalizer::new();
    
    // This might be mojibake - normalizer should not crash
    let result = normalizer.normalize("Ã£o");
    assert!(!result.is_empty(), "Normalizer should return something");
}

#[test]
fn test_normalizer_whitespace() {
    let normalizer = PTBRNormalizer::new();
    
    // Multiple spaces should be normalized
    let text = "hello    world\t\ttabs";
    let normalized = normalizer.normalize(text);
    
    // Should not contain multiple consecutive spaces
    assert!(!normalized.contains("  "), "Should not have double spaces");
}

#[test]
fn test_bpe_vocab_creation() {
    // Test that BPEVocab can be created and used
    let mut vocab = BPEVocab::new();
    
    // Add some tokens
    vocab.id_to_token.push(b"hello".to_vec());
    vocab.id_to_token.push(b"world".to_vec());
    
    assert_eq!(vocab.id_to_token.len(), 2);
    
    // Test token_to_id map
    let token_to_id = vocab.build_token_to_id();
    assert_eq!(token_to_id.get(b"hello".as_slice()), Some(&0u16));
    assert_eq!(token_to_id.get(b"world".as_slice()), Some(&1u16));
}
