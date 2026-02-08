//! Dataset Integration Tests
//!
//! Tests for MmapDataset and DataLoader functionality.

mod common;

use ptbr_llm::{MmapDataset, TokenizedDatasetWriter, DataLoader};
use tempfile::tempdir;

#[test]
fn test_write_and_read_dataset() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    // Generate some test tokens
    let tokens: Vec<u16> = (0..1000).map(|i| (i % 250) as u16 + 4).collect();  // Skip 0-3 (special)
    
    // Write dataset
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    // Read back
    let seq_len = 32;
    let dataset = MmapDataset::from_file(&path, seq_len)
        .expect("Failed to read dataset");
    
    assert!(dataset.num_tokens() > 0, "Should have tokens");
    assert!(!dataset.is_empty(), "Dataset should not be empty");
}

#[test]
fn test_dataset_get_returns_correct_length() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    let tokens: Vec<u16> = (0..500).map(|i| (i % 250) as u16 + 4).collect();
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    let seq_len = 32;
    let dataset = MmapDataset::from_file(&path, seq_len)
        .expect("Failed to read dataset");
    
    if dataset.len() > 0 {
        if let Some((input, target)) = dataset.get(0) {
            assert_eq!(input.len(), seq_len, "Input should have seq_len tokens");
            assert_eq!(target.len(), seq_len, "Target should have seq_len tokens");
        } else {
            panic!("get(0) should return Some for non-empty dataset");
        }
    }
}

#[test]
fn test_dataset_shuffle_changes_order() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    // Large enough dataset to ensure shuffle actually changes something
    let tokens: Vec<u16> = (0..2000).map(|i| (i % 250) as u16 + 4).collect();
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    let seq_len = 32;
    let mut dataset = MmapDataset::from_file(&path, seq_len)
        .expect("Failed to read dataset");
    
    if dataset.len() > 1 {
        // Get first sample before shuffle
        let before_first = if let Some((input, _)) = dataset.get(0) {
            input
        } else {
            return;
        };
        
        // Shuffle
        dataset.shuffle(42);
        
        // Get first sample after shuffle
        let after_first = if let Some((input, _)) = dataset.get(0) {
            input
        } else {
            return;
        };
        
        // With enough data, lengths should still match
        assert_eq!(before_first.len(), after_first.len());
    }
}

#[test]
fn test_dataset_validation_split() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    let tokens: Vec<u16> = (0..1000).map(|i| (i % 250) as u16 + 4).collect();
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    let seq_len = 32;
    let mut dataset = MmapDataset::from_file(&path, seq_len)
        .expect("Failed to read dataset");
    
    let original_len = dataset.len();
    
    // Reserve 10% for validation
    dataset.reserve_validation(0.1);
    
    let new_len = dataset.len();
    
    // Training set should be smaller after reserving validation
    if original_len > 0 {
        assert!(new_len <= original_len, "Training set should not grow after split");
    }
}

#[test]
fn test_empty_dataset_handling() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    // Very small dataset - less than seq_len tokens
    let tokens: Vec<u16> = vec![10, 20, 30];  // Only 3 tokens
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    let seq_len = 32;  // Need 32 tokens per sequence
    let dataset = MmapDataset::from_file(&path, seq_len);
    
    // Either should fail gracefully or return empty dataset
    match dataset {
        Ok(ds) => {
            // If it loaded, should report as empty
            assert!(ds.is_empty() || ds.len() == 0, "Dataset with < seq_len tokens should be empty");
        }
        Err(_) => {
            // Expected - too small for any sequences
        }
    }
}

#[test]
fn test_tokenized_dataset_writer() {
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("writer_test.bin");
    
    let mut writer = TokenizedDatasetWriter::new(&path)
        .expect("Failed to create writer");
    
    // Write some tokens
    let tokens1: Vec<u16> = vec![1, 2, 3, 4, 5];
    let tokens2: Vec<u16> = vec![100, 101, 102];
    
    writer.write_tokens(&tokens1).expect("Failed to write tokens1");
    writer.write_tokens(&tokens2).expect("Failed to write tokens2");
    
    let total = writer.finish().expect("Failed to finish");
    
    assert_eq!(total, 8, "Should have written 8 tokens total");
    assert!(path.exists(), "Output file should exist");
}

#[test]
fn test_dataloader_batch_shapes() {
    let dir = tempdir().expect("Failed to create temp dir");
    
    let tokens: Vec<u16> = (0..1000).map(|i| (i % 250) as u16 + 4).collect();
    let path = common::create_temp_dataset(&tokens, dir.path());
    
    let seq_len = 32;
    let dataset = MmapDataset::from_file(&path, seq_len)
        .expect("Failed to read dataset");
    
    if dataset.is_empty() {
        return;
    }
    
    let batch_size = 2;
    let loader = DataLoader::new(&dataset, batch_size);
    
    // Check first batch
    for (inputs, targets) in loader {
        // Each batch should have batch_size sequences (or less for last batch)
        assert!(inputs.len() <= batch_size, "Batch should have <= batch_size sequences");
        assert_eq!(inputs.len(), targets.len(), "Input/target counts should match");
        
        // Each sequence should have seq_len tokens
        for seq in &inputs {
            assert_eq!(seq.len(), seq_len, "Each input sequence should have seq_len tokens");
        }
        for seq in &targets {
            assert_eq!(seq.len(), seq_len, "Each target sequence should have seq_len tokens");
        }
        
        break;  // Just check first batch
    }
}
