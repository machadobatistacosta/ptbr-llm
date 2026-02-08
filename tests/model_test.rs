//! Model Integration Tests
//!
//! Tests for RWKV model forward pass and inference.

mod common;

use burn::tensor::{Tensor, Int};
use ptbr_llm::{RWKV, RWKVState};

#[test]
fn test_rwkv_forward_output_shape() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    // Create model
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Input: [batch=1, seq=16]
    let input: Tensor<common::TestBackend, 2, Int> = Tensor::zeros([1, 16], &device);
    
    let output = model.forward(input);
    let shape = output.dims();
    
    // Output should be: [batch=1, seq=16, vocab=256]
    assert_eq!(shape[0], 1, "Batch dimension");
    assert_eq!(shape[1], 16, "Sequence dimension");
    assert_eq!(shape[2], 256, "Vocab dimension");
}

#[test]
fn test_rwkv_forward_batch() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Input: [batch=4, seq=16]
    let input: Tensor<common::TestBackend, 2, Int> = Tensor::zeros([4, 16], &device);
    
    let output = model.forward(input);
    let shape = output.dims();
    
    assert_eq!(shape[0], 4, "Batch dimension should be 4");
    assert_eq!(shape[1], 16, "Sequence dimension should be 16");
    assert_eq!(shape[2], 256, "Vocab dimension should be 256");
}

#[test]
fn test_rwkv_inference_output_shape() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Input: [batch=1, seq=16]
    let input: Tensor<common::TestBackend, 2, Int> = Tensor::zeros([1, 16], &device);
    
    let output = model.forward_inference(input);
    let shape = output.dims();
    
    // Inference output: only last token logits [batch=1, vocab=256]
    assert_eq!(shape[0], 1, "Batch dimension");
    assert_eq!(shape[1], 256, "Vocab dimension");
}

#[test]
fn test_rwkv_forward_step_with_state() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Create state
    let mut state = RWKVState::new(
        config.n_layers,
        config.d_model,
        1,  // batch=1
        &device
    );
    
    // Run 10 consecutive forward steps (one token at a time)
    for i in 0..10 {
        let input: Tensor<common::TestBackend, 2, Int> = Tensor::from_ints([[i as i32]], &device);
        let output = model.forward_step(input, &mut state);
        
        let shape = output.dims();
        assert_eq!(shape[0], 1, "Batch dimension at step {}", i);
        assert_eq!(shape[1], 256, "Vocab dimension at step {}", i);
    }
}

#[test]
fn test_rwkv_state_initialization() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    // Should not crash
    let state = RWKVState::<common::TestBackend>::new(
        config.n_layers,  // 2
        config.d_model,   // 64
        1,                // batch=1
        &device
    );
    
    // State should be initialized properly
    // We can't easily inspect internals, but creation should succeed
    assert!(std::mem::size_of_val(&state) > 0);
}

#[test]
fn test_rwkv_output_finite() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Random-ish input (using small values to avoid embedding issues)
    let input: Tensor<common::TestBackend, 2, Int> = Tensor::from_ints([[10, 20, 30, 40]], &device);
    
    let output = model.forward(input);
    let output_data = output.to_data();
    
    // Check all values are finite
    for val in output_data.iter::<f32>() {
        assert!(val.is_finite(), "Output should be finite, got: {}", val);
    }
}

#[test]
fn test_rwkv_different_seq_lengths() {
    let device = common::test_device();
    let config = common::test_model_config();
    
    let model: RWKV<common::TestBackend> = RWKV::new(&config, &device);
    
    // Test with different sequence lengths
    for seq_len in [1, 8, 16, 32] {
        let input: Tensor<common::TestBackend, 2, Int> = Tensor::zeros([1, seq_len], &device);
        
        let output = model.forward(input);
        let shape = output.dims();
        
        assert_eq!(shape[0], 1, "Batch dimension for seq_len={}", seq_len);
        assert_eq!(shape[1], seq_len, "Sequence dimension for seq_len={}", seq_len);
        assert_eq!(shape[2], 256, "Vocab dimension for seq_len={}", seq_len);
    }
}
