//! RWKVModel trait — Shared interface for RWKV v4 and v7
//!
//! Enables the Trainer and Evaluator to work with any RWKV version
//! without code duplication.

use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
};

/// Trait that all RWKV model variants must implement.
///
/// This is used by `Trainer<B, M>` and `Evaluator` to work generically
/// with both RWKV v4 and RWKV-7 (and future versions).
///
/// Bounds: `Module<B>` (for save/load, valid(), optimizer), `Clone`, `Send`, `Sync`, `Sized`.
pub trait RWKVModel<B: Backend>: Module<B> + Clone + Send + Sized + 'static {
    /// Forward pass for training.
    /// Input: `[batch_size, seq_len]` token IDs
    /// Output: `[batch_size, seq_len, vocab_size]` logits
    fn forward_train(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3>;

    /// Number of trainable parameters in the model.
    fn num_parameters(&self) -> usize;

    /// Vocabulary size (output dimension).
    fn vocab_size(&self) -> usize;

    /// Model hidden dimension.
    fn d_model(&self) -> usize;

    /// Snapshot of a representative learned parameter, for tracking update norms.
    ///
    /// Returns the raw f32 values. Different versions use different params:
    /// - v4: `blocks[0].time_mixing.time_decay`
    /// - v7: `blocks[0].time_mix.w0`
    fn representative_param_snapshot(&self) -> Vec<f32>;
}
