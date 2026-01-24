mod config;
mod evaluator;
mod lr_finder;
mod rwkv;
mod trainer;
pub mod wkv_optimized;
pub mod precision;
pub mod validation;

pub use config::{RWKVConfig, TrainingConfig};
pub use evaluator::Evaluator;
pub use lr_finder::{LRFinderResult, find_lr};
pub use rwkv::RWKV;
pub use trainer::Trainer;

// Exports opcionais
#[allow(unused_imports)]
pub use rwkv::{ChannelMixing, RWKVBlock, RWKVState, TimeMixing};