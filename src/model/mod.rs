//! Módulo do modelo RWKV (v4 + v7)

mod config;
mod evaluator;
mod lr_finder;
mod rwkv;
mod rwkv_v7;
mod traits;
mod trainer;
mod validation;
mod precision;
mod wkv_optimized;
mod wkv_v7;
pub mod wkv_cuda_ffi;

// Shared trait
pub use traits::RWKVModel;

// RWKV-4 exports
pub use config::{RWKVConfig, TrainingConfig};
pub use evaluator::{Evaluator, EvalMetrics};
pub use lr_finder::{find_lr, LRFinderResult};
pub use rwkv::{RWKV, RWKVState, RWKVBlock, TimeMixing, ChannelMixing};
pub use trainer::{Trainer, TrainStats};

// RWKV-7 exports
pub use rwkv_v7::RWKV_V7;

// Exports opcionais
pub use validation::{Validator, ValidationMetrics, EarlyStopping};
pub use precision::{Precision, GradScaler};
pub use wkv_optimized::{WKVConfig, wkv_linear, wkv_parallel_scan};