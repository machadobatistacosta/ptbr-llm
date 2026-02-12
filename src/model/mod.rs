//! Módulo do modelo RWKV

mod config;
mod evaluator;
mod lr_finder;
mod rwkv;
mod trainer;
mod validation;
mod precision;
mod wkv_optimized;
pub mod wkv_cuda_ffi;

// Exports principais
pub use config::{RWKVConfig, TrainingConfig};
pub use evaluator::{Evaluator, EvalMetrics};
pub use lr_finder::{find_lr, LRFinderResult};
pub use rwkv::{RWKV, RWKVState, RWKVBlock, TimeMixing, ChannelMixing};
pub use trainer::{Trainer, TrainStats};

// Exports opcionais
pub use validation::{Validator, ValidationMetrics, EarlyStopping};
pub use precision::{Precision, GradScaler};
pub use wkv_optimized::{WKVConfig, wkv_linear, wkv_parallel_scan};