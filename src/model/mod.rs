mod adapters;
mod config;
mod evaluator;
mod rwkv;
mod trainer;

pub use config::{RWKVConfig, TrainingConfig};
pub use evaluator::{Evaluator, EvalMetrics};
pub use rwkv::RWKV;
pub use trainer::{Trainer, TrainStats};

// Exports opcionais
#[allow(unused_imports)]
pub use adapters::{Domain, DomainAdapterBank, DomainFineTuneConfig, LoRAAdapter};
#[allow(unused_imports)]
pub use rwkv::{ChannelMixing, RWKVBlock, RWKVState, TimeMixing};