mod adapters;
mod config;
mod rwkv;
mod trainer;

pub use config::{RWKVConfig, TrainingConfig};
pub use rwkv::RWKV;
pub use trainer::Trainer;

// Exports opcionais - só quando usados
#[allow(unused_imports)]
pub use adapters::{
    Domain, DomainAdapterBank, DomainFineTuneConfig, DomainRegistry, LoRAAdapter, LoRABuilder,
};
#[allow(unused_imports)]
pub use rwkv::{ChannelMixing, RWKVBlock, RWKVState, TimeMixing};
