mod config;
mod rwkv;
mod trainer;
mod adapters;

pub use config::{RWKVConfig, TrainingConfig};
pub use rwkv::RWKV;
pub use trainer::Trainer;

// Exports opcionais - só quando usados
#[allow(unused_imports)]
pub use rwkv::{RWKVBlock, TimeMixing, ChannelMixing, RWKVState};
#[allow(unused_imports)]
pub use adapters::{
    LoRAAdapter, 
    DomainAdapterBank, 
    Domain, 
    DomainFineTuneConfig,
    DomainRegistry,
    LoRABuilder,
};