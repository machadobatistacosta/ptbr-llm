// src/model/mod.rs

pub mod config;
pub mod rwkv;
pub mod trainer;
pub mod adapters;

pub use config::{RWKVConfig, TrainingConfig};
pub use rwkv::RWKV;
pub use trainer::Trainer;