// src/logger/mod.rs
//! Sistema de logging para treino

mod metrics;

// Exports opcionais - não usados no main.rs mas podem ser úteis
#[allow(unused_imports)]
pub use metrics::{MetricsCSV, TrainLogger};