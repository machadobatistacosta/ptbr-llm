// src/model/precision.rs
//! Configuração de precisão numérica

#[derive(Debug, Clone, Copy)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    Mixed, // FP16 forward, FP32 backward
}

impl Precision {
    pub fn recommended_for_gpu() -> Self {
        // BF16 é melhor para estabilidade em treino
        Precision::BF16
    }

    pub fn recommended_for_cpu() -> Self {
        Precision::FP32
    }
}

/// Loss scaling para mixed precision
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_growth: usize,
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_growth: 0,
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn update(&mut self, had_inf: bool) {
        if had_inf {
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}