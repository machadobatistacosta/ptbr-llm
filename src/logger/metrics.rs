// src/logger/metrics.rs
//! Logging de treino para arquivo e métricas CSV

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Logger de treino para arquivo de texto
#[allow(dead_code)]
pub struct TrainLogger {
    file: File,
    start_time: Instant,
    log_every: usize,
}

#[allow(dead_code)]
impl TrainLogger {
    /// Cria novo logger
    pub fn new(output_dir: &Path, log_every: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(output_dir)?;
        let log_path = output_dir.join("training.log");

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        Ok(Self {
            file,
            start_time: Instant::now(),
            log_every,
        })
    }

    /// Loga um step de treino
    pub fn log_step(&mut self, step: usize, loss: f32, lr: f64, grad_norm: f32) {
        if step % self.log_every != 0 {
            return;
        }

        let elapsed = self.start_time.elapsed().as_secs();
        let ppl = (loss as f64).exp();

        let line = format!(
            "[{:>6}s] Step {:>6} | Loss: {:.4} | PPL: {:>8.2} | LR: {:.2e} | GradNorm: {:.4}\n",
            elapsed, step, loss, ppl, lr, grad_norm
        );

        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Loga resultado de avaliação
    pub fn log_eval(&mut self, step: usize, eval_loss: f32, eval_ppl: f32) {
        let elapsed = self.start_time.elapsed().as_secs();
        let line = format!(
            "[{:>6}s] EVAL Step {:>6} | Loss: {:.4} | PPL: {:.2}\n",
            elapsed, step, eval_loss, eval_ppl
        );
        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Loga checkpoint salvo
    pub fn log_checkpoint(&mut self, step: usize, path: &str) {
        let elapsed = self.start_time.elapsed().as_secs();
        let line = format!("[{:>6}s] CHECKPOINT Step {} -> {}\n", elapsed, step, path);
        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Loga mensagem genérica
    pub fn log_message(&mut self, msg: &str) {
        let elapsed = self.start_time.elapsed().as_secs();
        let line = format!("[{:>6}s] {}\n", elapsed, msg);
        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Loga início de epoch
    pub fn log_epoch(&mut self, epoch: usize) {
        let elapsed = self.start_time.elapsed().as_secs();
        let line = format!("[{:>6}s] === EPOCH {} ===\n", elapsed, epoch);
        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Loga sample gerado
    pub fn log_sample(&mut self, prompt: &str, generated: &str) {
        let elapsed = self.start_time.elapsed().as_secs();
        let line = format!(
            "[{:>6}s] SAMPLE: \"{}\" -> \"{}\"\n",
            elapsed,
            prompt,
            generated.chars().take(100).collect::<String>()
        );
        let _ = self.file.write_all(line.as_bytes());
        let _ = self.file.flush();
    }

    /// Retorna tempo decorrido em segundos
    pub fn elapsed_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

/// CSV de métricas para análise posterior
#[allow(dead_code)]
pub struct MetricsCSV {
    file: File,
    #[allow(dead_code)]
    has_header: bool,
}

#[allow(dead_code)]
impl MetricsCSV {
    /// Cria novo arquivo CSV de métricas
    pub fn new(output_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(output_dir)?;
        let path = output_dir.join("metrics.csv");
        let mut file = File::create(&path)?;

        // Header
        writeln!(
            file,
            "step,loss,ppl,lr,grad_norm,tokens_per_sec,eval_loss,eval_ppl,epoch"
        )?;

        Ok(Self {
            file,
            has_header: true,
        })
    }

    /// Abre CSV existente para append
    pub fn open_append(output_dir: &Path) -> std::io::Result<Self> {
        let path = output_dir.join("metrics.csv");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        Ok(Self {
            file,
            has_header: true,
        })
    }

    /// Registra métricas de treino
    pub fn record_train(
        &mut self,
        step: usize,
        loss: f32,
        lr: f64,
        grad_norm: f32,
        tokens_per_sec: f32,
    ) {
        let ppl = (loss as f64).exp();
        let _ = writeln!(
            self.file,
            "{},{:.6},{:.2},{:.2e},{:.4},{:.1},,,,",
            step, loss, ppl, lr, grad_norm, tokens_per_sec
        );
        let _ = self.file.flush();
    }

    /// Registra métricas de avaliação
    pub fn record_eval(&mut self, step: usize, eval_loss: f32, eval_ppl: f32) {
        let _ = writeln!(
            self.file,
            "{},,,,,,{:.6},{:.2},",
            step, eval_loss, eval_ppl
        );
        let _ = self.file.flush();
    }

    /// Registra métricas completas (treino + eval)
    pub fn record_full(
        &mut self,
        step: usize,
        loss: f32,
        lr: f64,
        grad_norm: f32,
        tokens_per_sec: f32,
        eval_loss: Option<f32>,
        eval_ppl: Option<f32>,
        epoch: usize,
    ) {
        let ppl = (loss as f64).exp();
        let eval_loss_str = eval_loss.map(|v| format!("{:.6}", v)).unwrap_or_default();
        let eval_ppl_str = eval_ppl.map(|v| format!("{:.2}", v)).unwrap_or_default();

        let _ = writeln!(
            self.file,
            "{},{:.6},{:.2},{:.2e},{:.4},{:.1},{},{},{}",
            step, loss, ppl, lr, grad_norm, tokens_per_sec, eval_loss_str, eval_ppl_str, epoch
        );
        let _ = self.file.flush();
    }

    /// Registra início de epoch
    pub fn record_epoch(&mut self, epoch: usize, step: usize) {
        let _ = writeln!(self.file, "# Epoch {} started at step {}", epoch, step);
        let _ = self.file.flush();
    }
}

/// Estatísticas agregadas de treino
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub total_tokens: usize,
    pub total_epochs: usize,
    pub best_loss: f32,
    pub best_eval_loss: f32,
    pub final_loss: f32,
    pub training_time_secs: u64,
}

#[allow(dead_code)]
impl TrainingStats {
    pub fn new() -> Self {
        Self {
            best_loss: f32::MAX,
            best_eval_loss: f32::MAX,
            ..Default::default()
        }
    }

    pub fn update_loss(&mut self, loss: f32) {
        self.final_loss = loss;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
    }

    pub fn update_eval_loss(&mut self, eval_loss: f32) {
        if eval_loss < self.best_eval_loss {
            self.best_eval_loss = eval_loss;
        }
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.training_time_secs > 0 {
            self.total_tokens as f64 / self.training_time_secs as f64
        } else {
            0.0
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Steps: {} | Tokens: {} | Epochs: {} | Best Loss: {:.4} | Best Eval: {:.4} | Time: {}s | {:.1}K tok/s",
            self.total_steps,
            self.total_tokens,
            self.total_epochs,
            self.best_loss,
            self.best_eval_loss,
            self.training_time_secs,
            self.tokens_per_second() / 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::tempdir;

    #[test]
    fn test_train_logger() {
        let dir = tempdir().unwrap();
        let mut logger = TrainLogger::new(dir.path(), 1).unwrap();

        logger.log_step(1, 5.5, 1e-4, 0.5);
        logger.log_eval(1, 5.0, 148.4);
        logger.log_message("Test message");

        let log_path = dir.path().join("training.log");
        let mut content = String::new();
        File::open(log_path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("Step"));
        assert!(content.contains("EVAL"));
        assert!(content.contains("Test message"));
    }

    #[test]
    fn test_metrics_csv() {
        let dir = tempdir().unwrap();
        let mut csv = MetricsCSV::new(dir.path()).unwrap();

        csv.record_train(1, 5.5, 1e-4, 0.5, 1000.0);
        csv.record_eval(1, 5.0, 148.4);

        let csv_path = dir.path().join("metrics.csv");
        let mut content = String::new();
        File::open(csv_path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("step,loss,ppl"));
        assert!(content.contains("5.5"));
    }
}