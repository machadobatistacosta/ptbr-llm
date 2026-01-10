// src/utils/format.rs
//! Funções de formatação para exibição

/// Formata número de parâmetros (85M, 1.5B, etc.)
pub fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

/// Formata número genérico com sufixo K/M
pub fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

/// Formata bytes em KB/MB/GB
pub fn format_bytes(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} GB", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1} KB", n as f64 / 1e3)
    } else {
        format!("{} B", n)
    }
}

/// Formata duração em segundos para formato legível
pub fn format_duration(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;

    if h > 0 {
        format!("{}h{}m{}s", h, m, s)
    } else if m > 0 {
        format!("{}m{}s", m, s)
    } else {
        format!("{}s", s)
    }
}

/// Formata tokens por segundo
pub fn format_throughput(tokens_per_sec: f64) -> String {
    if tokens_per_sec >= 1_000_000.0 {
        format!("{:.2}M tok/s", tokens_per_sec / 1e6)
    } else if tokens_per_sec >= 1_000.0 {
        format!("{:.1}K tok/s", tokens_per_sec / 1e3)
    } else {
        format!("{:.1} tok/s", tokens_per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(85_000_000), "85.0M");
        assert_eq!(format_params(1_500_000_000), "1.50B");
        assert_eq!(format_params(768), "768");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_500_000), "1.5 MB");
        assert_eq!(format_bytes(16_000_000_000), "16.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(125), "2m5s");
        assert_eq!(format_duration(3725), "1h2m5s");
    }
}