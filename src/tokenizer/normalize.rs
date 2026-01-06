// src/tokenizer/normalize.rs

use unicode_normalization::UnicodeNormalization;

/// Normalizador específico para Português do Brasil
pub struct PTBRNormalizer {
    lowercase: bool,
    fix_encoding: bool,
}

impl PTBRNormalizer {
    pub fn new() -> Self {
        Self {
            lowercase: false,
            fix_encoding: true,
        }
    }

    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // 1. Normalização Unicode (NFC)
        result = result.nfc().collect();

        // 2. Corrige encoding comum (mojibake)
        if self.fix_encoding {
            result = self.fix_mojibake(&result);
        }

        // 3. Normaliza aspas e hífen
        result = self.normalize_punctuation(&result);

        // 4. Remove caracteres de controle (exceto newline)
        result = result
            .chars()
            .filter(|c| !c.is_control() || *c == '\n')
            .collect();

        // 5. Lowercase opcional
        if self.lowercase {
            result = result.to_lowercase();
        }

        result
    }

    fn fix_mojibake(&self, text: &str) -> String {
        text.replace("\u{00C3}\u{00A1}", "a")  // á
            .replace("\u{00C3}\u{00A9}", "e")  // é
            .replace("\u{00C3}\u{00AD}", "i")  // í
            .replace("\u{00C3}\u{00B3}", "o")  // ó
            .replace("\u{00C3}\u{00BA}", "u")  // ú
            .replace("\u{00C3}\u{00A3}", "a")  // ã
            .replace("\u{00C3}\u{00B5}", "o")  // õ
            .replace("\u{00C3}\u{00A7}", "c")  // ç
    }

    fn normalize_punctuation(&self, text: &str) -> String {
        text
            // Aspas curvas para retas
            .replace('\u{201C}', "\"")  // "
            .replace('\u{201D}', "\"")  // "
            .replace('\u{2018}', "'")   // '
            .replace('\u{2019}', "'")   // '
            .replace('\u{00AB}', "\"")  // «
            .replace('\u{00BB}', "\"")  // »
            // Travessões para hífen
            .replace('\u{2013}', "-")   // –
            .replace('\u{2014}', "-")   // —
            // Espaços especiais
            .replace('\u{00A0}', " ")   // Non-breaking space
    }
}

impl Default for PTBRNormalizer {
    fn default() -> Self {
        Self::new()
    }
}