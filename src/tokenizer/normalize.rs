use unicode_normalization::UnicodeNormalization;

/// Normalizador específico para Português Brasileiro
pub struct PTBRNormalizer {
    fix_encoding: bool,
}

impl PTBRNormalizer {
    pub fn new() -> Self {
        Self { fix_encoding: true }
    }

    /// Normaliza texto para formato consistente
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();

        // 1. Unicode NFC normalization
        result = result.nfc().collect();

        // 2. Fix mojibake (encoding problems)
        if self.fix_encoding {
            result = self.fix_mojibake(&result);
        }

        // 3. Normalize punctuation
        result = self.normalize_punctuation(&result);

        // 4. Remove control characters (except newline)
        result = result
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();

        // 5. Normalize whitespace
        result = self.normalize_whitespace(&result);

        result
    }

    /// Corrige problemas comuns de encoding (mojibake)
    fn fix_mojibake(&self, text: &str) -> String {
        // UTF-8 mal interpretado como Latin-1
        // Ã seguido de byte específico = caractere acentuado
        text
            // Minúsculas acentuadas
            .replace("\u{00C3}\u{00A1}", "\u{00E1}") // á
            .replace("\u{00C3}\u{00A9}", "\u{00E9}") // é
            .replace("\u{00C3}\u{00AD}", "\u{00ED}") // í
            .replace("\u{00C3}\u{00B3}", "\u{00F3}") // ó
            .replace("\u{00C3}\u{00BA}", "\u{00FA}") // ú
            .replace("\u{00C3}\u{00A3}", "\u{00E3}") // ã
            .replace("\u{00C3}\u{00B5}", "\u{00F5}") // õ
            .replace("\u{00C3}\u{00A7}", "\u{00E7}") // ç
            .replace("\u{00C3}\u{00A0}", "\u{00E0}") // à
            .replace("\u{00C3}\u{00A2}", "\u{00E2}") // â
            .replace("\u{00C3}\u{00AA}", "\u{00EA}") // ê
            .replace("\u{00C3}\u{00AE}", "\u{00EE}") // î
            .replace("\u{00C3}\u{00B4}", "\u{00F4}") // ô
            .replace("\u{00C3}\u{00BB}", "\u{00FB}") // û
            .replace("\u{00C3}\u{00BC}", "\u{00FC}") // ü
            .replace("\u{00C3}\u{00B1}", "\u{00F1}") // ñ
            // Maiúsculas acentuadas
            .replace("\u{00C3}\u{0081}", "\u{00C1}") // Á
            .replace("\u{00C3}\u{0089}", "\u{00C9}") // É
            .replace("\u{00C3}\u{008D}", "\u{00CD}") // Í
            .replace("\u{00C3}\u{0093}", "\u{00D3}") // Ó
            .replace("\u{00C3}\u{009A}", "\u{00DA}") // Ú
            .replace("\u{00C3}\u{0083}", "\u{00C3}") // Ã
            .replace("\u{00C3}\u{0095}", "\u{00D5}") // Õ
            .replace("\u{00C3}\u{0087}", "\u{00C7}") // Ç
    }

    /// Normaliza pontuação para formato padrão
    fn normalize_punctuation(&self, text: &str) -> String {
        text
            // Aspas curvas para retas
            .replace('\u{201C}', "\"") // "
            .replace('\u{201D}', "\"") // "
            .replace('\u{2018}', "'") // '
            .replace('\u{2019}', "'") // '
            .replace('\u{00AB}', "\"") // «
            .replace('\u{00BB}', "\"") // »
            // Travessões
            .replace('\u{2013}', "-") // –
            .replace('\u{2014}', "-") // —
            // Espaços especiais
            .replace('\u{00A0}', " ") // Non-breaking space
            .replace('\u{2002}', " ") // En space
            .replace('\u{2003}', " ") // Em space
            .replace('\u{2009}', " ") // Thin space
            .replace('\u{200B}', "") // Zero-width space
            // Reticências
            .replace('\u{2026}', "...") // …
            // Windows-1252 control chars que às vezes aparecem
            .replace('\u{0092}', "'") // Windows apostrophe
            .replace('\u{0093}', "\"") // Windows open quote
            .replace('\u{0094}', "\"") // Windows close quote
            .replace('\u{0096}', "-") // Windows dash
            .replace('\u{0097}', "-") // Windows em dash
    }

    /// Normaliza espaços em branco
    fn normalize_whitespace(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_was_space = false;
        let mut prev_was_newline = false;

        for c in text.chars() {
            match c {
                ' ' | '\t' => {
                    if !prev_was_space && !prev_was_newline {
                        result.push(' ');
                        prev_was_space = true;
                    }
                }
                '\n' => {
                    if !prev_was_newline {
                        // Remove trailing space before newline
                        if prev_was_space {
                            result.pop();
                        }
                        result.push('\n');
                        prev_was_newline = true;
                        prev_was_space = false;
                    }
                }
                _ => {
                    result.push(c);
                    prev_was_space = false;
                    prev_was_newline = false;
                }
            }
        }

        result.trim().to_string()
    }
}

impl Default for PTBRNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mojibake_fix() {
        let norm = PTBRNormalizer::new();
        // Testa padrões comuns de mojibake
        let input = "Ã¡gua coraÃ§Ã£o";
        let expected = "água coração";
        assert_eq!(norm.normalize(input), expected);
    }

    #[test]
    fn test_punctuation() {
        let norm = PTBRNormalizer::new();
        // Aspas curvas
        let input = "\u{201C}test\u{201D}";
        assert_eq!(norm.normalize(input), "\"test\"");

        // Travessão
        let input2 = "a\u{2014}b";
        assert_eq!(norm.normalize(input2), "a-b");
    }

    #[test]
    fn test_whitespace() {
        let norm = PTBRNormalizer::new();
        assert_eq!(norm.normalize("a  b   c"), "a b c");
        assert_eq!(norm.normalize("a \n b"), "a\nb");
    }

    #[test]
    fn test_preserve_accents() {
        let norm = PTBRNormalizer::new();
        // Texto que já está correto deve permanecer igual
        let input = "São Paulo é uma cidade brasileira";
        assert_eq!(norm.normalize(input), input);
    }
}
