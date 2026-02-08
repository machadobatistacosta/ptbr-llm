use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

/// Regex patterns compilados uma vez
static PATTERNS: Lazy<CleanerPatterns> = Lazy::new(CleanerPatterns::new);

struct CleanerPatterns {
    // Tabelas
    table_block: Regex,
    table_row: Regex,

    // Templates (níveis de aninhamento)
    template_simple: Regex,
    template_nested: Regex,

    // HTML
    html_tag: Regex,
    html_entity: Regex,
    html_comment: Regex,
    ref_tag: Regex,

    // Links
    category_link: Regex,
    file_link: Regex,
    wiki_link_labeled: Regex,
    wiki_link_simple: Regex,
    external_link_labeled: Regex,
    external_link_simple: Regex,

    // Infobox params
    infobox_params: Regex,
    coordinates: Regex,

    // Formatação
    bold_italic: Regex,
    headers: Regex,
    list_items: Regex,

    // Anos grudados
    year_stuck: Regex,

    // Cleanup
    multi_space: Regex,
    multi_newline: Regex,
    pipe_cleanup: Regex,
    
    // V14
    ocr_hyphen: Regex,
}

impl CleanerPatterns {
    fn new() -> Self {
        Self {
            // Tabelas
            table_block: Regex::new(r"(?s)\{\|.*?\|\}").unwrap(),
            table_row: Regex::new(r"\|-[^\n]*\n?").unwrap(),
            
            // Templates (múltiplas passadas)
            template_simple: Regex::new(r"\{\{[^{}]*\}\}").unwrap(),
            template_nested: Regex::new(r"\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}").unwrap(),
            
            // HTML
            html_tag: Regex::new(r"<[^>]+>").unwrap(),
            html_entity: Regex::new(r"&[a-zA-Z]+;|&#\d+;").unwrap(),
            html_comment: Regex::new(r"<!--[\s\S]*?-->").unwrap(),
            ref_tag: Regex::new(r"(?s)<ref[^>]*>.*?</ref>|<ref[^>]*/>").unwrap(),
            
            // Links
            category_link: Regex::new(r"\[\[(?:Categoria|Category):[^\]]+\]\]").unwrap(),
            file_link: Regex::new(r"\[\[(?:Ficheiro|Imagem|File|Image):[^\]]+\]\]").unwrap(),
            wiki_link_labeled: Regex::new(r"\[\[[^\]|]+\|([^\]]+)\]\]").unwrap(),
            wiki_link_simple: Regex::new(r"\[\[([^\]]+)\]\]").unwrap(),
            external_link_labeled: Regex::new(r"\[https?://[^\s\]]+\s+([^\]]+)\]").unwrap(),
            external_link_simple: Regex::new(r"\[https?://[^\]]+\]").unwrap(),
            
            // Parâmetros de infobox
            infobox_params: Regex::new(
                r#"(?i)(align|width|height|style|class|colspan|rowspan|bgcolor|valign|border|cellpadding|cellspacing|scope|lang)\s*=\s*"?[^"|}\n]*"?"#
            ).unwrap(),
            coordinates: Regex::new(r"(?i)(lat|lon|latitude|longitude)[MDSG]?\s*=\s*[\d.\-]+").unwrap(),
            
            // Formatação
            bold_italic: Regex::new(r"'{2,5}").unwrap(),
            headers: Regex::new(r"^={1,6}\s*([^=]+?)\s*={1,6}\s*$").unwrap(),
            list_items: Regex::new(r"^[\*#:;]+\s*").unwrap(),
            
            // Anos grudados
            year_stuck: Regex::new(r"([a-zA-ZÀ-ÿ])(\d{4})([^\d]|$)").unwrap(),
            
            // Cleanup final
            multi_space: Regex::new(r"[ \t]+").unwrap(),
            multi_newline: Regex::new(r"\n{3,}").unwrap(),
            pipe_cleanup: Regex::new(r"\|+").unwrap(),
            
            // V14: OCR Hyphen Fix (lookaround substitute)
            // Regex crate doesn't support lookaround, so we use capture groups:
            // ([a-zà-ú])-\s+([a-zà-ú]) -> $1$2
            ocr_hyphen: Regex::new(r"([a-zà-ú])-\s+([a-zà-ú])").unwrap(),
        }
    }
}

/// Cleaner principal para texto Wikipedia
pub struct WikiCleaner {
    garbage_markers: HashSet<&'static str>,
}

#[derive(Debug, Clone)]
pub struct DirtySample {
    pub line: String,
    pub score: f32,
    pub reason: String,
}

impl WikiCleaner {
    /// Calcula pontuação de "sujeira" (0.0 = limpo, 1.0 = lixo total)
    pub fn audit_line(line: &str) -> Option<DirtySample> {
        let mut score = 0.0;
        let mut reasons = Vec::new();

        let len = line.len();
        if len == 0 { return None; }

        let chars: Vec<char> = line.chars().collect();
        let total = chars.len() as f32;

        // 1. DENSIDADE ASCII vs UNICODE
        // Em PT-BR, a maioria é ASCII (a-z, A-Z, 0-9, pontuação).
        // Unicode esperado: á, é, í, ó, ú, â, ê, ô, ã, õ, ç, À, etc.
        // Símbolos matemáticos, emojis, ou outros scripts são suspeitos.
        let mut _ascii_count = 0;
        let mut _valid_pt_unicode = 0;
        let mut symbol_count = 0;
        let mut uppercase_count = 0;

        for c in &chars {
            if c.is_ascii() {
                _ascii_count += 1;
                if c.is_ascii_uppercase() {
                    uppercase_count += 1;
                }
                if !c.is_alphanumeric() && !c.is_whitespace() {
                     // Pontuação ASCII comum é OK, mas se for excessiva...
                     if !".,;?!-()\"' ".contains(*c) {
                         symbol_count += 1;
                     }
                }
            } else {
                // Lista branca de Unicode PT-BR
                if "áéíóúÁÉÍÓÚâêîôûÂÊÎÔÛãõÃÕçÇàÀüÜ".contains(*c) {
                    _valid_pt_unicode += 1;
                } else if "–—…°ªº§".contains(*c) {
                    // Pontuação estendida aceitável
                } else {
                    symbol_count += 3; // Penalidade maior para unicode estranho
                }
            }
        }

        // HEURÍSTICA 1: Excesso de Símbolos
        let symbol_ratio = symbol_count as f32 / total;
        if symbol_ratio > 0.15 {
            score += symbol_ratio * 3.0; // Boost no score
            reasons.push(format!("High Symbol Density: {:.1}%", symbol_ratio * 100.0));
        }

        // HEURÍSTICA 2: Uppercase Shout (Caps lock)
        let upper_ratio = uppercase_count as f32 / total;
        if upper_ratio > 0.6 && len > 50 {
             score += 0.5;
             reasons.push(format!("Caps Lock Abuse: {:.1}%", upper_ratio * 100.0));
        }

        // HEURÍSTICA 3: Encoding Fantasma (BOM ou escapes)
        if line.contains('\u{FEFF}') || line.contains('\u{FFFD}') {
            score += 1.0;
            reasons.push("Ghost BOM / Replacement Char Detected".to_string());
        }
        
        // Malformed Escapes detecção (ex: \x00, \u0000 literais)
        if line.contains("\\x") || line.contains("\\u") {
             // Pode ser código fonte escapado
             score += 0.4;
             reasons.push("Potential Malformed Escape Sequence".to_string());
        }

        // HEURÍSTICA 4: Palavras Impossíveis (4 consoantes raras seguidas)
        // Consoantes: bcdfghjklmnpqrstvwxyz (removemos vogais e 'ç')
        // Vamos ser estritos: Sequências que não ocorrem em PT-BR.
        // Tentei ser 'estatístico', mas checker simples de cluster funciona bem.
        // Clusters comuns: trans, const, nstr, bstr (abstrato) -> max 4 consoantes.
        // Se tiver 5, é muito provável erro ou estrangeiro/sigla ruim.
        let mut cons_streak = 0;
        let mut max_streak = 0;
        for c in &chars {
            if "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ".contains(*c) {
                cons_streak += 1;
            } else {
                if cons_streak > max_streak { max_streak = cons_streak; }
                cons_streak = 0;
            }
        }
        if cons_streak > max_streak { max_streak = cons_streak; }

        if max_streak >= 5 {
             score += 0.8;
             reasons.push(format!("Impossible Consonant Cluster (len={})", max_streak));
        }

        if score > 0.5 {
            Some(DirtySample {
                line: line.to_string(),
                score,
                reason: reasons.join(" | "),
            })
        } else {
            None
        }
    }
}

impl WikiCleaner {
    pub fn new() -> Self {
        let garbage_markers: HashSet<&'static str> = [
            "align=",
            "width=",
            "style=",
            "colspan=",
            "rowspan=",
            "latM=",
            "lonM=",
            "latS=",
            "lonS=",
            "latG=",
            "lonG=",
            "class=",
            "bgcolor=",
            "valign=",
            "border=",
            "{{Infobox",
            "{{Caixa",
            "{{Info/",
            "{{Taxobox",
            "{{Se-",
            "{{Ref",
            "{{Ver ",
            "{{Main",
            "!--",
            "-->",
            "{|",
            "|-",
            "|}",
            "||",
        ]
        .iter()
        .cloned()
        .collect();

        Self { garbage_markers }
    }

    pub fn fix_planalto(&self, text: &str) -> String {
        // Mapa de correção de caracteres corrompidos do Planalto
        // 'ę': 'ê', 'ş': 'º', 'ŕ': 'à', 'ă': 'ã', 'č': 'è', 'ľ': 'ç', 'î': 'í', 'đ': 'ð'
        text.replace('ę', "ê")
            .replace('ş', "º")
            .replace('ŕ', "à")
            .replace('ă', "ã")
            .replace('č', "è")
            .replace('ľ', "ç")
            .replace('î', "í")
            .replace('đ', "ð")
    }

    /// Limpa texto de marcação Wikipedia
    pub fn clean(&self, text: &str) -> String {
        // 0. Planalto Fix (Correção de Mojibake específico)
        let mut result = self.fix_planalto(text);

        // 0.5 HTML Unescape (usando quick_xml ou match básico se falhar)
        if let Ok(unescaped) = quick_xml::escape::unescape(&result) {
            result = unescaped.to_string();
        }

        // 1. Remove comentários HTML
        result = PATTERNS.html_comment.replace_all(&result, "").to_string();

        // 2. Remove refs
        result = PATTERNS.ref_tag.replace_all(&result, "").to_string();

        // 3. Remove tabelas
        result = PATTERNS.table_block.replace_all(&result, " ").to_string();
        result = PATTERNS.table_row.replace_all(&result, " ").to_string();

        // 4. Remove templates (múltiplas passadas para aninhados)
        for _ in 0..5 {
            let prev = result.clone();
            result = PATTERNS
                .template_simple
                .replace_all(&result, "")
                .to_string();
            result = PATTERNS
                .template_nested
                .replace_all(&result, "")
                .to_string();
            if result == prev {
                break;
            }
        }

        // 5. Remove links especiais
        result = PATTERNS.category_link.replace_all(&result, "").to_string();
        result = PATTERNS.file_link.replace_all(&result, "").to_string();

        // 6. Converte links para texto
        result = PATTERNS
            .wiki_link_labeled
            .replace_all(&result, "$1")
            .to_string();
        result = PATTERNS
            .wiki_link_simple
            .replace_all(&result, "$1")
            .to_string();
        result = PATTERNS
            .external_link_labeled
            .replace_all(&result, "$1")
            .to_string();
        result = PATTERNS
            .external_link_simple
            .replace_all(&result, "")
            .to_string();

        // 7. Remove HTML
        result = PATTERNS.html_tag.replace_all(&result, "").to_string();
        result = PATTERNS.html_entity.replace_all(&result, " ").to_string();

        // 8. Remove parâmetros de infobox
        result = PATTERNS.infobox_params.replace_all(&result, "").to_string();
        result = PATTERNS.coordinates.replace_all(&result, "").to_string();

        // 9. Remove formatação wiki
        result = PATTERNS.bold_italic.replace_all(&result, "").to_string();

        // 10. Converte headers para texto
        result = PATTERNS.headers.replace_all(&result, "$1\n").to_string();

        // 11. Remove marcadores de lista
        let result = result
            .lines()
            .map(|line| PATTERNS.list_items.replace(line, "").to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // 12. Limpa pipes residuais
        let mut result = PATTERNS.pipe_cleanup.replace_all(&result, " ").to_string();

        // 13. Corrige anos grudados
        result = PATTERNS
            .year_stuck
            .replace_all(&result, "$1 $2$3")
            .to_string();

        // 13.5 OCR Hyphen Fix ("cons- tituição" -> "constituição")
        result = PATTERNS
            .ocr_hyphen
            .replace_all(&result, "$1$2")
            .to_string();

        // 14. Normaliza espaços
        result = PATTERNS.multi_space.replace_all(&result, " ").to_string();
        result = PATTERNS
            .multi_newline
            .replace_all(&result, "\n\n")
            .to_string();

        // 15. Filtra linhas
        result = self.filter_lines(&result);

        result.trim().to_string()
    }

    fn filter_lines(&self, text: &str) -> String {
        text.lines()
            .filter(|line| self.is_valid_line(line))
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn is_valid_line(&self, line: &str) -> bool {
        let line = line.trim();

        // Muito curta
        if line.len() < 30 {
            return false;
        }

        // Contém marcadores de lixo
        for marker in &self.garbage_markers {
            if line.contains(marker) {
                return false;
            }
        }

        // Muitos caracteres especiais
        let special_count = line.chars().filter(|c| "{}[]|<>=".contains(*c)).count();
        if special_count > 3 {
            return false;
        }

        // Proporção de letras muito baixa
        let alpha = line.chars().filter(|c| c.is_alphabetic()).count();
        let total = line.chars().filter(|c| !c.is_whitespace()).count();
        if total > 0 && (alpha as f64 / total as f64) < 0.7 {
            return false;
        }

        // Poucas palavras
        let words = line.split_whitespace().count();
        if words < 5 {
            return false;
        }

        // Começa com caractere problemático
        if let Some(first) = line.chars().next() {
            if "{}[]|<>=*#!:;".contains(first) {
                return false;
            }
        }

        true
    }
}

impl Default for WikiCleaner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_template() {
        let cleaner = WikiCleaner::new();
        let input = "Texto {{template|param=value}} mais texto";
        let output = cleaner.clean(input);
        assert!(!output.contains("{{"));
        assert!(!output.contains("}}"));
    }

    #[test]
    fn test_clean_link() {
        let cleaner = WikiCleaner::new();
        // Longer input to pass filter_lines (min 30 chars, 5 words)
        let input = "O [[Brasil|pa\u{00ED}s]] \u{00E9} grande e muito bonito com suas praias e montanhas";
        let output = cleaner.clean(input);
        assert!(output.contains("pa\u{00ED}s"), "Output was: {}", output);
        assert!(!output.contains("[["));
    }

    #[test]
    fn test_clean_table() {
        let cleaner = WikiCleaner::new();
        let input = "Antes {| class=\"wikitable\"\n|-\n| cel1 || cel2\n|} depois";
        let output = cleaner.clean(input);
        assert!(!output.contains("{|"));
        assert!(!output.contains("|}"));
    }
}
