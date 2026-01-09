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
        }
    }
}

/// Cleaner principal para texto Wikipedia
pub struct WikiCleaner {
    garbage_markers: HashSet<&'static str>,
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

    /// Limpa texto de marcação Wikipedia
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();

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
        let input = "O [[Brasil|país]] é grande";
        let output = cleaner.clean(input);
        assert!(output.contains("país"));
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
