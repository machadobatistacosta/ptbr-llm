// src/data/cleaner.rs

use regex::Regex;
use std::collections::HashSet;

pub struct WikiCleaner {
    // Patterns compilados uma vez
    table_block: Regex,
    table_row: Regex,
    table_cell: Regex,
    template: Regex,
    nested_template: Regex,
    html_tag: Regex,
    html_entity: Regex,
    html_comment: Regex,
    
    // Links
    category_link: Regex,
    file_link: Regex,
    wiki_link_labeled: Regex,
    wiki_link_simple: Regex,
    external_link: Regex,
    
    // Coordenadas e infobox
    coordinates: Regex,
    infobox_params: Regex,
    
    // Formatação
    bold_italic: Regex,
    headers: Regex,
    #[allow(dead_code)]
    lists: Regex,
    
    // Números grudados
    year_stuck: Regex,
    
    // Limpeza final
    multi_space: Regex,
    multi_newline: Regex,
    
    // Marcadores de lixo
    garbage_markers: HashSet<&'static str>,
}

impl WikiCleaner {
    pub fn new() -> Self {
        Self {
            // Tabelas (mais agressivo)
            table_block: Regex::new(r"(?s)\{\|.*?\|\}").unwrap(),
            table_row: Regex::new(r"\|-[^\n]*").unwrap(),
            table_cell: Regex::new(r"\|\|").unwrap(),
            
            // Templates (incluindo aninhados)
            template: Regex::new(r"\{\{[^{}]*\}\}").unwrap(),
            nested_template: Regex::new(r"\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}").unwrap(),
            
            // HTML
            html_tag: Regex::new(r"<[^>]+>").unwrap(),
            html_entity: Regex::new(r"&[a-zA-Z]+;|&#\d+;").unwrap(),
            html_comment: Regex::new(r"<!--.*?-->").unwrap(),
            
            // Links especiais
            category_link: Regex::new(r"\[\[Categoria:[^\]]+\]\]").unwrap(),
            file_link: Regex::new(r"\[\[(Ficheiro|Imagem|File|Image):[^\]]+\]\]").unwrap(),
            wiki_link_labeled: Regex::new(r"\[\[[^\]|]+\|([^\]]+)\]\]").unwrap(),
            wiki_link_simple: Regex::new(r"\[\[([^\]]+)\]\]").unwrap(),
            external_link: Regex::new(r"\[https?://[^\]]+\]").unwrap(),
            
            // Coordenadas e parâmetros de infobox
            coordinates: Regex::new(r"(lat|lon)[MDSG]?\s*=\s*[\d.-]+").unwrap(),
            infobox_params: Regex::new(
                r#"(align|width|height|style|class|colspan|rowspan|bgcolor|valign|border|cellpadding|cellspacing)\s*=\s*"?[^"|}\n]*"?"#
            ).unwrap(),
            
            // Formatação wiki
            bold_italic: Regex::new(r"'{2,5}").unwrap(),
            headers: Regex::new(r"={2,6}\s*[^=]+\s*={2,6}").unwrap(),
            lists: Regex::new(r"^[\*#:;]+\s*", ).unwrap(),
            
            // Anos grudados: "de1975" -> "de 1975"
            year_stuck: Regex::new(r"([a-zA-ZÀ-ÿ])(\d{4})([^0-9]|$)").unwrap(),
            
            // Normalização de espaços
            multi_space: Regex::new(r"[ \t]+").unwrap(),
            multi_newline: Regex::new(r"\n{3,}").unwrap(),
            
            // Marcadores de conteúdo problemático
            garbage_markers: [
                "align=", "width=", "style=", "colspan=", "rowspan=",
                "latM=", "lonM=", "latS=", "lonS=", "latG=", "lonG=",
                "class=", "bgcolor=", "valign=", "border=",
                "{{Infobox", "{{Caixa", "{{Info/", "{{Taxobox",
                "{{Se-", "{{Se ", "{{Ref", "{{Ver ",
                "!--", "-->", "<ref", "</ref>",
                "|}", "{|", "|-",
            ].iter().cloned().collect(),
        }
    }
    
    /// Limpeza principal - use esta para processar artigos
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // 1. Remove blocos grandes primeiro
        result = self.html_comment.replace_all(&result, "").to_string();
        result = self.table_block.replace_all(&result, " ").to_string();
        
        // 2. Remove templates (múltiplas passadas para aninhados)
        for _ in 0..5 {
            let prev = result.clone();
            result = self.template.replace_all(&result, "").to_string();
            result = self.nested_template.replace_all(&result, "").to_string();
            if result == prev {
                break;
            }
        }
        
        // 3. Remove links especiais
        result = self.category_link.replace_all(&result, "").to_string();
        result = self.file_link.replace_all(&result, "").to_string();
        result = self.external_link.replace_all(&result, "").to_string();
        
        // 4. Converte links wiki para texto
        result = self.wiki_link_labeled.replace_all(&result, "$1").to_string();
        result = self.wiki_link_simple.replace_all(&result, "$1").to_string();
        
        // 5. Remove HTML
        result = self.html_tag.replace_all(&result, "").to_string();
        result = self.html_entity.replace_all(&result, " ").to_string();
        
        // 6. Remove resíduos de tabelas
        result = self.table_row.replace_all(&result, " ").to_string();
        result = self.table_cell.replace_all(&result, " ").to_string();
        result = result.replace("|", " ");
        
        // 7. Remove parâmetros problemáticos
        result = self.coordinates.replace_all(&result, "").to_string();
        result = self.infobox_params.replace_all(&result, "").to_string();
        
        // 8. Remove formatação wiki
        result = self.bold_italic.replace_all(&result, "").to_string();
        result = self.headers.replace_all(&result, "\n").to_string();
        
        // 9. Corrige anos grudados
        result = self.year_stuck.replace_all(&result, "$1 $2$3").to_string();
        
        // 10. Normaliza espaços
        result = self.multi_space.replace_all(&result, " ").to_string();
        result = self.multi_newline.replace_all(&result, "\n\n").to_string();
        
        // 11. Filtra linhas
        result = self.filter_lines(&result);
        
        result.trim().to_string()
    }
    
    /// Filtra linhas problemáticas
    fn filter_lines(&self, text: &str) -> String {
        text.lines()
            .filter(|line| self.is_valid_line(line))
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// Verifica se uma linha é válida (não é lixo)
    pub fn is_valid_line(&self, line: &str) -> bool {
        let line = line.trim();
        
        // Muito curta
        if line.len() < 20 {
            return false;
        }
        
        // Contém marcadores de lixo
        for marker in &self.garbage_markers {
            if line.contains(marker) {
                return false;
            }
        }
        
        // Muitos pipes (tabela)
        if line.matches('|').count() > 2 {
            return false;
        }
        
        // Muitos colchetes ou chaves
        if line.matches('[').count() > 5 || line.matches('{').count() > 3 {
            return false;
        }
        
        // Proporção de letras vs outros caracteres
        let alpha_count = line.chars().filter(|c| c.is_alphabetic()).count();
        let total_count = line.chars().filter(|c| !c.is_whitespace()).count();
        
        if total_count > 0 && (alpha_count as f64 / total_count as f64) < 0.6 {
            return false;
        }
        
        // Poucas palavras
        let word_count = line.split_whitespace().count();
        if word_count < 4 {
            return false;
        }
        
        // Começa com caractere problemático
        if let Some(first) = line.chars().next() {
            if "|{[<>=*#!:;".contains(first) {
                return false;
            }
        }
        
        true
    }
    
    /// Verifica se um artigo inteiro é válido
    #[allow(dead_code)]
    pub fn is_valid_article(&self, text: &str) -> bool {
        let clean = self.clean(text);
        
        // Mínimo de caracteres após limpeza
        if clean.len() < 200 {
            return false;
        }
        
        // Mínimo de linhas válidas
        let valid_lines = clean.lines().filter(|l| l.len() > 30).count();
        if valid_lines < 3 {
            return false;
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
    fn test_clean_table() {
        let cleaner = WikiCleaner::new();
        let input = "Texto antes {| class=\"wikitable\"\n|-\n| cel1 || cel2\n|} texto depois";
        let output = cleaner.clean(input);
        assert!(!output.contains("{|"));
        assert!(!output.contains("||"));
    }
    
    #[test]
    fn test_clean_coordinates() {
        let cleaner = WikiCleaner::new();
        let input = "A cidade latM=23 lonM=46 fica no Brasil";
        let output = cleaner.clean(input);
        assert!(!output.contains("latM"));
        assert!(!output.contains("lonM"));
    }
    
    #[test]
    fn test_fix_stuck_years() {
        let cleaner = WikiCleaner::new();
        let input = "Nasceu em1975 e morreu de2020.";
        let output = cleaner.clean(input);
        assert!(output.contains(" 1975"));
        assert!(output.contains(" 2020"));
    }
    
    #[test]
    fn test_valid_line() {
        let cleaner = WikiCleaner::new();
        
        // Linha boa
        assert!(cleaner.is_valid_line("O Brasil é um país localizado na América do Sul."));
        
        // Linha ruim
        assert!(!cleaner.is_valid_line("align=\"center\" width=\"100\""));
        assert!(!cleaner.is_valid_line("|-"));
        assert!(!cleaner.is_valid_line("latM=23|lonM=46"));
    }
}