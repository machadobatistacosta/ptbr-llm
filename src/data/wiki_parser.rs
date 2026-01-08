// src/data/wiki_parser.rs

use std::fs::File;
use std::io::BufReader;
use bzip2::read::BzDecoder;
use quick_xml::events::Event;
use quick_xml::Reader;

/// Configuração do parser
#[derive(Clone)]
pub struct WikiParserConfig {
    pub chunk_size: usize,
    pub max_article_len: usize,
    pub min_article_len: usize,
}

impl Default for WikiParserConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            max_article_len: 100_000,
            min_article_len: 200,
        }
    }
}

/// Artigo parseado da Wikipedia
#[derive(Debug, Clone)]
pub struct WikiArticle {
    #[allow(dead_code)]
    pub title: String,
    pub text: String,
}

/// Parser streaming
pub struct WikiStreamParser {
    config: WikiParserConfig,
}

impl WikiStreamParser {
    pub fn new(config: WikiParserConfig) -> Self {
        Self { config }
    }

    pub fn parse_streaming(&self, path: &str) -> WikiArticleIterator {
        let file = File::open(path).expect("Erro ao abrir arquivo");
        let decompressor = BzDecoder::new(BufReader::with_capacity(self.config.chunk_size, file));
        let reader = BufReader::with_capacity(self.config.chunk_size, decompressor);
        WikiArticleIterator::new(reader, self.config.clone())
    }
}

/// Iterator que parseia artigos sob demanda
pub struct WikiArticleIterator {
    reader: Reader<BufReader<BzDecoder<BufReader<File>>>>,
    config: WikiParserConfig,
    buf: Vec<u8>,
    current_title: Option<String>,
    in_text: bool,
    in_title: bool,
}

impl WikiArticleIterator {
    fn new(reader: BufReader<BzDecoder<BufReader<File>>>, config: WikiParserConfig) -> Self {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.trim_text(true);
        
        Self {
            reader: xml_reader,
            config,
            buf: Vec::with_capacity(8192),
            current_title: None,
            in_text: false,
            in_title: false,
        }
    }
}

impl Iterator for WikiArticleIterator {
    type Item = WikiArticle;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            
            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"title" => self.in_title = true,
                        b"text" => self.in_text = true,
                        _ => {}
                    }
                }
                Ok(Event::Text(ref e)) => {
                    if self.in_title {
                        self.current_title = e.unescape().ok().map(|s| s.to_string());
                    } else if self.in_text {
                        if let Some(title) = self.current_title.take() {
                            if let Ok(text) = e.unescape() {
                                let text = text.to_string();
                                let len = text.len();
                                
                                if len >= self.config.min_article_len 
                                    && len <= self.config.max_article_len 
                                {
                                    self.in_text = false;
                                    return Some(WikiArticle { title, text });
                                }
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    match e.name().as_ref() {
                        b"title" => self.in_title = false,
                        b"text" => self.in_text = false,
                        _ => {}
                    }
                }
                Ok(Event::Eof) => return None,
                Err(_) => continue,
                _ => {}
            }
        }
    }
}