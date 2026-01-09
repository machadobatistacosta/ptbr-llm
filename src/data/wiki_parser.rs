#![allow(dead_code)]
use bzip2::read::BzDecoder;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Artigo parseado da Wikipedia
#[derive(Debug, Clone)]
pub struct WikiArticle {
    pub title: String,
    pub text: String,
}

/// Parser streaming para Wikipedia dumps
pub struct WikiStreamParser {
    min_chars: usize,
}

impl WikiStreamParser {
    pub fn new(min_chars: usize) -> Self {
        Self { min_chars }
    }

    /// Retorna iterator boxed para evitar generics complexos
    pub fn parse_streaming(&self, path: &str) -> Box<dyn Iterator<Item = WikiArticle>> {
        let file = File::open(path).expect("Erro abrindo arquivo");
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let decompressor = BzDecoder::new(reader);
        let buf_reader = BufReader::with_capacity(4 * 1024 * 1024, decompressor);

        Box::new(WikiArticleIterator::new(buf_reader, self.min_chars))
    }
}

struct WikiArticleIterator<R: BufRead> {
    reader: Reader<R>,
    min_chars: usize,
    buf: Vec<u8>,
    current_title: String,
    current_text: String,
    current_ns: i32,
    in_title: bool,
    in_text: bool,
    in_ns: bool,
}

impl<R: BufRead> WikiArticleIterator<R> {
    fn new(reader: R, min_chars: usize) -> Self {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.trim_text(false);

        Self {
            reader: xml_reader,
            min_chars,
            buf: Vec::with_capacity(64 * 1024),
            current_title: String::new(),
            current_text: String::new(),
            current_ns: -1,
            in_title: false,
            in_text: false,
            in_ns: false,
        }
    }

    fn is_redirect(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        lower.starts_with("#redirect") || lower.starts_with("#redirecionamento")
    }
}

impl<R: BufRead> Iterator for WikiArticleIterator<R> {
    type Item = WikiArticle;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();

            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(ref e)) => match e.name().as_ref() {
                    b"title" => {
                        self.in_title = true;
                        self.current_title.clear();
                    }
                    b"text" => {
                        self.in_text = true;
                        self.current_text.clear();
                    }
                    b"ns" => {
                        self.in_ns = true;
                    }
                    b"page" => {
                        self.current_title.clear();
                        self.current_text.clear();
                        self.current_ns = -1;
                    }
                    _ => {}
                },

                Ok(Event::Text(ref e)) => {
                    if self.in_title {
                        if let Ok(text) = e.unescape() {
                            self.current_title.push_str(&text);
                        }
                    } else if self.in_text {
                        if let Ok(text) = e.unescape() {
                            self.current_text.push_str(&text);
                        }
                    } else if self.in_ns {
                        if let Ok(text) = e.unescape() {
                            self.current_ns = text.trim().parse().unwrap_or(-1);
                        }
                    }
                }

                Ok(Event::End(ref e)) => match e.name().as_ref() {
                    b"title" => self.in_title = false,
                    b"text" => self.in_text = false,
                    b"ns" => self.in_ns = false,
                    b"page" => {
                        if self.current_ns == 0
                            && self.current_text.len() >= self.min_chars
                            && !self.current_title.is_empty()
                            && !self.is_redirect(&self.current_text)
                        {
                            return Some(WikiArticle {
                                title: std::mem::take(&mut self.current_title),
                                text: std::mem::take(&mut self.current_text),
                            });
                        }
                    }
                    _ => {}
                },

                Ok(Event::CData(ref e)) => {
                    if self.in_text {
                        if let Ok(text) = std::str::from_utf8(e.as_ref()) {
                            self.current_text.push_str(text);
                        }
                    }
                }

                Ok(Event::Eof) => return None,
                Err(_) => continue,
                _ => {}
            }
        }
    }
}
