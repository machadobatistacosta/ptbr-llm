// src/data/mod.rs

mod wiki_parser;
mod cleaner;
mod dataset;

pub use wiki_parser::{WikiStreamParser, WikiParserConfig, WikiArticle};
pub use cleaner::WikiCleaner;
pub use dataset::{MmapDataset, DataLoader, TokenizedDatasetWriter};