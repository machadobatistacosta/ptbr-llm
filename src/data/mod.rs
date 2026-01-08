mod wiki_parser;
mod cleaner;
mod dataset;

pub use wiki_parser::WikiStreamParser;
pub use cleaner::WikiCleaner;
pub use dataset::{MmapDataset, DataLoader, TokenizedDatasetWriter};

// Export opcional
#[allow(unused_imports)]
pub use wiki_parser::WikiArticle;