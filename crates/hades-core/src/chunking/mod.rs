//! Document chunking engine.
//!
//! Pure Rust text chunking with multiple strategies.  Each strategy
//! implements the [`ChunkingStrategy`] trait, producing [`TextChunk`]
//! values with character offsets for source attribution.
//!
//! Late chunking does NOT live here. It is pooled server-side, next to the
//! model, because token-level embeddings are `seq_len x 2048` and a long
//! document runs to roughly 120 MB in float32 before pooling. See
//! `persephone::embedding::embed_late_chunked` and
//! `services/embedding/jina_v4.py`.
//!
//! A client-side `late` submodule used to sit here, unreferenced. It required
//! exactly the token-level tensors that cannot cross the wire, so it could
//! never have been called, and it was removed rather than left for the next
//! reader to find and reimplement against.

mod strategies;

pub use strategies::{SentenceChunking, SlidingWindowChunking, TokenChunking};

/// A chunk of text with positional metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// The chunk text content.
    pub text: String,
    /// Start byte offset in the original document (inclusive).
    pub start_char: usize,
    /// End byte offset in the original document (exclusive).
    pub end_char: usize,
    /// Zero-based chunk index within the document.
    pub chunk_index: usize,
    /// Total number of chunks produced from the document.
    ///
    /// Populated by the chunking strategy before returning.
    pub total_chunks: usize,
}

/// Strategy for splitting text into chunks.
pub trait ChunkingStrategy {
    /// Split `text` into a vector of [`TextChunk`]s.
    ///
    /// The returned chunks have `total_chunks` already filled in.
    fn chunk(&self, text: &str) -> Vec<TextChunk>;
}
