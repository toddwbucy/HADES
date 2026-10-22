//! Document processing pipeline — extract → chunk → embed → store.
//!
//! Orchestrates the Persephone extraction and embedding clients with the
//! chunking engine and ArangoDB storage.  Supports single-document and
//! batch processing with two-phase GPU memory optimization.

mod orchestrator;
pub mod profiled_code;
pub mod profiled_documents;
mod profiled_vectors;

pub use orchestrator::{DocumentResult, Pipeline, PipelineConfig, PipelineError, PipelineSummary};
