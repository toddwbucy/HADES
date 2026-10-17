//! Mechanical graphable-document mapping. Source assertions stay source text.
//! No model calls, inferred predicates, or cross-document equivalence decisions.
mod mapping;
mod source;
mod store;

use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub use store::import;

pub const MAPPING_VERSION: &str = "graphable-mechanical-v1";
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid graphable source: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Database(#[from] crate::db::ArangoError),
    #[error(transparent)]
    Handler(#[from] crate::dispatch::HandlerError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Immutable, validated plan. Callers cannot inject arbitrary rows into import.
#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    graph: String,
    doc_id: String,
    version: String,
    source_sha256: String,
    nodes: Vec<Value>,
    edges: Vec<Value>,
    warnings: Vec<String>,
    #[serde(skip_serializing)]
    manifest: Value,
}

impl Plan {
    /// Read a bounded source without following any links declared by its content.
    pub fn from_path(path: &Path, graph: &str) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(source::MAX_SOURCE as u64 + 1)
            .read_to_end(&mut bytes)?;
        let format = path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        Self::from_bytes(&bytes, &format, graph)
    }

    pub fn from_bytes(bytes: &[u8], format: &str, graph: &str) -> Result<Self> {
        crate::dispatch::kg_names(graph)?;
        let source = source::decode(bytes, format)?;
        mapping::map(bytes, source, graph)
    }

    /// Attach an operator-supplied original to a reviewed Markdown extraction.
    /// This records provenance, not a claim that HADES verified the extraction.
    pub fn with_original_source(mut self, path: &Path, extraction_note: &str) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(source::MAX_SOURCE as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > source::MAX_SOURCE || extraction_note.trim().is_empty() {
            return Err(Error::Invalid(
                "original source must fit 8 MiB and have an extraction note".into(),
            ));
        }
        let provenance = json!({"original_source_sha256":hash(&bytes),
            "format":path.extension().and_then(|s|s.to_str()).unwrap_or("binary"),
            "extraction_note":extraction_note,"association":"operator_supplied",
            "source_text_role":"derived_graphable_markdown"});
        self.nodes[0]["original_source"] = provenance.clone();
        self.manifest["original_source"] = provenance;
        self.manifest["original_source_base64"] = json!(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &bytes
        ));
        self.manifest["payload_sha256"] = json!(payload_hash(&self.nodes, &self.edges));
        if serde_json::to_vec(&(&self.nodes, &self.edges, &self.manifest))?.len() > 12 * 1024 * 1024
        {
            return Err(Error::Invalid(
                "mapped payload and archive exceed 12 MiB".into(),
            ));
        }
        Ok(self)
    }

    pub fn nodes(&self) -> &[Value] {
        &self.nodes
    }
    pub fn edges(&self) -> &[Value] {
        &self.edges
    }
    pub fn summary(&self) -> Value {
        json!({"graph":self.graph,"doc_id":self.doc_id,"source_version":self.version,
            "source_sha256":self.source_sha256,"mapping_version":MAPPING_VERSION,
            "nodes":self.nodes.len(),"edges":self.edges.len(),"warnings":self.warnings,
            "original_source":self.nodes[0]["original_source"],
            "root_id":format!("{}_nodes/{}",self.graph,self.nodes[0]["_key"].as_str().unwrap()),
            "embedding":"not_requested","semantic_search":"not_verified"})
    }
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn payload_hash(nodes: &[Value], edges: &[Value]) -> String {
    hash(&crate::canonical_json::to_vec(&json!([nodes, edges])))
}

fn key(kind: &str, parts: &[&str]) -> String {
    // JSON tuple encoding avoids delimiter collisions in document-controlled IDs.
    format!(
        "{kind}_{}",
        &hash(&serde_json::to_vec(parts).unwrap())[..32]
    )
}

#[cfg(test)]
mod tests;
