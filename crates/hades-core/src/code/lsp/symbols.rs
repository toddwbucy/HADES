//! Language-neutral semantic extraction records consumed by graph resolution.

use serde::{Deserialize, Serialize};

pub fn symbol_kind_name(kind: u64) -> &'static str {
    match kind {
        1 => "file",
        2 => "module",
        3 => "namespace",
        4 => "package",
        5 => "class",
        6 => "method",
        7 => "property",
        8 => "field",
        9 => "constructor",
        10 => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        15 => "string",
        16 => "number",
        17 => "boolean",
        18 => "array",
        19 => "object",
        20 => "key",
        21 => "null",
        22 => "enum_member",
        23 => "struct",
        24 => "event",
        25 => "operator",
        26 => "type_parameter",
        _ => "unknown",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedSymbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub visibility: String,
    pub signature: String,
    /// Zero-based LSP line.
    pub start_line: u32,
    /// Zero-based LSP line.
    pub end_line: u32,
    pub parent_symbol: Option<String>,
    /// Rust trait or Go interface implemented by this symbol, when directly known.
    pub impl_trait: Option<String>,
    pub is_pyo3: bool,
    pub is_ffi: bool,
    pub is_unsafe: bool,
    pub derives: Vec<String>,
    pub python_name: Option<String>,
    pub calls: Vec<CallTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallTarget {
    pub qualified_name: String,
    pub name: String,
    pub file: String,
    /// Zero-based LSP line.
    pub line: u32,
}

/// Go's implicit interface satisfaction discovered by gopls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationTarget {
    /// Index of the interface declaration in this file extraction.
    pub interface_symbol: usize,
    pub interface_name: String,
    pub interface_qualified_name: String,
    pub implementor_file: String,
    /// Zero-based LSP line of the implementing method.
    pub implementor_line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileExtraction {
    /// Uncapped symbol indices; the store maps these to scoped graph keys, never names.
    #[serde(default)]
    pub failed_edge_symbols: std::collections::HashSet<usize>,
    /// Interface symbol indices with an incomplete implementation query.
    #[serde(default)]
    pub failed_implementation_interfaces: std::collections::HashSet<usize>,
    #[serde(default)]
    pub no_calls: Vec<FailedRequest>,
    #[serde(default)]
    pub no_call_count: usize,
    #[serde(default)]
    pub failed_requests: Vec<FailedRequest>,
    #[serde(default)]
    pub failed_request_count: usize,
    pub symbols: Vec<ExtractedSymbol>,
    pub impl_blocks: Vec<ImplBlock>,
    pub implementations: Vec<ImplementationTarget>,
    pub pyo3_exports: Vec<String>,
    pub ffi_boundaries: Vec<String>,
    pub analyzed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplBlock {
    pub self_type: String,
    pub trait_name: Option<String>,
    pub methods: Vec<String>,
}

impl FileExtraction {
    pub fn empty() -> Self {
        Self {
            failed_edge_symbols: Default::default(),
            failed_implementation_interfaces: Default::default(),
            no_calls: Vec::new(),
            no_call_count: 0,
            failed_requests: Vec::new(),
            failed_request_count: 0,
            symbols: Vec::new(),
            impl_blocks: Vec::new(),
            implementations: Vec::new(),
            pyo3_exports: Vec::new(),
            ffi_boundaries: Vec::new(),
            analyzed_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Details are capped independently from the total, so large files cannot
/// inflate the ingest envelope without bound (#179).
pub const FAILED_REQUEST_LIMIT: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedRequest {
    pub file: String,
    pub symbol: String,
    pub request: String,
    pub reason: String,
}
