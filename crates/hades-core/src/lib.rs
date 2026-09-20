pub mod batch;
pub(crate) mod canonical_json;
pub mod chunking;
pub mod code;
pub mod config;
pub mod daemon_client;
pub mod db;
pub mod dispatch;
pub mod graph;
pub mod ingest_routing;
pub mod persephone;
pub mod pipeline;
pub mod retrieval;
pub mod schema_apply;
pub mod service;
/// Throwaway-database harness for write tests. Feature-gated: see the module.
#[cfg(feature = "test-support")]
pub mod test_support;
pub mod training;

pub use config::HadesConfig;
