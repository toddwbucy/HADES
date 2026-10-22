//! Internal conventional code preparation and atomic publication of a single file graph.
//! Analyzers, filesystem admission, cross-file enrichment and public routing are separate owners.
mod graph;
mod storage;

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    code::{FileAnalysis, compute_content_hash},
    db::{ArangoError, ArangoPool, collections::CODEBASE, keys},
    embedding_profile::{
        BoundEmbeddingRequest, ChunkPolicy, EmbeddingRole, ProfileState, WorkloadKind,
        store::{ProfileSnapshot, ProfileStore},
    },
    persephone::{
        bound_embedding::BoundEmbeddingClient,
        passage_plan::{PassagePlan, validate_source},
        provider_registry::ProviderCatalog,
    },
    pipeline::{PipelineError, profiled_vectors::validate_vector},
};

pub use graph::{LocalEndpoint, LocalRelationship, RelationshipKind};

const FORMAT: &str = "hades-bound-code-v1";
const MAX_PREPARED_BYTES: usize = 16 * 1024 * 1024;
const MAX_SYMBOLS: usize = 1024;
const MAX_RELATIONSHIPS: usize = 4096;
const MAX_ROWS: usize = MAX_SYMBOLS + MAX_RELATIONSHIPS;
type Rows = BTreeMap<String, Vec<Value>>;

/// Already analyzed source bytes. Callers cannot provide vectors, token counts or database keys.
pub struct CodeInput {
    pub namespace: String,
    pub path: String,
    pub text: String,
    pub analysis: FileAnalysis,
    pub relationships: Vec<LocalRelationship>,
}

/// Opaque, complete graph preparation bound to one database/profile/provider observation.
pub struct PreparedCodeFile {
    snapshot: ProfileSnapshot,
    binding: BoundEmbeddingRequest,
    previous_revision: Option<String>,
    file: Value,
    rows: Rows,
}

impl PreparedCodeFile {
    pub fn file_key(&self) -> &str {
        self.file["_key"].as_str().expect("constructed file key")
    }
}

#[derive(Debug, Serialize)]
pub struct CodeReceipt {
    pub file_key: String,
    pub revision: String,
    pub chunks: usize,
    pub symbols: usize,
    pub relationships: usize,
    pub binding: BoundEmbeddingRequest,
}

/// Internal storage admission; this does not enable CLI, job or MCP code ingestion.
pub struct ProfiledCode {
    pool: ArangoPool,
    store: ProfileStore,
    snapshot: ProfileSnapshot,
    passage: BoundEmbeddingClient,
}

impl ProfiledCode {
    pub async fn connect(
        pool: ArangoPool,
        catalog: &ProviderCatalog,
    ) -> Result<Self, PipelineError> {
        let store = ProfileStore::new(pool.clone())?;
        let snapshot = store.load().await?;
        let profile = snapshot.profile();
        ensure(
            profile.state == ProfileState::Ready,
            "workspace profile is not ready",
        )?;
        let workload = profile
            .workloads
            .code
            .as_ref()
            .ok_or_else(|| invalid("code is disabled_by_profile"))?;
        ensure(
            matches!(workload.chunking, ChunkPolicy::Conventional { .. }),
            "this code path requires conventional chunking; contextual code is unavailable",
        )?;
        let client = pool.writer().clone().with_response_limit(64 * 1024)?;
        for (name, kind) in CODEBASE.all_collections() {
            let properties = client.get(&format!("collection/{name}/properties")).await?;
            ensure(
                properties["name"] == name && properties["type"] == kind,
                "code storage requires its complete document/edge schema",
            )?;
        }
        let observed = catalog.verify_profile(profile, snapshot.manifest()).await?;
        let passage = observed
            .client(
                profile,
                snapshot.manifest(),
                WorkloadKind::Code,
                EmbeddingRole::Passage,
            )
            .await?;
        ensure(
            store.load().await? == snapshot,
            "workspace profile changed during provider verification",
        )?;
        Ok(Self {
            pool,
            store,
            snapshot,
            passage,
        })
    }

    /// Validate the graph before external work, then plan/embed the exact original source.
    /// Replacement must be explicit; no legacy adoption or lower-tier graph replacement is allowed.
    pub async fn prepare(
        &self,
        input: CodeInput,
        replace: bool,
    ) -> Result<PreparedCodeFile, PipelineError> {
        graph::validate_input(&input)?;
        self.current().await?;
        let key = keys::scoped_file_key(&input.namespace, &input.path);
        let (mut file, mut rows) = graph::records(&input, &key, self.passage.binding())?;
        let previous_revision = storage::admit(
            self.pool.writer(),
            &key,
            replace,
            &self.snapshot,
            input.analysis.analysis_tier,
            &rows[CODEBASE.symbols],
        )
        .await?;
        let planned = self.passage.plan_passages(&input.text).await?;
        ensure(
            planned.binding == *self.passage.binding(),
            "planned code binding changed",
        )?;
        self.current().await?;
        let texts: Vec<String> = planned
            .output
            .spans
            .iter()
            .map(|span| input.text[span.start_byte..span.end_byte].to_owned())
            .collect();
        let embedded = self.passage.embed(&texts, Some(16)).await?;
        ensure(
            embedded.binding == planned.binding && embedded.output.embeddings.len() == texts.len(),
            "provider returned an incomplete or differently bound code file",
        )?;
        let binding = embedded.binding;
        let normalization = self
            .snapshot
            .profile()
            .workloads
            .code
            .as_ref()
            .unwrap()
            .space
            .representation
            .normalize;
        for vector in &embedded.output.embeddings {
            validate_vector(vector, binding.dimension, normalization)?;
        }
        file["passage_plan"] = json!(planned.output);
        file["chunk_count"] = json!(texts.len());
        for (index, ((span, text), vector)) in planned
            .output
            .spans
            .iter()
            .zip(texts)
            .zip(embedded.output.embeddings)
            .enumerate()
        {
            let chunk = format!("bcode_{}", digest(&json!([binding.generation, key, index])));
            let common = json!({"_key":chunk,"format":FORMAT,"file_key":key,
                "chunk_key":chunk,"chunk_index":index,"content_hash":file["content_hash"],
                "embedding_binding":binding});
            let mut passage = common.clone();
            passage["text"] = json!(text);
            passage["span"] = json!(span);
            rows.get_mut(CODEBASE.chunks).unwrap().push(passage);
            let mut embedding = common;
            embedding["model"] = json!(binding.model);
            embedding["dimension"] = json!(binding.dimension);
            embedding["embedding"] = json!(vector);
            rows.get_mut(CODEBASE.embeddings).unwrap().push(embedding);
        }
        // The inventory detects orphan rows, edits and unknown authored fields before replacement.
        file["record_manifest"] = json!(graph::manifest(&rows)?);
        ensure(
            serde_json::to_vec(&json!([file, rows]))
                .map_err(ArangoError::from)?
                .len()
                <= MAX_PREPARED_BYTES,
            "prepared code graph exceeds 16 MiB; split the source explicitly",
        )?;
        Ok(PreparedCodeFile {
            snapshot: self.snapshot.clone(),
            binding,
            previous_revision,
            file,
            rows,
        })
    }

    pub async fn commit(&self, prepared: PreparedCodeFile) -> Result<CodeReceipt, PipelineError> {
        ensure(
            prepared.snapshot == self.snapshot && prepared.binding == *self.passage.binding(),
            "prepared code belongs to a different database or provider binding",
        )?;
        let mut receipt = CodeReceipt {
            file_key: prepared.file_key().into(),
            revision: String::new(),
            chunks: prepared.rows[CODEBASE.chunks].len(),
            symbols: prepared.rows[CODEBASE.symbols].len(),
            relationships: CODEBASE
                .edge_collections()
                .iter()
                .map(|name| prepared.rows[*name].len())
                .sum(),
            binding: prepared.binding.clone(),
        };
        receipt.revision = self
            .store
            .guarded_write(
                &self.snapshot,
                CODEBASE
                    .all_collections()
                    .iter()
                    .map(|(name, _)| (*name).into())
                    .collect(),
                move |client| storage::write(client, prepared),
            )
            .await?;
        Ok(receipt)
    }

    async fn current(&self) -> Result<(), PipelineError> {
        ensure(
            self.store.load().await? == self.snapshot,
            "stale database profile; resolve a fresh code binding",
        )?;
        Ok(())
    }
}

fn digest(value: &Value) -> String {
    Sha256::digest(crate::canonical_json::to_vec(value))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn invalid(message: &str) -> ArangoError {
    ArangoError::Request(format!("profile-bound code: {message}"))
}

fn ensure(condition: bool, message: &str) -> Result<(), ArangoError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
