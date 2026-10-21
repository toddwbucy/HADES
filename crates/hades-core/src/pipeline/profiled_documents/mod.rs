//! Internal document admission -> verified embedding -> atomic storage -> retrieval.
//!
//! Shared search and foreground file ingestion use this path. A trusted caller
//! may also supply extracted text while the provider plans token-checked spans.
//! Only conventional document text and cosine ranking are supported.

pub mod files;
mod storage;

use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    db::{ArangoError, ArangoPool},
    embedding_profile::{
        BoundEmbeddingRequest, ChunkPolicy, Distance, EmbeddingRole, Normalization, ProfileState,
        WorkloadKind,
        store::{ProfilePin, ProfileSnapshot, ProfileStore},
    },
    persephone::{
        bound_embedding::BoundEmbeddingClient,
        passage_plan::{MAX_PASSAGES, MAX_TEXT_BYTES, PassagePlan, TokenPassage, validate_source},
        provider_registry::ProviderCatalog,
    },
    pipeline::PipelineError,
};

const FORMAT: &str = "hades-bound-document-v2";
const MAX_PREPARED_BYTES: usize = 16 * 1024 * 1024;
const COLLECTIONS: [&str; 3] = ["documents", "chunks", "embeddings"];

/// Already extracted text. The verified provider supplies every passage boundary.
pub struct DocumentInput {
    pub source_id: String,
    pub text: String,
}

/// Private prepared rows cannot be replaced with arbitrary vectors by the caller.
pub struct PreparedDocument {
    snapshot: ProfileSnapshot,
    binding: BoundEmbeddingRequest,
    previous_revision: Option<String>,
    document: Value,
    chunks: Vec<Value>,
    embeddings: Vec<Value>,
}

impl PreparedDocument {
    /// Stable source key; passage keys additionally include the corpus generation.
    pub fn document_key(&self) -> &str {
        self.document["_key"]
            .as_str()
            .expect("constructed document key")
    }
}

/// Published document identity and the verified provider observation used to embed it.
#[derive(Debug, Serialize)]
pub struct DocumentReceipt {
    pub document_key: String,
    pub chunks: usize,
    pub binding: BoundEmbeddingRequest,
}

/// Complete exact-search results. A scan limit is an error, never a partial ranking.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub items: Vec<Value>,
    pub scanned: u64,
    pub binding: BoundEmbeddingRequest,
}

/// One fixed database/profile/provider binding, with separate passage and query roles.
pub struct ProfiledDocuments {
    pool: ArangoPool,
    store: ProfileStore,
    snapshot: ProfileSnapshot,
    passage: BoundEmbeddingClient,
    query: BoundEmbeddingClient,
    normalization: Normalization,
}

impl ProfiledDocuments {
    /// Resolve the stored profile before constructing any embedding client.
    /// Resolve the saved provider ID through protected registration and actual HTTP verification.
    pub async fn connect(
        pool: ArangoPool,
        catalog: &ProviderCatalog,
    ) -> Result<Self, PipelineError> {
        Self::connect_retaining(pool, catalog, ()).await
    }

    pub(crate) async fn connect_retaining<R: Clone + Send + 'static>(
        pool: ArangoPool,
        catalog: &ProviderCatalog,
        retention: R,
    ) -> Result<Self, PipelineError> {
        Self::connect_checked(pool, catalog, None, retention).await
    }

    /// An inherited job must retain its parent's database/profile admission.
    pub async fn connect_pinned(
        pool: ArangoPool,
        catalog: &ProviderCatalog,
        pin: &ProfilePin,
    ) -> Result<Self, PipelineError> {
        Self::connect_checked(pool, catalog, Some(pin), ()).await
    }

    /// Compact evidence for a child worker; no host paths or manifest bytes are exposed.
    pub fn job_pin(&self) -> Result<ProfilePin, PipelineError> {
        Ok(self.snapshot.job_pin()?)
    }

    async fn connect_checked<R: Clone + Send + 'static>(
        pool: ArangoPool,
        catalog: &ProviderCatalog,
        pin: Option<&ProfilePin>,
        retention: R,
    ) -> Result<Self, PipelineError> {
        let store = ProfileStore::new(pool.clone())?;
        let snapshot = store.load_retaining(retention.clone()).await?;
        if let Some(pin) = pin {
            pin.check(&snapshot)?;
        }
        let profile = snapshot.profile();
        ensure(
            profile.state == ProfileState::Ready,
            "workspace profile is not ready",
        )?;
        let workload = profile
            .workloads
            .documents
            .as_ref()
            .ok_or_else(|| invalid("documents are disabled_by_profile"))?;
        ensure(
            matches!(workload.chunking, ChunkPolicy::Conventional { .. }),
            "this document path requires conventional chunking",
        )?;
        ensure(
            workload.space.representation.distance == Distance::Cosine,
            "this document path supports cosine distance only",
        )?;
        let normalization = workload.space.representation.normalize;
        // Schema is a provisioning responsibility; never create collections as a query side effect.
        for collection in COLLECTIONS {
            let properties = pool
                .writer()
                .clone()
                .with_response_limit(64 * 1024)?
                .get(&format!("collection/{collection}/properties"))
                .await?;
            ensure(
                properties["type"].as_u64() == Some(2),
                "document storage requires three document collections",
            )?;
        }
        let observed = catalog.verify_profile(profile, snapshot.manifest()).await?;
        let passage = observed
            .client(
                profile,
                snapshot.manifest(),
                WorkloadKind::Documents,
                EmbeddingRole::Passage,
            )
            .await?;
        let query = observed
            .client(
                profile,
                snapshot.manifest(),
                WorkloadKind::Documents,
                EmbeddingRole::Query,
            )
            .await?;
        ensure(
            store.load_retaining(retention).await? == snapshot,
            "workspace profile changed during provider verification",
        )?;
        Ok(Self {
            pool,
            store,
            snapshot,
            passage,
            query,
            normalization,
        })
    }

    /// Plan and embed the complete extraction under the stored conventional policy.
    /// Replacement is explicit and pins the previous metadata revision before inference.
    /// Provider overflow is returned unchanged; no truncation or fallback is attempted.
    pub async fn prepare(
        &self,
        input: DocumentInput,
        replace: bool,
    ) -> Result<PreparedDocument, PipelineError> {
        validate_input(&input)?;
        self.current().await?;
        let key = digest_key("bd", &json!([input.source_id]));
        let admission = storage::admit_source(
            self.pool.writer(),
            &key,
            replace,
            self.passage.binding(),
            &self.snapshot,
            None,
        )
        .await?;
        self.prepare_admitted(input, admission.revision, None, ())
            .await
    }

    /// Preserve admission from before extraction through planning and inference.
    async fn prepare_admitted<R: Clone + Send + 'static>(
        &self,
        input: DocumentInput,
        previous_revision: Option<String>,
        ingestion: Option<files::FileProvenance>,
        retention: R,
    ) -> Result<PreparedDocument, PipelineError> {
        validate_input(&input)?;
        if let Some(ingestion) = &ingestion {
            ingestion.validate(&input.source_id)?;
        }
        self.current_retaining(retention.clone()).await?;
        let key = digest_key("bd", &json!([input.source_id]));
        let planned = self.passage.plan_passages(&input.text).await?;
        ensure(
            planned.binding == *self.passage.binding(),
            "planned provider binding changed",
        )?;
        // Detect policy changes during planning before spending any inference.
        self.current_retaining(retention).await?;
        let plan = planned.output;
        let texts: Vec<String> = plan
            .spans
            .iter()
            .map(|span| input.text[span.start_byte..span.end_byte].to_owned())
            .collect();
        let result = self.passage.embed(&texts, Some(16)).await?;
        ensure(
            result.binding == *self.passage.binding(),
            "prepared provider binding changed",
        )?;
        ensure(
            result.output.embeddings.len() == plan.spans.len(),
            "provider returned an incomplete document",
        )?;
        let source_hash = hash(input.text.as_bytes());
        let binding = result.binding;
        let mut document = json!({
            "_key":key, "format":FORMAT, "source_id":input.source_id,
            "source_sha256":source_hash, "full_text":input.text,
            "chunk_count":plan.spans.len(), "embedding_binding":binding,
            "passage_plan":plan
        });
        if let Some(ingestion) = ingestion {
            document["file_ingestion"] = json!(ingestion);
        }
        let mut chunks = Vec::new();
        let mut embeddings = Vec::new();
        for (index, ((span, text), vector)) in plan
            .spans
            .iter()
            .zip(texts)
            .zip(result.output.embeddings)
            .enumerate()
        {
            validate_vector(&vector, binding.dimension, self.normalization)?;
            let chunk_key = digest_key("bc", &json!([binding.generation, key, index]));
            let common = json!({
                "_key":chunk_key, "format":FORMAT, "parent_key":key, "chunk_key":chunk_key,
                "chunk_index":index, "source_sha256":source_hash, "embedding_binding":binding
            });
            let mut chunk = common.clone();
            chunk["text"] = json!(text);
            chunk["span"] = json!(span);
            chunks.push(chunk);
            let mut embedding = common;
            embedding["model"] = json!(binding.model);
            embedding["dimension"] = json!(binding.dimension);
            embedding["embedding"] = json!(vector);
            embeddings.push(embedding);
        }
        ensure(
            serde_json::to_vec(&json!([document, chunks, embeddings]))
                .map_err(ArangoError::from)?
                .len()
                <= MAX_PREPARED_BYTES,
            "prepared document exceeds 16 MiB; split the source explicitly",
        )?;
        Ok(PreparedDocument {
            snapshot: self.snapshot.clone(),
            binding,
            previous_revision,
            document,
            chunks,
            embeddings,
        })
    }

    /// Commit a complete prepared result only under its original database admission.
    /// Conflicts, profile drift, write failures, and cancellation before commit abort atomically.
    /// A lost commit acknowledgment has an uncertain outcome and requires reconciliation.
    pub async fn commit(
        &self,
        prepared: PreparedDocument,
    ) -> Result<DocumentReceipt, PipelineError> {
        self.commit_retaining(prepared, ()).await
    }

    /// Keep bounded file-ingestion admission until transaction cleanup finishes.
    async fn commit_retaining<R: Clone + Send + 'static>(
        &self,
        prepared: PreparedDocument,
        retention: R,
    ) -> Result<DocumentReceipt, PipelineError> {
        ensure(
            prepared.snapshot == self.snapshot && prepared.binding == *self.passage.binding(),
            "prepared document belongs to a different database or provider binding",
        )?;
        let receipt = DocumentReceipt {
            document_key: prepared.document_key().into(),
            chunks: prepared.chunks.len(),
            binding: prepared.binding.clone(),
        };
        self.store
            .guarded_write_retaining(
                &self.snapshot,
                COLLECTIONS.iter().map(|c| (*c).into()).collect(),
                retention,
                move |client| storage::write(client, prepared),
            )
            .await?;
        Ok(receipt)
    }

    /// Embed one query and rank all admitted document vectors under a coherent DB lock.
    /// Old serving loads remain compatible only when artifact, space, profile, and
    /// generation match. Legacy or foreign rows fail before their vectors are scored.
    pub async fn search(
        &self,
        text: &str,
        top_k: usize,
        max_rows: u64,
    ) -> Result<SearchResult, PipelineError> {
        self.search_retaining(text, top_k, max_rows, ()).await
    }

    pub(crate) async fn search_retaining<R: Clone + Send + 'static>(
        &self,
        text: &str,
        top_k: usize,
        max_rows: u64,
        retention: R,
    ) -> Result<SearchResult, PipelineError> {
        ensure(
            !text.trim().is_empty() && text.len() <= 64 * 1024,
            "query must contain 1..=65536 UTF-8 bytes",
        )?;
        ensure(
            (1..=1000).contains(&top_k) && (1..=100_000).contains(&max_rows),
            "search limits must be top_k 1..=1000 and max_rows 1..=100000",
        )?;
        ensure(
            self.store.load_retaining(retention.clone()).await? == self.snapshot,
            "stale database profile; resolve a fresh document binding",
        )?;
        let result = self.query.embed(&[text.into()], Some(1)).await?;
        let vector = result
            .output
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| invalid("query provider returned no vector"))?;
        validate_vector(&vector, result.binding.dimension, self.normalization)?;
        let passage = self.passage.binding().clone();
        let normalization = self.normalization;
        let snapshot = self.snapshot.clone();
        let (items, scanned) = self
            .store
            .guarded_write_retaining(
                &self.snapshot,
                COLLECTIONS.iter().map(|c| (*c).into()).collect(),
                retention,
                move |client| {
                    storage::search(
                        client,
                        passage,
                        vector,
                        normalization,
                        top_k,
                        max_rows,
                        snapshot,
                    )
                },
            )
            .await?;
        Ok(SearchResult {
            items,
            scanned,
            binding: result.binding,
        })
    }

    /// Detect stale policy before external work; commit/search recheck under locks.
    async fn current(&self) -> Result<(), PipelineError> {
        self.current_retaining(()).await
    }

    /// Snapshot reads retain caller admission through their own transaction cleanup.
    async fn current_retaining<R: Clone + Send + 'static>(
        &self,
        retention: R,
    ) -> Result<(), PipelineError> {
        ensure(
            self.store.load_retaining(retention).await? == self.snapshot,
            "stale database profile; resolve a fresh document binding",
        )?;
        Ok(())
    }
}

/// Reject malformed extraction metadata before contacting either external service.
fn validate_input(input: &DocumentInput) -> Result<(), PipelineError> {
    ensure(
        !input.source_id.trim().is_empty()
            && input.source_id.len() <= 4096
            && !input.source_id.chars().any(char::is_control),
        "invalid source identifier",
    )?;
    validate_source(&input.text)?;
    Ok(())
}

/// Reject zero/nonfinite vectors and check the declared normalization before storage/scoring.
fn validate_vector(
    vector: &[f32],
    dimension: u32,
    normalization: Normalization,
) -> Result<(), ArangoError> {
    ensure(
        vector.len() == dimension as usize && vector.iter().all(|v| v.is_finite()),
        "invalid vector dimension or finite values",
    )?;
    let norm: f64 = vector.iter().map(|v| f64::from(*v).powi(2)).sum();
    ensure(
        norm.is_finite() && norm > 0.0,
        "vector has zero or invalid norm",
    )?;
    ensure(
        normalization != Normalization::L2 || (norm.sqrt() - 1.0).abs() <= 0.001,
        "vector violates the profile's L2 normalization",
    )?;
    Ok(())
}

/// Encode stable keys from exact JSON tuples; arbitrary source IDs never become paths/AQL.
fn digest_key(prefix: &str, value: &Value) -> String {
    format!("{prefix}_{}", hash(&crate::canonical_json::to_vec(value)))
}

/// Digest exact source bytes without Unicode or whitespace normalization.
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Report rejected document policy without including source content.
fn invalid(message: &str) -> ArangoError {
    ArangoError::Request(format!("profile-bound documents: {message}"))
}

/// Refuse failed invariants without changing policy or falling back to a legacy writer.
fn ensure(condition: bool, message: &str) -> Result<(), ArangoError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
