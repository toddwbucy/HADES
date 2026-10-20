//! Explicit profile-bound transport for #225, not yet used by production callers.
//!
//! A trusted provider resolver must supply the endpoint and an observed receipt
//! issued after verified loading. This client enforces that receipt over the
//! wire; it cannot establish the provider's honesty or authorize database writes.

use std::time::Duration;

use crate::embedding_profile::{
    ArtifactManifest, BoundEmbeddingRequest, ChunkPolicy, EmbeddingProfile, EmbeddingRole,
    ProviderReceipt, WorkloadKind, require,
};

use super::embedding::{
    EmbedResult, EmbeddingClient, EmbeddingClientConfig, EmbeddingEndpoint, EmbeddingError,
    LateChunkEmbedResult,
};

/// Operational connection resolved from a trusted installation registry.
/// Model, dimension, task, and expected identity come only from the profile.
/// No default provider or fallback endpoint is selected by this factory.
#[derive(Clone)]
pub struct ProviderConnection {
    pub provider_id: String,
    pub endpoint: EmbeddingEndpoint,
    pub timeout: Duration,
    pub max_response_bytes: usize,
}

/// Complete results plus the profile/provider snapshot that admitted them.
/// Persistence must still recheck database revision and generation before commit.
pub struct BoundEmbeddingResult<T> {
    pub output: T,
    pub binding: BoundEmbeddingRequest,
}

/// A fixed workload/role binding, including a single serving generation.
/// Clones retain the same pin; a reload requires a newly verified connection.
#[derive(Clone)]
pub struct BoundEmbeddingClient {
    inner: EmbeddingClient,
    binding: BoundEmbeddingRequest,
    chunking: ChunkPolicy,
    role: EmbeddingRole,
}

impl BoundEmbeddingClient {
    /// Validate and snapshot a ready profile and trusted provider observation.
    /// Transport construction is lazy; this does not probe or load the provider.
    pub async fn connect(
        profile: &EmbeddingProfile,
        manifest: &ArtifactManifest,
        workload: WorkloadKind,
        role: EmbeddingRole,
        provider: ProviderConnection,
        observed: &ProviderReceipt,
    ) -> Result<Self, EmbeddingError> {
        let binding = profile.bind(manifest, workload, role, observed)?;
        require(
            provider.provider_id == binding.provider_id,
            "resolved provider does not match the profile's provider ID",
        )?;
        require(
            !provider.timeout.is_zero(),
            "provider timeout must be positive",
        )?;
        require(
            provider.max_response_bytes > 0,
            "response limit must be positive",
        )?;
        // bind() above already rejects a disabled workload.
        let selected = match workload {
            WorkloadKind::Documents => profile.workloads.documents.as_ref(),
            WorkloadKind::Code => profile.workloads.code.as_ref(),
        }
        .expect("validated enabled workload");
        let inner = EmbeddingClient::connect(EmbeddingClientConfig {
            endpoint: provider.endpoint,
            model: binding.model.clone(),
            expected_dimension: binding.dimension,
            timeout: provider.timeout,
            connect_timeout: provider.timeout,
        })
        .await?
        .with_response_limit(provider.max_response_bytes)?
        .with_expected_receipt(binding.expected.clone());
        Ok(Self {
            inner,
            binding,
            chunking: selected.chunking.clone(),
            role,
        })
    }

    /// Inspect the immutable pin without exposing an unbound client or setter.
    pub fn binding(&self) -> &BoundEmbeddingRequest {
        &self.binding
    }

    /// Embed queries or conventional passages using only the bound task.
    /// Every batch and OOM retry carries the same expected receipt. Any failure
    /// discards the entire result, including previously completed batches.
    pub async fn embed(
        &self,
        texts: &[String],
        batch_size: Option<u32>,
    ) -> Result<BoundEmbeddingResult<EmbedResult>, EmbeddingError> {
        require(!texts.is_empty(), "bound embedding input must be nonempty")?;
        require(
            self.role == EmbeddingRole::Query
                || matches!(self.chunking, ChunkPolicy::Conventional { .. }),
            "passage workload requires late chunking; conventional fallback is disabled",
        )?;
        let output = self
            .inner
            .embed(texts, &self.binding.task, batch_size)
            .await?;
        Ok(BoundEmbeddingResult {
            output,
            binding: self.binding.clone(),
        })
    }

    /// Embed profile-enabled late passages with explicit character boundaries.
    /// Token counting and chunk-budget enforcement remain provider/orchestrator
    /// responsibilities; this client never invents a tokenizer or re-chunks text.
    pub async fn embed_late_chunked(
        &self,
        texts: &[String],
        boundaries: &[Vec<(usize, usize)>],
    ) -> Result<BoundEmbeddingResult<LateChunkEmbedResult>, EmbeddingError> {
        require(
            self.role == EmbeddingRole::Passage
                && matches!(self.chunking, ChunkPolicy::Late { .. }),
            "late chunking is disabled for this workload or role",
        )?;
        require(
            !texts.is_empty() && texts.len() == boundaries.len(),
            "late embedding requires one boundary list per nonempty input batch",
        )?;
        // Validate every input before dispatching the first one; the underlying
        // legacy client skips empty boundary lists, which cannot yield a receipt.
        for (text, spans) in texts.iter().zip(boundaries) {
            let length = text.chars().count();
            require(
                !spans.is_empty()
                    && spans
                        .iter()
                        .all(|&(start, end)| start < end && end <= length),
                "late boundaries must be nonempty character ranges within each input",
            )?;
        }
        let output = self
            .inner
            .embed_late_chunked(texts, &self.binding.task, boundaries)
            .await?;
        Ok(BoundEmbeddingResult {
            output,
            binding: self.binding.clone(),
        })
    }
}
