use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ArtifactManifest, ContractError, Md5Digest, hex, label, require};

/// Portable profile records are bounded separately from the full manifest.
pub const MAX_PROFILE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SpaceId(String);

impl TryFrom<String> for SpaceId {
    type Error = ContractError;

    /// Require a complete lowercase SHA-256 encoding for a behavior-space identity.
    fn try_from(value: String) -> Result<Self, Self::Error> {
        require(
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "space ID must contain exactly 64 lowercase hexadecimal digits",
        )?;
        Ok(Self(value))
    }
}

impl From<SpaceId> for String {
    /// Transfer the validated space identity to its wire representation.
    fn from(value: SpaceId) -> Self {
        value.0
    }
}

impl SpaceId {
    /// Borrow the validated lowercase space identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpaceFormat {
    #[serde(rename = "hades-embedding-space-v1")]
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalization {
    None,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distance {
    Cosine,
    Dot,
    Euclidean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorOutput {
    #[serde(rename = "dense-f32")]
    DenseF32,
}

/// Numerical behavior, independent of endpoint, device name, and batch size.
/// Opaque labels must name exact adapter contracts, not mutable presets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Representation {
    pub dimension: u32,
    pub output: VectorOutput,
    pub normalize: Normalization,
    pub pooling: String,
    pub distance: Distance,
    pub inference_precision: String,
    pub implementation: String,
}

/// Fully resolved task behavior; prompts are exact strings, including whitespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskBehavior {
    pub task: String,
    pub prompt: String,
    pub adapter: String,
    pub special_tokens: String,
}

impl TaskBehavior {
    /// Bound exact prompts and require explicit task, adapter, and token-policy labels.
    fn validate(&self) -> Result<(), ContractError> {
        label(&self.task)?;
        label(&self.adapter)?;
        label(&self.special_tokens)?;
        require(
            self.prompt.len() <= 16384,
            "task prompt exceeds 16384 UTF-8 bytes",
        )
    }
}

/// One compatible passage/query pair. Asymmetric tasks share this space;
/// unrelated code and document adapters receive different space IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpace {
    pub format: SpaceFormat,
    pub artifact_md5: Md5Digest,
    pub tokenizer_manifest_md5: Md5Digest,
    pub tokenizer_contract: String,
    pub representation: Representation,
    pub passage: TaskBehavior,
    pub query: TaskBehavior,
}

impl EmbeddingSpace {
    /// Validate dimensions and behavior labels before deriving a space identity.
    pub fn validate(&self) -> Result<(), ContractError> {
        require(
            (1..=8192).contains(&self.representation.dimension),
            "dimension must be in 1..=8192 (current retrieval limit)",
        )?;
        label(&self.tokenizer_contract)?;
        label(&self.representation.pooling)?;
        label(&self.representation.inference_precision)?;
        label(&self.representation.implementation)?;
        self.passage.validate()?;
        self.query.validate()
    }

    /// SHA-256 identifies the canonical behavior document. Artifact identity
    /// remains MD5; this digest does not authenticate a model or its source.
    pub fn id(&self) -> Result<SpaceId, ContractError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|e| ContractError(e.to_string()))?;
        Ok(SpaceId(hex(&Sha256::digest(
            crate::canonical_json::to_vec(&value),
        ))))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileFormat {
    #[serde(rename = "hades-embedding-profile-v1")]
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileState {
    Initializing,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub id: String,
    pub revision: String,
    pub manifest_md5: Md5Digest,
    /// Exact subset of manifest paths needed by the tokenizer. Never inferred
    /// from filenames; the registering adapter must declare these dependencies.
    pub tokenizer_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub text: bool,
    pub image_embedding: bool,
    pub specialized_code: bool,
    pub late_chunking: bool,
    pub tokenizer_offsets: bool,
    pub tasks: Vec<String>,
    pub max_input_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChunkPolicy {
    Conventional {
        passage_tokens: u32,
        overlap_tokens: u32,
    },
    Late {
        context_cap_tokens: u32,
        minimum_window_overlap_tokens: u32,
        /// Explicit document passages; absent for legacy caller-supplied/AST boundaries.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        passages: Option<PassagePolicy>,
    },
}

/// Retrieval passage sizes are independent of their larger encoding windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassagePolicy {
    pub passage_tokens: u32,
    pub overlap_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub space: EmbeddingSpace,
    pub chunking: ChunkPolicy,
    /// False permits code as ordinary text without claiming a code adapter.
    pub specialized_code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workloads {
    /// Null means disabled, independently of model capabilities.
    pub documents: Option<Workload>,
    pub code: Option<Workload>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    Documents,
    Code,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRole {
    Passage,
    Query,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBinding {
    pub provider_id: String,
}

/// Proposed persisted profile. A missing legacy identity cannot be represented
/// as a ready profile: inventory/adoption must precede construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingProfile {
    pub format: ProfileFormat,
    pub revision: u64,
    pub state: ProfileState,
    pub generation: String,
    pub model: ModelIdentity,
    pub capabilities: Capabilities,
    pub workloads: Workloads,
    pub runtime_binding: RuntimeBinding,
}

impl EmbeddingProfile {
    /// Parse bounded profile JSON and validate it against the supplied artifact manifest.
    pub fn from_json(bytes: &[u8], manifest: &ArtifactManifest) -> Result<Self, ContractError> {
        require(bytes.len() <= MAX_PROFILE_BYTES, "profile exceeds 64 KiB")?;
        let profile: Self = serde_json::from_slice(bytes)
            .map_err(|e| ContractError(format!("invalid profile: {e}")))?;
        profile.validate(manifest)?;
        Ok(profile)
    }

    /// Check artifact/tokenizer consistency, capabilities, tasks, and chunk budgets.
    /// This validates declared data; it does not verify a running provider.
    pub fn validate(&self, manifest: &ArtifactManifest) -> Result<(), ContractError> {
        require(
            serde_json::to_vec(self)
                .map_err(|e| ContractError(e.to_string()))?
                .len()
                <= MAX_PROFILE_BYTES,
            "profile exceeds 64 KiB",
        )?;
        require(self.revision > 0, "profile revision must be positive")?;
        label(&self.generation)?;
        label(&self.model.id)?;
        label(&self.model.revision)?;
        label(&self.runtime_binding.provider_id)?;
        manifest.check_digest(&self.model.manifest_md5)?;
        let mut tokenizer_files = Vec::new();
        for path in &self.model.tokenizer_files {
            let entry = manifest
                .files
                .iter()
                .find(|file| &file.path == path)
                .ok_or_else(|| ContractError("tokenizer asset is absent from manifest".into()))?;
            tokenizer_files.push(entry.clone());
        }
        // Also rejects empty/duplicate tokenizer declarations.
        let tokenizer_md5 = ArtifactManifest::new(tokenizer_files)?.digest()?;
        require(
            self.capabilities.text,
            "document/code workloads require text capability",
        )?;
        require(
            self.capabilities.max_input_tokens > 0,
            "context token limit must be positive",
        )?;
        let mut tasks = std::collections::BTreeSet::new();
        for task in &self.capabilities.tasks {
            label(task)?;
            require(tasks.insert(task), "duplicate supported task")?;
        }
        require(
            self.workloads.documents.is_some() || self.workloads.code.is_some(),
            "enable at least one workload",
        )?;
        for (kind, workload) in [
            (WorkloadKind::Documents, &self.workloads.documents),
            (WorkloadKind::Code, &self.workloads.code),
        ] {
            let Some(workload) = workload else { continue };
            workload.space.validate()?;
            require(
                workload.space.artifact_md5 == self.model.manifest_md5,
                "workload artifact differs from database model",
            )?;
            require(
                workload.space.tokenizer_manifest_md5 == tokenizer_md5,
                "workload tokenizer differs from declared tokenizer assets",
            )?;
            require(
                tasks.contains(&workload.space.passage.task)
                    && tasks.contains(&workload.space.query.task),
                "workload uses an unsupported task",
            )?;
            require(
                !workload.specialized_code
                    || (kind == WorkloadKind::Code && self.capabilities.specialized_code),
                "specialized code capability is unavailable",
            )?;
            let (tokens, overlap) = match workload.chunking {
                ChunkPolicy::Conventional {
                    passage_tokens,
                    overlap_tokens,
                } => (passage_tokens, overlap_tokens),
                ChunkPolicy::Late {
                    context_cap_tokens,
                    minimum_window_overlap_tokens,
                    passages,
                } => {
                    require(
                        self.capabilities.late_chunking && self.capabilities.tokenizer_offsets,
                        "late chunking requires late-chunk and tokenizer-offset capabilities",
                    )?;
                    if let Some(passages) = passages {
                        require(
                            passages.passage_tokens > 0
                                && passages.passage_tokens <= context_cap_tokens
                                && passages.overlap_tokens < passages.passage_tokens,
                            "late passage policy must fit its context and exceed passage overlap",
                        )?;
                    }
                    (context_cap_tokens, minimum_window_overlap_tokens)
                }
            };
            require(
                tokens > 0 && tokens <= self.capabilities.max_input_tokens && overlap < tokens,
                "chunk token budget must fit context and exceed overlap",
            )?;
        }
        Ok(())
    }

    /// Resolve a request from a validated ready profile and observed receipt.
    /// The caller must obtain the provider from the trusted registry; matching
    /// fields alone do not prove that a provider loaded the declared artifacts.
    pub fn bind(
        &self,
        manifest: &ArtifactManifest,
        kind: WorkloadKind,
        role: EmbeddingRole,
        observed: &ProviderReceipt,
    ) -> Result<BoundEmbeddingRequest, ContractError> {
        self.validate(manifest)?;
        require(
            self.state == ProfileState::Ready,
            "workspace profile is not ready",
        )?;
        let workload = match kind {
            WorkloadKind::Documents => &self.workloads.documents,
            WorkloadKind::Code => &self.workloads.code,
        }
        .as_ref()
        .ok_or_else(|| ContractError("workload is disabled_by_profile".into()))?;
        observed.validate()?;
        require(
            observed.artifact_md5 == self.model.manifest_md5,
            "provider artifact mismatch; reverify the registered provider",
        )?;
        require(
            observed.space_id == workload.space.id()?,
            "provider embedding space mismatch; incompatible task or representation",
        )?;
        let task = match role {
            EmbeddingRole::Passage => &workload.space.passage.task,
            EmbeddingRole::Query => &workload.space.query.task,
        };
        Ok(BoundEmbeddingRequest {
            profile_revision: self.revision,
            generation: self.generation.clone(),
            provider_id: self.runtime_binding.provider_id.clone(),
            model: self.model.id.clone(),
            dimension: workload.space.representation.dimension,
            task: task.clone(),
            expected: observed.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptFormat {
    #[serde(rename = "hades-provider-receipt-v1")]
    V1,
}

/// Issued only after a trusted adapter loads a verified immutable snapshot.
/// A new load/reload must use a fresh serving generation, even for equal bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderReceipt {
    pub format: ReceiptFormat,
    pub artifact_md5: Md5Digest,
    pub space_id: SpaceId,
    pub serving_generation: String,
}

impl ProviderReceipt {
    /// Require a bounded serving-generation label in addition to typed digest syntax.
    pub fn validate(&self) -> Result<(), ContractError> {
        label(&self.serving_generation)
    }

    /// Use at request admission and again on the response. A changed serving
    /// generation requires a fresh binding, never acceptance under the old one.
    pub fn check_matches(&self, expected: &Self) -> Result<(), ContractError> {
        self.validate()?;
        expected.validate()?;
        require(
            self == expected,
            "provider receipt changed; discard vectors and reverify binding",
        )
    }
}

/// Resolved model and dimension have no Jina defaults. This is a planning
/// value, not a connected client or authorization to persist vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundEmbeddingRequest {
    pub profile_revision: u64,
    pub generation: String,
    pub provider_id: String,
    pub model: String,
    pub dimension: u32,
    pub task: String,
    pub expected: ProviderReceipt,
}
