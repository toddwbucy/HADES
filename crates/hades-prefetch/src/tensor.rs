//! Tensor serialization and IPC for the training pipeline.
//!
//! Serializes [`GraphData`] plus edge splits and negative samples into
//! the [safetensors](https://huggingface.co/docs/safetensors) format.
//! The file can be memory-mapped for zero-copy reads by the training loop.
//!
//! ## Safetensors layout
//!
//! **Tensors:**
//! - `node_features`    — `F32  [N, D]`
//! - `has_embedding`    — `BOOL [N]`
//! - `node_collections` — `U32  [N]`
//! - `edge_src`         — `U32  [E]`
//! - `edge_dst`         — `U32  [E]`
//! - `edge_type`        — `U32  [E]`
//! - `train_idx`        — `U32  [E_train]`
//! - `val_idx`          — `U32  [E_val]`
//! - `test_idx`         — `U32  [E_test]`
//! - `neg_src`          — `U32  [E_neg]`
//! - `neg_dst`          — `U32  [E_neg]`
//!
//! **Metadata (JSON header):**
//! - `num_nodes`, `num_edges`, `num_relations`, `feature_dim`
//! - `collection_names` (JSON array)
//! - `val_ratio`, `test_ratio`, `neg_sampling_ratio`

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use tracing::info;

use hades_core::graph::types::GraphData;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("graph has no edges — cannot split or sample")]
    EmptyGraph,

    #[error("tensor '{name}' not found in safetensors file")]
    MissingTensor { name: String },

    #[error("metadata key '{key}' not found")]
    MissingMetadata { key: String },

    #[error("metadata parse error for '{key}': {message}")]
    MetadataParse { key: String, message: String },

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor '{name}' has dtype {actual:?}, expected {expected:?}")]
    DtypeMismatch {
        name: String,
        expected: Dtype,
        actual: Dtype,
    },

    #[error("invalid split config: val_ratio ({val}) + test_ratio ({test}) = {sum} > 1.0")]
    InvalidSplitConfig { val: f64, test: f64, sum: f64 },

    #[error("tensor '{name}' has {len} bytes, which is not a whole number of 4-byte values")]
    RaggedTensor { name: String, len: usize },

    #[error("serialization validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("invalid neg_sampling_ratio: {neg} (must be finite and positive)")]
    InvalidNegSamplingRatio { neg: f64 },
}

// ---------------------------------------------------------------------------
// Edge split configuration
// ---------------------------------------------------------------------------

/// Configuration for train/val/test edge splitting and negative sampling.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Fraction of edges for validation (default: 0.1).
    pub val_ratio: f64,
    /// Fraction of edges for test (default: 0.1).
    pub test_ratio: f64,
    /// Ratio of negative to positive samples (default: 1.0).
    pub neg_sampling_ratio: f64,
    /// Seed for reproducible pair partitions and initial negative samples.
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge split result
// ---------------------------------------------------------------------------

/// Train/val/test edge split — indices into the edge arrays.
#[derive(Debug, Clone)]
pub struct EdgeSplit {
    /// Indices of training edges.
    pub train_idx: Vec<u32>,
    /// Indices of validation edges.
    pub val_idx: Vec<u32>,
    /// Indices of test edges.
    pub test_idx: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Negative samples
// ---------------------------------------------------------------------------

/// Negative edge samples (non-existing node pairs).
#[derive(Debug, Clone)]
pub struct NegativeSamples {
    /// Source node indices.
    pub src: Vec<u32>,
    /// Destination node indices.
    pub dst: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Edge splitting — random permutation
// ---------------------------------------------------------------------------

/// Split edges into train/val/test sets via random permutation.
///
/// Matches Python `RGCNTrainer._split_edges()`.
pub fn split_edges(num_edges: usize, config: &SplitConfig) -> Result<EdgeSplit, TensorError> {
    if num_edges > u32::MAX as usize {
        return Err(TensorError::ValidationFailed {
            message: "edge count exceeds u32 index space".into(),
        });
    }
    if num_edges == 0 {
        return Err(TensorError::EmptyGraph);
    }

    let sum = config.val_ratio + config.test_ratio;
    if !config.val_ratio.is_finite()
        || !config.test_ratio.is_finite()
        || sum > 1.0
        || config.val_ratio < 0.0
        || config.test_ratio < 0.0
    {
        return Err(TensorError::InvalidSplitConfig {
            val: config.val_ratio,
            test: config.test_ratio,
            sum,
        });
    }

    let mut perm: Vec<u32> = (0..num_edges as u32).collect();
    perm.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));

    let val_size = (num_edges as f64 * config.val_ratio) as usize;
    let test_size = (num_edges as f64 * config.test_ratio) as usize;
    let train_size = num_edges - val_size - test_size;

    if train_size == 0 || val_size == 0 || test_size == 0 {
        return Err(TensorError::ValidationFailed {
            message: format!(
                "training requires nonempty train/validation/test partitions; {num_edges} pair groups produce {train_size}/{val_size}/{test_size}; increase graph size or adjust ratios"
            ),
        });
    }

    let train_idx = perm[..train_size].to_vec();
    let val_idx = perm[train_size..train_size + val_size].to_vec();
    let test_idx = perm[train_size + val_size..].to_vec();

    info!(
        train = train_size,
        val = val_size,
        test = test_size,
        "edge split"
    );

    Ok(EdgeSplit {
        train_idx,
        val_idx,
        test_idx,
    })
}

// ---------------------------------------------------------------------------
// Negative sampling — rejection sampling
// ---------------------------------------------------------------------------

/// Partition unordered endpoint pairs, keeping duplicate and inverse relations
/// together. The link decoder scores endpoints without relation types, so even
/// differently typed edges between the same endpoints must share a split.
pub fn split_graph_edges(
    graph: &GraphData,
    config: &SplitConfig,
) -> Result<EdgeSplit, TensorError> {
    let mut groups: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();
    for (index, (&src, &dst)) in graph.edge_src.iter().zip(&graph.edge_dst).enumerate() {
        groups
            .entry((src.min(dst), src.max(dst)))
            .or_default()
            .push(index as u32);
    }
    let groups: Vec<_> = groups.into_values().collect();
    let partition = split_edges(groups.len(), config)?;
    let expand = |indices: &[u32]| {
        indices
            .iter()
            .flat_map(|&i| groups[i as usize].iter().copied())
            .collect()
    };
    Ok(EdgeSplit {
        train_idx: expand(&partition.train_idx),
        val_idx: expand(&partition.val_idx),
        test_idx: expand(&partition.test_idx),
    })
}

/// Largest pair of u32 vectors whose combined byte length fits isize.
const MAX_NEGATIVE_SAMPLES: usize = isize::MAX as usize / (2 * std::mem::size_of::<u32>());

pub fn negative_sample_count(positives: usize, ratio: f64) -> Result<usize, TensorError> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(TensorError::InvalidNegSamplingRatio { neg: ratio });
    }
    let count = positives as f64 * ratio;
    if !count.is_finite()
        || count < 1.0
        || count > MAX_NEGATIVE_SAMPLES as f64
        || count as usize > MAX_NEGATIVE_SAMPLES
    {
        return Err(TensorError::ValidationFailed {
            message:
                "negative sample count is zero, nonfinite, or exceeds supported vector capacity"
                    .into(),
        });
    }
    Ok(count as usize)
}

/// Generate negative edge samples via rejection sampling.
///
/// Matches Python `RGCNTrainer._negative_sample()`. Samples node pairs
/// that are **not** existing edges, their inverses, or self-loops. The exclusion
/// set is built from the full edge set (not just train edges).
pub fn negative_sample(graph: &GraphData, num_neg: usize) -> Result<NegativeSamples, TensorError> {
    negative_sample_seeded(graph, num_neg, 0)
}

/// Deterministic bounded sampling; never return fewer samples than requested.
pub fn negative_sample_seeded(
    graph: &GraphData,
    num_neg: usize,
    seed: u64,
) -> Result<NegativeSamples, TensorError> {
    if num_neg == 0 {
        return Ok(NegativeSamples {
            src: Vec::new(),
            dst: Vec::new(),
        });
    }
    if num_neg > MAX_NEGATIVE_SAMPLES {
        return Err(TensorError::ValidationFailed {
            message: "negative sample count exceeds supported vector capacity".into(),
        });
    }
    if graph.num_nodes < 2 || graph.num_nodes > u32::MAX as usize {
        return Err(TensorError::ValidationFailed {
            message: "negative sampling requires 2..=u32::MAX nodes".into(),
        });
    }

    // Build existing edge set for O(1) lookup
    let mut existing: HashSet<(u32, u32)> = HashSet::with_capacity(graph.num_edges);
    for (&s, &d) in graph.edge_src.iter().zip(&graph.edge_dst) {
        existing.insert((s, d));
        existing.insert((d, s));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut neg_src = Vec::new();
    let mut neg_dst = Vec::new();
    for values in [&mut neg_src, &mut neg_dst] {
        values
            .try_reserve_exact(num_neg)
            .map_err(|error| TensorError::ValidationFailed {
                message: format!("cannot allocate negative samples: {error}"),
            })?;
    }
    let max_attempts = num_neg.saturating_mul(10);
    let mut attempts = 0;
    let num_nodes = graph.num_nodes as u32;

    while neg_src.len() < num_neg && attempts < max_attempts {
        // Generate candidates in batches for efficiency
        let batch_size = num_neg.min(1024).min(max_attempts - attempts);
        for _ in 0..batch_size {
            let s: u32 = rng.random_range(0..num_nodes);
            let d: u32 = rng.random_range(0..num_nodes);
            if s != d && !existing.contains(&(s, d)) {
                neg_src.push(s);
                neg_dst.push(d);
                if neg_src.len() >= num_neg {
                    break;
                }
            }
        }
        attempts += batch_size;
    }

    if neg_src.len() != num_neg {
        return Err(TensorError::ValidationFailed {
            message: format!(
                "insufficient negative samples: requested {num_neg}, found {} within {max_attempts} attempts; graph may be too dense",
                neg_src.len()
            ),
        });
    }
    Ok(NegativeSamples {
        src: neg_src,
        dst: neg_dst,
    })
}

// ---------------------------------------------------------------------------
// Serialization — GraphData + splits → safetensors
// ---------------------------------------------------------------------------

/// Helper: reinterpret a `&[T]` as `&[u8]`.
///
/// # Safety assumption
/// Safetensors uses little-endian byte order. This transmute is correct
/// on little-endian platforms (x86, ARM LE, RISC-V LE). The compile-time
/// assertion below prevents silent corruption on big-endian targets.
fn as_bytes<T>(slice: &[T]) -> &[u8] {
    const {
        assert!(
            cfg!(target_endian = "little"),
            "safetensors requires little-endian"
        )
    }
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn graph_contract_json(graph: &GraphData) -> Result<String, TensorError> {
    let contract = graph
        .contract
        .as_ref()
        .ok_or_else(|| TensorError::ValidationFailed {
            message: "graph has no semantic contract; reload through the schema-aware loader"
                .into(),
        })?;
    contract
        .validate(graph)
        .map_err(|message| TensorError::ValidationFailed { message })?;
    Ok(serde_json::to_string(contract)?)
}

/// Serialize a graph with edge splits and negative samples to safetensors format.
///
/// Returns the serialized bytes. Use [`serialize_to_file`] to write directly
/// to disk.
pub fn serialize_graph(
    graph: &GraphData,
    split: &EdgeSplit,
    neg: &NegativeSamples,
    config: &SplitConfig,
) -> Result<Vec<u8>, TensorError> {
    // Validate split indices are within edge bounds
    let num_edges = graph.num_edges as u32;
    for &idx in split
        .train_idx
        .iter()
        .chain(&split.val_idx)
        .chain(&split.test_idx)
    {
        if idx >= num_edges {
            return Err(TensorError::ValidationFailed {
                message: format!("split index {idx} >= num_edges {}", graph.num_edges),
            });
        }
    }

    // Validate negative sample consistency and node bounds
    if neg.src.len() != neg.dst.len() {
        return Err(TensorError::ValidationFailed {
            message: format!(
                "neg_src length ({}) != neg_dst length ({})",
                neg.src.len(),
                neg.dst.len()
            ),
        });
    }
    let num_nodes = graph.num_nodes as u32;
    for (&s, &d) in neg.src.iter().zip(&neg.dst) {
        if s >= num_nodes || d >= num_nodes {
            return Err(TensorError::ValidationFailed {
                message: format!("negative sample ({s}, {d}) out of bounds for {num_nodes} nodes"),
            });
        }
    }

    // Convert has_embedding Vec<bool> → Vec<u8> (safetensors BOOL is 1 byte)
    let has_emb_u8: Vec<u8> = graph.has_embedding.iter().map(|&b| b as u8).collect();

    let tensors: Vec<(&str, TensorView<'_>)> = vec![
        (
            "node_features",
            TensorView::new(
                Dtype::F32,
                vec![graph.num_nodes, graph.feature_dim],
                as_bytes(&graph.node_features),
            )?,
        ),
        (
            "has_embedding",
            TensorView::new(Dtype::BOOL, vec![graph.num_nodes], &has_emb_u8)?,
        ),
        (
            "node_collections",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_nodes],
                as_bytes(&graph.node_collections),
            )?,
        ),
        (
            "edge_src",
            TensorView::new(Dtype::U32, vec![graph.num_edges], as_bytes(&graph.edge_src))?,
        ),
        (
            "edge_dst",
            TensorView::new(Dtype::U32, vec![graph.num_edges], as_bytes(&graph.edge_dst))?,
        ),
        (
            "edge_type",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_edges],
                as_bytes(&graph.edge_type),
            )?,
        ),
        (
            "train_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.train_idx.len()],
                as_bytes(&split.train_idx),
            )?,
        ),
        (
            "val_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.val_idx.len()],
                as_bytes(&split.val_idx),
            )?,
        ),
        (
            "test_idx",
            TensorView::new(
                Dtype::U32,
                vec![split.test_idx.len()],
                as_bytes(&split.test_idx),
            )?,
        ),
        (
            "neg_src",
            TensorView::new(Dtype::U32, vec![neg.src.len()], as_bytes(&neg.src))?,
        ),
        (
            "neg_dst",
            TensorView::new(Dtype::U32, vec![neg.dst.len()], as_bytes(&neg.dst))?,
        ),
    ];

    // Metadata: scalars + collection_names as JSON in the header
    let mut metadata = HashMap::new();
    metadata.insert("graph_contract".into(), graph_contract_json(graph)?);
    metadata.insert("num_nodes".into(), graph.num_nodes.to_string());
    metadata.insert("num_edges".into(), graph.num_edges.to_string());
    metadata.insert("num_relations".into(), graph.num_relations.to_string());
    metadata.insert("feature_dim".into(), graph.feature_dim.to_string());
    metadata.insert(
        "collection_names".into(),
        serde_json::to_string(&graph.collection_names)?,
    );
    metadata.insert("sampling_seed".into(), config.seed.to_string());
    metadata.insert("val_ratio".into(), config.val_ratio.to_string());
    metadata.insert("test_ratio".into(), config.test_ratio.to_string());
    metadata.insert(
        "neg_sampling_ratio".into(),
        config.neg_sampling_ratio.to_string(),
    );

    let bytes = safetensors::tensor::serialize(tensors, Some(metadata))?;
    Ok(bytes)
}

/// Serialize graph structure only (no edge splits or negative samples).
///
/// Produces a safetensors file suitable for inference / embedding update:
/// the GPU service reads the core tensors (`node_features`, `edge_src`,
/// `edge_dst`, `edge_type`, `has_embedding`, `node_collections`) and
/// metadata, then runs a forward pass through a loaded checkpoint.
pub fn serialize_graph_for_inference(graph: &GraphData) -> Result<Vec<u8>, TensorError> {
    let has_emb_u8: Vec<u8> = graph.has_embedding.iter().map(|&b| b as u8).collect();

    let tensors: Vec<(&str, TensorView<'_>)> = vec![
        (
            "node_features",
            TensorView::new(
                Dtype::F32,
                vec![graph.num_nodes, graph.feature_dim],
                as_bytes(&graph.node_features),
            )?,
        ),
        (
            "has_embedding",
            TensorView::new(Dtype::BOOL, vec![graph.num_nodes], &has_emb_u8)?,
        ),
        (
            "node_collections",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_nodes],
                as_bytes(&graph.node_collections),
            )?,
        ),
        (
            "edge_src",
            TensorView::new(Dtype::U32, vec![graph.num_edges], as_bytes(&graph.edge_src))?,
        ),
        (
            "edge_dst",
            TensorView::new(Dtype::U32, vec![graph.num_edges], as_bytes(&graph.edge_dst))?,
        ),
        (
            "edge_type",
            TensorView::new(
                Dtype::U32,
                vec![graph.num_edges],
                as_bytes(&graph.edge_type),
            )?,
        ),
    ];

    let mut metadata = HashMap::new();
    metadata.insert("graph_contract".into(), graph_contract_json(graph)?);
    metadata.insert("num_nodes".into(), graph.num_nodes.to_string());
    metadata.insert("num_edges".into(), graph.num_edges.to_string());
    metadata.insert("num_relations".into(), graph.num_relations.to_string());
    metadata.insert("feature_dim".into(), graph.feature_dim.to_string());
    metadata.insert(
        "collection_names".into(),
        serde_json::to_string(&graph.collection_names)?,
    );
    metadata.insert("mode".into(), "inference".into());

    let bytes = safetensors::tensor::serialize(tensors, Some(metadata))?;
    Ok(bytes)
}

/// Write inference-only safetensors to disk (atomic write).
///
/// Uses a unique temp file in the target directory to avoid collisions
/// with concurrent writers, then atomically persists to the final path.
pub fn serialize_graph_for_inference_to_file(
    path: &Path,
    graph: &GraphData,
) -> Result<(), TensorError> {
    let bytes = serialize_graph_for_inference(graph)?;

    let dir = path.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(&bytes)?;
    tmp.flush()?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(std::io::Error::from)?;

    // `NamedTempFile` creates the file mode 0600 (owner-only). The training
    // service runs as a separate user and must *read* this file via LoadGraph,
    // so widen it to 0644. Without this, cross-user `graph-embed update` fails
    // at LoadGraph with a permission error surfaced as FileNotFoundError.
    //
    // World-read (0o004) is deliberate, not drift: when the CLI user's group
    // doesn't match the service's (the common case, unless the checkpoint dir
    // is setgid to the service group), the world bit is the *only* thing that
    // lets the service read the file. The sibling training artifact
    // (`serialize_to_file` via `File::create`, typically 0664) is likewise
    // world-readable. Restricting to 0640 would reintroduce this bug. The file
    // is a transient inference graph in the checkpoint dir; its node features
    // already live in the database the operator can read.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644))?;
    }

    info!(
        path = %path.display(),
        size_mb = bytes.len() as f64 / (1024.0 * 1024.0),
        "wrote inference safetensors file"
    );

    Ok(())
}

/// Serialize a graph to a safetensors file on disk.
///
/// Uses atomic write semantics: data is written to a temporary file in the
/// same directory, flushed, then renamed to the target path. This prevents
/// concurrent `MappedGraph::open()` from seeing a partial file.
pub fn serialize_to_file(
    path: &Path,
    graph: &GraphData,
    split: &EdgeSplit,
    neg: &NegativeSamples,
    config: &SplitConfig,
) -> Result<(), TensorError> {
    let bytes = serialize_graph(graph, split, neg, config)?;

    // Write to a temp file, then atomically rename
    let tmp_path = path.with_extension("safetensors.tmp");
    let mut file = fs::File::create(&tmp_path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp_path, path)?;

    info!(
        path = %path.display(),
        size_mb = bytes.len() as f64 / (1024.0 * 1024.0),
        "wrote safetensors file"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-mapped reader
// ---------------------------------------------------------------------------

/// Memory-mapped safetensors file for zero-copy tensor access.
///
/// The file is mapped into memory and tensors are accessed as byte slices
/// without copying. The `MappedGraph` holds the mmap and provides typed
/// accessors for the training loop.
///
/// # Safety
///
/// The underlying file must not be modified or truncated while the map is
/// active. External modification can cause SIGBUS or undefined behavior.
/// [`serialize_to_file`] uses atomic write (write-to-tmp + rename) to
/// reduce risk, but concurrent non-atomic writers or file truncation can
/// still corrupt the mapping. For full safety, use exclusive file locking
/// or open a read-only snapshot.
pub struct MappedGraph {
    mmap: Mmap,
}

impl MappedGraph {
    /// Open a safetensors file via memory mapping.
    ///
    /// # Safety
    ///
    /// The caller must ensure the file is not concurrently modified.
    /// See [`MappedGraph`] for details.
    pub fn open(path: &Path) -> Result<Self, TensorError> {
        let file = fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// Get the raw safetensors data.
    fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Parse the safetensors header and access tensors.
    pub fn tensors(&self) -> Result<SafeTensors<'_>, TensorError> {
        Ok(SafeTensors::deserialize(self.data())?)
    }

    /// Read a U32 tensor as a `Vec<u32>`.
    pub fn read_u32_tensor(&self, name: &str) -> Result<Vec<u32>, TensorError> {
        let st = self.tensors()?;
        let view = st.tensor(name).map_err(|_| TensorError::MissingTensor {
            name: name.to_string(),
        })?;
        if view.dtype() != Dtype::U32 {
            return Err(TensorError::DtypeMismatch {
                name: name.to_string(),
                expected: Dtype::U32,
                actual: view.dtype(),
            });
        }
        let bytes = view.data();
        // `as_chunks().0` discards a ragged tail with no diagnostic, so the
        // discard site says out loud what it assumes.
        //
        // Not the primary defence, and the comment first written here claimed
        // it was. safetensors validates on deserialize that a tensor's byte
        // range equals shape times dtype width (`e - s != size` ->
        // `TensorInvalidInfo`), so a U32 view reaching this point already has a
        // length divisible by four and this branch is unreachable today. It is
        // kept because the assumption belongs next to the code that depends on
        // it: if the dtype check above ever widens, an unguarded `as_chunks`
        // returns a short Vec with Ok and breaks the caller's
        // `len == count * width` invariant far from the cause.
        if !bytes.len().is_multiple_of(4) {
            return Err(TensorError::RaggedTensor {
                name: name.to_string(),
                len: bytes.len(),
            });
        }
        let result: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        Ok(result)
    }

    /// Read the F32 node features as a flat `Vec<f32>`.
    pub fn read_node_features(&self) -> Result<Vec<f32>, TensorError> {
        let st = self.tensors()?;
        let view = st
            .tensor("node_features")
            .map_err(|_| TensorError::MissingTensor {
                name: "node_features".into(),
            })?;
        if view.dtype() != Dtype::F32 {
            return Err(TensorError::DtypeMismatch {
                name: "node_features".into(),
                expected: Dtype::F32,
                actual: view.dtype(),
            });
        }
        let bytes = view.data();
        // `as_chunks().0` discards a ragged tail with no diagnostic, so the
        // discard site says out loud what it assumes.
        //
        // Not the primary defence, and the comment first written here claimed
        // it was. safetensors validates on deserialize that a tensor's byte
        // range equals shape times dtype width (`e - s != size` ->
        // `TensorInvalidInfo`), so a U32 view reaching this point already has a
        // length divisible by four and this branch is unreachable today. It is
        // kept because the assumption belongs next to the code that depends on
        // it: if the dtype check above ever widens, an unguarded `as_chunks`
        // returns a short Vec with Ok and breaks the caller's
        // `len == count * width` invariant far from the cause.
        if !bytes.len().is_multiple_of(4) {
            return Err(TensorError::RaggedTensor {
                name: "node_features".into(),
                len: bytes.len(),
            });
        }
        let result: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(result)
    }

    /// Read a metadata value from the safetensors header.
    pub fn metadata_value(&self, key: &str) -> Result<String, TensorError> {
        let (_, meta) = SafeTensors::read_metadata(self.data())?;
        meta.metadata()
            .as_ref()
            .and_then(|m: &HashMap<String, String>| m.get(key))
            .cloned()
            .ok_or_else(|| TensorError::MissingMetadata {
                key: key.to_string(),
            })
    }

    /// Read num_nodes from metadata.
    pub fn num_nodes(&self) -> Result<usize, TensorError> {
        self.metadata_value("num_nodes")?
            .parse()
            .map_err(|e| TensorError::MetadataParse {
                key: "num_nodes".into(),
                message: format!("{e}"),
            })
    }

    /// Read num_edges from metadata.
    pub fn num_edges(&self) -> Result<usize, TensorError> {
        self.metadata_value("num_edges")?
            .parse()
            .map_err(|e| TensorError::MetadataParse {
                key: "num_edges".into(),
                message: format!("{e}"),
            })
    }

    /// Read collection_names from metadata.
    pub fn collection_names(&self) -> Result<Vec<String>, TensorError> {
        let raw = self.metadata_value("collection_names")?;
        serde_json::from_str(&raw).map_err(|e| TensorError::MetadataParse {
            key: "collection_names".into(),
            message: format!("{e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience: full pipeline
// ---------------------------------------------------------------------------

/// Load graph, compute splits + negatives, and serialize to safetensors.
///
/// This is the high-level entry point combining all steps. The `IDMap` is
/// not serialized (it's only needed for embedding export, handled separately).
pub fn prepare_and_serialize(
    path: &Path,
    graph: &GraphData,
    config: &SplitConfig,
) -> Result<EdgeSplit, TensorError> {
    if !config.neg_sampling_ratio.is_finite() || config.neg_sampling_ratio <= 0.0 {
        return Err(TensorError::InvalidNegSamplingRatio {
            neg: config.neg_sampling_ratio,
        });
    }

    let split = split_graph_edges(graph, config)?;

    // Negative samples: 1 per positive train edge by default
    let num_neg = negative_sample_count(split.train_idx.len(), config.neg_sampling_ratio)?;
    let neg = negative_sample_seeded(graph, num_neg, config.seed)?;

    serialize_to_file(path, graph, &split, &neg, config)?;
    Ok(split)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Local test fixtures matching the historical NL ontology shape.
    const JINA_DIM: usize = 2048;
    const NUM_RELATIONS: usize = 22;

    /// Build a small test graph for unit tests.
    fn test_graph() -> GraphData {
        let num_nodes = 10;
        let mut graph = GraphData::with_capacity(num_nodes, 0);

        // Add some edges across different relation types
        let edges = [
            (0, 1, 0),
            (1, 2, 0),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 2),
            (5, 6, 3),
            (6, 7, 4),
            (7, 8, 5),
            (8, 9, 6),
            (0, 9, 7),
            (1, 5, 8),
            (2, 8, 9),
            (3, 7, 10),
            (4, 6, 11),
            (5, 0, 0),
            (6, 1, 1),
            (7, 2, 2),
            (8, 3, 3),
            (9, 4, 4),
            (0, 5, 5),
        ];
        for &(s, d, r) in &edges {
            graph.add_edge(s, d, r);
        }

        // Set collection names and indices
        graph.collection_names = vec!["col_a".into(), "col_b".into()];
        for i in 0..num_nodes {
            graph.node_collections[i] = (i % 2) as u32;
        }

        // Set a few embeddings
        let emb = vec![1.0f32; JINA_DIM];
        graph.set_node_features(0, &emb);
        graph.set_node_features(3, &emb);
        graph.set_node_features(7, &emb);
        graph.contract = Some(hades_core::graph::types::GraphContract {
            version: 1,
            relation_order: (0..graph.num_relations)
                .map(|i| format!("fixture_rel_{i}"))
                .collect(),
            collection_names: graph.collection_names.clone(),
            feature_dim: graph.feature_dim,
            architecture: "rgcn".into(),
            feature_policy: hades_core::graph::types::GraphContract::FEATURE_POLICY.into(),
            feature_models: graph
                .collection_names
                .iter()
                .map(|name| (name.clone(), vec!["fixture:v1".into()]))
                .collect(),
        });

        graph
    }

    #[test]
    fn test_split_edges_ratios() {
        let split = split_edges(100, &SplitConfig::default()).unwrap();
        assert_eq!(
            split.train_idx.len() + split.val_idx.len() + split.test_idx.len(),
            100
        );
        assert_eq!(split.val_idx.len(), 10);
        assert_eq!(split.test_idx.len(), 10);
        assert_eq!(split.train_idx.len(), 80);
    }

    #[test]
    fn grouped_split_keeps_inverse_and_duplicate_pairs_together() {
        let mut graph = test_graph();
        graph.add_edge(0, 1, 0);
        graph.add_edge(1, 0, 1);
        let split = split_graph_edges(&graph, &SplitConfig::default()).unwrap();
        let mut owner = vec![usize::MAX; graph.num_edges];
        for (group, indices) in [&split.train_idx, &split.val_idx, &split.test_idx]
            .iter()
            .enumerate()
        {
            for &index in indices.iter() {
                assert_eq!(owner[index as usize], usize::MAX);
                owner[index as usize] = group;
            }
        }
        assert!(!owner.contains(&usize::MAX));
        assert_eq!(owner[0], owner[graph.num_edges - 1]);
        assert_eq!(owner[0], owner[graph.num_edges - 2]);
    }

    #[test]
    fn prepared_split_is_exactly_the_serialized_partition() {
        let graph = test_graph();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("partition.safetensors");
        let split = prepare_and_serialize(&path, &graph, &SplitConfig::default()).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let file = SafeTensors::deserialize(&bytes).unwrap();
        for (name, expected) in [
            ("train_idx", split.train_idx),
            ("val_idx", split.val_idx),
            ("test_idx", split.test_idx),
        ] {
            let actual: Vec<u32> = file
                .tensor(name)
                .unwrap()
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| u32::from_le_bytes(*bytes))
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_split_edges_no_overlap() {
        let split = split_edges(50, &SplitConfig::default()).unwrap();
        let mut all: Vec<u32> = Vec::new();
        all.extend(&split.train_idx);
        all.extend(&split.val_idx);
        all.extend(&split.test_idx);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 50, "splits must be disjoint and cover all edges");
    }

    #[test]
    fn test_split_edges_empty() {
        let err = split_edges(0, &SplitConfig::default()).unwrap_err();
        assert!(matches!(err, TensorError::EmptyGraph));
    }

    #[test]
    fn seeded_splits_and_samples_repeat() {
        let graph = test_graph();
        let config = SplitConfig {
            seed: 29,
            ..Default::default()
        };
        let first = split_graph_edges(&graph, &config).unwrap();
        let second = split_graph_edges(&graph, &config).unwrap();
        assert_eq!(first.train_idx, second.train_idx);
        assert_eq!(first.val_idx, second.val_idx);
        let first = negative_sample_seeded(&graph, 20, 29).unwrap();
        let second = negative_sample_seeded(&graph, 20, 29).unwrap();
        assert_eq!(first.src, second.src);
        assert_eq!(first.dst, second.dst);
        for (&src, &dst) in first.src.iter().zip(&first.dst) {
            assert!(
                !graph
                    .edge_src
                    .iter()
                    .zip(&graph.edge_dst)
                    .any(|(&s, &d)| s == dst && d == src)
            );
        }
    }

    #[test]
    fn excessive_sampling_ratios_fail_before_allocation() {
        for ratio in [f64::MAX, 1e100, 0.001] {
            assert!(negative_sample_count(10, ratio).is_err());
        }
        assert!(negative_sample_seeded(&test_graph(), usize::MAX, 0).is_err());
        assert_eq!(negative_sample_count(10, 1.5).unwrap(), 15);
    }

    #[test]
    fn tiny_splits_and_dense_negatives_fail_explicitly() {
        for count in 1..10 {
            assert!(split_edges(count, &SplitConfig::default()).is_err());
        }
        let mut graph = GraphData::with_capacity(3, 0);
        for src in 0..3 {
            for dst in (src + 1)..3 {
                graph.add_edge(src, dst, 0);
            }
        }
        assert!(negative_sample_seeded(&graph, 1, 0).is_err());
        assert!(negative_sample_seeded(&GraphData::with_capacity(1, 0), 1, 0).is_err());
    }

    #[test]
    fn test_negative_sample_basic() {
        let graph = test_graph();
        let neg = negative_sample(&graph, 15).unwrap();
        assert_eq!(neg.src.len(), neg.dst.len());
        assert_eq!(neg.src.len(), 15);

        // No self-loops
        for (&s, &d) in neg.src.iter().zip(&neg.dst) {
            assert_ne!(s, d, "negative sample should not have self-loops");
        }

        // No existing edges
        let existing: HashSet<(u32, u32)> = graph
            .edge_src
            .iter()
            .zip(&graph.edge_dst)
            .map(|(&s, &d)| (s, d))
            .collect();
        for (&s, &d) in neg.src.iter().zip(&neg.dst) {
            assert!(
                !existing.contains(&(s, d)),
                "negative sample ({s}, {d}) is an existing edge"
            );
        }
    }

    #[test]
    fn test_negative_sample_empty() {
        let graph = test_graph();
        let neg = negative_sample(&graph, 0).unwrap();
        assert!(neg.src.is_empty());
        assert!(neg.dst.is_empty());
    }

    #[test]
    fn semantic_contract_survives_both_serialization_modes() {
        let graph = test_graph();
        let split = split_graph_edges(&graph, &SplitConfig::default()).unwrap();
        let negatives = negative_sample(&graph, 10).unwrap();
        let train = serialize_graph(&graph, &split, &negatives, &SplitConfig::default()).unwrap();
        let inference = serialize_graph_for_inference(&graph).unwrap();
        for bytes in [train, inference] {
            let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
            let contract: hades_core::graph::types::GraphContract =
                serde_json::from_str(&meta.metadata().as_ref().unwrap()["graph_contract"]).unwrap();
            assert_eq!(Some(contract), graph.contract);
        }
    }

    #[test]
    fn refuses_serialization_without_verified_contract() {
        let mut graph = test_graph();
        graph.contract = None;
        assert!(serialize_graph_for_inference(&graph).is_err());
        let mut graph = test_graph();
        graph.contract.as_mut().unwrap().collection_names.reverse();
        assert!(serialize_graph_for_inference(&graph).is_err());
        let mut graph = test_graph();
        graph.contract.as_mut().unwrap().feature_models.clear();
        assert!(serialize_graph_for_inference(&graph).is_err());
    }

    #[test]
    fn test_serialize_roundtrip() {
        let graph = test_graph();
        let split = split_edges(graph.num_edges, &SplitConfig::default()).unwrap();
        let neg = negative_sample(&graph, 10).unwrap();
        let config = SplitConfig::default();

        let bytes = serialize_graph(&graph, &split, &neg, &config).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize and verify
        let st = SafeTensors::deserialize(&bytes).unwrap();

        // Check tensor shapes
        let nf = st.tensor("node_features").unwrap();
        assert_eq!(nf.shape(), &[graph.num_nodes, graph.feature_dim]);

        let he = st.tensor("has_embedding").unwrap();
        assert_eq!(he.shape(), &[graph.num_nodes]);

        let es = st.tensor("edge_src").unwrap();
        assert_eq!(es.shape(), &[graph.num_edges]);

        let ti = st.tensor("train_idx").unwrap();
        let vi = st.tensor("val_idx").unwrap();
        let tsi = st.tensor("test_idx").unwrap();
        assert_eq!(
            ti.shape()[0] + vi.shape()[0] + tsi.shape()[0],
            graph.num_edges
        );

        // Check metadata
        let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = meta.metadata().as_ref().unwrap();
        assert_eq!(meta["num_nodes"], graph.num_nodes.to_string());
        assert_eq!(meta["num_edges"], graph.num_edges.to_string());
        assert_eq!(meta["num_relations"], NUM_RELATIONS.to_string());
    }

    #[test]
    fn test_file_roundtrip() {
        let graph = test_graph();
        let config = SplitConfig::default();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        prepare_and_serialize(&path, &graph, &config).unwrap();
        assert!(path.exists());

        // Read back via MappedGraph
        let mapped = MappedGraph::open(&path).unwrap();
        assert_eq!(mapped.num_nodes().unwrap(), graph.num_nodes);
        assert_eq!(mapped.num_edges().unwrap(), graph.num_edges);

        let names = mapped.collection_names().unwrap();
        assert_eq!(names, graph.collection_names);

        let edge_src = mapped.read_u32_tensor("edge_src").unwrap();
        assert_eq!(edge_src, graph.edge_src);

        let features = mapped.read_node_features().unwrap();
        assert_eq!(features.len(), graph.num_nodes * graph.feature_dim);
        // Node 0 should have all 1.0
        assert_eq!(features[0], 1.0);
        // Node 1 should have all 0.0 (no embedding set)
        assert_eq!(features[JINA_DIM], 0.0);
    }

    #[test]
    fn test_split_config_invalid() {
        let bad = SplitConfig {
            val_ratio: 0.6,
            test_ratio: 0.6,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        let err = split_edges(100, &bad).unwrap_err();
        assert!(matches!(err, TensorError::InvalidSplitConfig { .. }));

        let negative = SplitConfig {
            val_ratio: -0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        let err = split_edges(100, &negative).unwrap_err();
        assert!(matches!(err, TensorError::InvalidSplitConfig { .. }));
    }

    #[test]
    fn test_split_config_nan_inf() {
        // NaN val_ratio
        let nan_val = SplitConfig {
            val_ratio: f64::NAN,
            test_ratio: 0.1,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        assert!(matches!(
            split_edges(100, &nan_val).unwrap_err(),
            TensorError::InvalidSplitConfig { .. }
        ));

        // Infinity test_ratio
        let inf_test = SplitConfig {
            val_ratio: 0.1,
            test_ratio: f64::INFINITY,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        assert!(matches!(
            split_edges(100, &inf_test).unwrap_err(),
            TensorError::InvalidSplitConfig { .. }
        ));

        // NaN neg_sampling_ratio — caught in prepare_and_serialize
        let nan_neg = SplitConfig {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: f64::NAN,
            seed: 0,
        };
        let graph = test_graph();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan.safetensors");
        assert!(matches!(
            prepare_and_serialize(&path, &graph, &nan_neg).unwrap_err(),
            TensorError::InvalidNegSamplingRatio { .. }
        ));

        // Negative neg_sampling_ratio
        let neg_neg = SplitConfig {
            val_ratio: 0.1,
            test_ratio: 0.1,
            neg_sampling_ratio: -1.0,
            seed: 0,
        };
        assert!(matches!(
            prepare_and_serialize(&path, &graph, &neg_neg).unwrap_err(),
            TensorError::InvalidNegSamplingRatio { .. }
        ));
    }

    #[test]
    fn test_serialize_inference_roundtrip() {
        let graph = test_graph();
        let bytes = serialize_graph_for_inference(&graph).unwrap();
        assert!(!bytes.is_empty());

        let st = SafeTensors::deserialize(&bytes).unwrap();

        // Core tensors present
        let nf = st.tensor("node_features").unwrap();
        assert_eq!(nf.shape(), &[graph.num_nodes, graph.feature_dim]);

        let es = st.tensor("edge_src").unwrap();
        assert_eq!(es.shape(), &[graph.num_edges]);

        let et = st.tensor("edge_type").unwrap();
        assert_eq!(et.shape(), &[graph.num_edges]);

        // Training tensors absent
        assert!(st.tensor("train_idx").is_err());
        assert!(st.tensor("val_idx").is_err());
        assert!(st.tensor("test_idx").is_err());
        assert!(st.tensor("neg_src").is_err());
        assert!(st.tensor("neg_dst").is_err());

        // Metadata
        let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = meta.metadata().as_ref().unwrap();
        assert_eq!(meta["num_nodes"], graph.num_nodes.to_string());
        assert_eq!(meta["mode"], "inference");
    }

    #[test]
    fn test_serialize_inference_file_roundtrip() {
        let graph = test_graph();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("inference.safetensors");

        serialize_graph_for_inference_to_file(&path, &graph).unwrap();
        assert!(path.exists());

        // Can read back via MappedGraph (core tensors)
        let mapped = MappedGraph::open(&path).unwrap();
        assert_eq!(mapped.num_nodes().unwrap(), graph.num_nodes);
        assert_eq!(mapped.num_edges().unwrap(), graph.num_edges);

        // The training service runs as a separate same-group user and reads
        // this file via LoadGraph, so it must be group-readable. NamedTempFile
        // defaults to 0600 (owner-only); the serializer must widen it. Without
        // this, cross-user `graph-embed update` fails at LoadGraph.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(
                mode & 0o040,
                0o040,
                "inference file not group-readable: {mode:o}"
            );
        }
    }

    #[test]
    fn test_split_config_custom() {
        let config = SplitConfig {
            val_ratio: 0.2,
            test_ratio: 0.2,
            neg_sampling_ratio: 2.0,
            seed: 0,
        };
        let split = split_edges(100, &config).unwrap();
        assert_eq!(split.val_idx.len(), 20);
        assert_eq!(split.test_idx.len(), 20);
        assert_eq!(split.train_idx.len(), 60);
    }
}
