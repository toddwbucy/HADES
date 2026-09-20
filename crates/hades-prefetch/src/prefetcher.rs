//! Async double-buffered prefetcher for RGCN training.
//!
//! Pre-computes negative samples for upcoming epochs while the GPU
//! trains on the current epoch.  Uses a bounded tokio channel for
//! backpressure — if the GPU is slower than sampling, the producer
//! blocks until the channel has room.
//!
//! ## Usage
//!
//! ```ignore
//! // One-time setup: load graph → split edges → serialize for Python
//! let data = prepare_training_data(&pool, &path, &split_config).await?;
//!
//! // Start prefetcher — background task pre-computes negatives
//! let mut pf = Prefetcher::start(
//!     data.graph.clone(),
//!     data.split.clone(),
//!     PrefetchConfig::default(),
//!     Some(200), // epochs
//! );
//!
//! while let Some(batch) = pf.next_batch().await {
//!     let batch = batch?;
//!     // send batch.train_neg / batch.val_neg to Python TrainingService
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, instrument};

use hades_core::db::ArangoPool;
use hades_core::graph::loader::GraphLoaderError;
use hades_core::graph::types::{GraphData, IDMap};

use crate::tensor::{
    EdgeSplit, NegativeSamples, SplitConfig, TensorError, negative_sample_cancellable,
    negative_sample_count, prepare_and_serialize,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from prefetcher setup and the training data pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PrefetchError {
    #[error("graph loading failed: {0}")]
    GraphLoad(#[from] GraphLoaderError),

    #[error("runtime schema load failed: {0}")]
    SchemaLoad(String),

    #[error("tensor/serialization error: {0}")]
    Tensor(#[from] TensorError),

    #[error("invalid prefetch_depth: must be >= 1")]
    InvalidDepth,

    #[error("invalid neg_sampling_ratio: {0} (must be finite and positive)")]
    InvalidNegRatio(f64),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the async prefetcher.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of epochs to prefetch ahead (channel buffer size).
    ///
    /// Higher values trade memory for latency tolerance.  Each buffered
    /// epoch holds two `NegativeSamples` (train + val), whose size is
    /// proportional to `num_train_edges × neg_sampling_ratio`.
    ///
    /// Default: 2 (double-buffered).
    pub prefetch_depth: usize,

    /// Ratio of negative to positive samples per epoch.
    /// Default: 1.0 (one negative per positive edge).
    pub neg_sampling_ratio: f64,
    /// Seed for per-epoch train samples and fixed validation samples.
    pub seed: u64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_depth: 2,
            neg_sampling_ratio: 1.0,
            seed: 0,
        }
    }
}

impl PrefetchConfig {
    /// Validate configuration values.
    fn validate(&self) -> Result<(), PrefetchError> {
        if self.prefetch_depth == 0 {
            return Err(PrefetchError::InvalidDepth);
        }
        if !self.neg_sampling_ratio.is_finite() || self.neg_sampling_ratio <= 0.0 {
            return Err(PrefetchError::InvalidNegRatio(self.neg_sampling_ratio));
        }
        Ok(())
    }

    /// Estimate data memory per buffered epoch in bytes.
    ///
    /// Each `NegativeSamples` holds two `Vec<u32>` of length
    /// `num_edges × neg_sampling_ratio`.  We buffer train + val negatives.
    /// This counts only element bytes, not per-Vec heap headers (~96 bytes
    /// per batch for four `Vec<u32>`) — negligible for real workloads.
    pub fn estimate_batch_bytes(&self, num_train_edges: usize, num_val_edges: usize) -> usize {
        let train_neg = (num_train_edges as f64 * self.neg_sampling_ratio) as usize;
        let val_neg = (num_val_edges as f64 * self.neg_sampling_ratio) as usize;
        // Each NegativeSamples has src + dst Vec<u32> = 2 × len × 4 bytes
        train_neg
            .saturating_add(val_neg)
            .saturating_mul(2 * std::mem::size_of::<u32>())
    }

    /// Estimate total memory for all buffered epochs.
    pub fn estimate_buffer_bytes(&self, num_train_edges: usize, num_val_edges: usize) -> usize {
        self.estimate_batch_bytes(num_train_edges, num_val_edges)
            .saturating_mul(self.prefetch_depth)
    }
}

// ---------------------------------------------------------------------------
// EpochBatch — one epoch's worth of negative samples
// ---------------------------------------------------------------------------

/// Pre-computed negative samples for a single training epoch.
///
/// Each epoch gets fresh random negatives so the model cannot memorize
/// specific negative pairs.
#[derive(Debug, Clone)]
pub struct EpochBatch {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Negative samples for training edges.
    pub train_neg: NegativeSamples,
    /// Negative samples for validation edges.
    pub val_neg: NegativeSamples,
}

// ---------------------------------------------------------------------------
// TrainingData — result of one-time setup
// ---------------------------------------------------------------------------

/// Result of [`prepare_training_data`]: everything needed to start training.
pub struct TrainingData {
    /// The loaded graph (shared ownership for the prefetcher).
    pub graph: Arc<GraphData>,
    /// Bidirectional ArangoDB `_id` ↔ index mapping.
    pub id_map: IDMap,
    /// Train/val/test edge split.
    pub split: Arc<EdgeSplit>,
}

// ---------------------------------------------------------------------------
// Prefetcher
// ---------------------------------------------------------------------------

/// Async prefetcher that pre-computes negative samples on a background task.
///
/// A bounded tokio mpsc channel connects the producer (background task
/// sampling negatives) to the consumer (training loop).  The channel
/// capacity equals `prefetch_depth`, providing natural backpressure.
///
/// The background task uses [`tokio::task::spawn_blocking`] for the
/// CPU-bound negative sampling work to avoid starving the async runtime.
pub struct Prefetcher {
    rx: mpsc::Receiver<Result<EpochBatch, PrefetchError>>,
    handle: JoinHandle<()>,
    cancel: CancellationToken,
}

impl Prefetcher {
    /// Start the prefetcher background task.
    ///
    /// * `graph` — shared graph data (only edges are read for rejection sampling).
    /// * `split` — edge split (used to compute negative sample counts).
    /// * `config` — prefetch depth and sampling ratio.
    /// * `num_epochs` — total epochs to produce.  `None` = unbounded
    ///   (stop by dropping the `Prefetcher` or calling [`stop()`](Self::stop)).
    pub fn start(
        graph: Arc<GraphData>,
        split: Arc<EdgeSplit>,
        config: PrefetchConfig,
        num_epochs: Option<usize>,
    ) -> Result<Self, PrefetchError> {
        config.validate()?;

        let num_train_neg =
            negative_sample_count(split.train_idx.len(), config.neg_sampling_ratio)?;
        let num_val_neg = negative_sample_count(split.val_idx.len(), config.neg_sampling_ratio)?;

        if num_train_neg == 0 || num_val_neg == 0 {
            return Err(TensorError::ValidationFailed {
                message: "training and validation require nonempty positive and negative samples"
                    .into(),
            }
            .into());
        }

        let buffer_bytes = config.estimate_buffer_bytes(split.train_idx.len(), split.val_idx.len());

        info!(
            prefetch_depth = config.prefetch_depth,
            num_train_neg,
            num_val_neg,
            buffer_mb = buffer_bytes as f64 / (1024.0 * 1024.0),
            num_epochs = ?num_epochs,
            "prefetcher starting"
        );

        let (tx, rx) = mpsc::channel(config.prefetch_depth);

        let cancel = CancellationToken::new();
        let handle = tokio::spawn(Self::producer(
            tx,
            graph,
            num_train_neg,
            num_val_neg,
            num_epochs,
            config.seed,
            cancel.clone(),
        ));

        Ok(Self { rx, handle, cancel })
    }

    /// Background producer task.
    async fn producer(
        tx: mpsc::Sender<Result<EpochBatch, PrefetchError>>,
        graph: Arc<GraphData>,
        num_train_neg: usize,
        num_val_neg: usize,
        num_epochs: Option<usize>,
        seed: u64,
        cancel: CancellationToken,
    ) {
        let mut epoch = 0;
        loop {
            if cancel.is_cancelled() {
                break;
            }
            if let Some(max) = num_epochs
                && epoch >= max
            {
                break;
            }

            // Negative sampling is CPU-bound — run both train and val
            // concurrently on the blocking thread pool.
            let g1 = Arc::clone(&graph);
            let g2 = Arc::clone(&graph);

            let train_cancel = cancel.clone();
            let val_cancel = cancel.clone();
            let train = tokio::task::spawn_blocking(move || {
                negative_sample_cancellable(
                    &g1,
                    num_train_neg,
                    seed.wrapping_add(epoch as u64).wrapping_add(10),
                    || train_cancel.is_cancelled(),
                )
            });
            let val = tokio::task::spawn_blocking(move || {
                negative_sample_cancellable(&g2, num_val_neg, seed.wrapping_add(1), || {
                    val_cancel.is_cancelled()
                })
            });
            let train_abort = train.abort_handle();
            let val_abort = val.abort_handle();
            let samples = async move { tokio::join!(train, val) };
            tokio::pin!(samples);
            let (train_neg, val_neg) = tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Abort queued jobs; running jobs observe the token. Join
                    // both before dropping the producer's ownership of work.
                    train_abort.abort();
                    val_abort.abort();
                    let _ = samples.await;
                    return;
                }
                result = &mut samples => result,
            };
            if cancel.is_cancelled() {
                return;
            }

            let samples = match (train_neg, val_neg) {
                (Ok(Ok(train)), Ok(Ok(val))) => Ok((train, val)),
                (Ok(Err(error)), _) | (_, Ok(Err(error))) => Err(PrefetchError::Tensor(error)),
                (Err(error), _) | (_, Err(error)) => Err(PrefetchError::SchemaLoad(format!(
                    "sampling task failed: {error}"
                ))),
            };
            let (train_neg, val_neg) = match samples {
                Ok(samples) => samples,
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            };

            let batch = EpochBatch {
                epoch,
                train_neg,
                val_neg,
            };

            debug!(epoch, "prefetched batch");

            // Channel send — blocks if buffer is full (backpressure).
            // Returns Err if receiver is dropped → stop producing.
            if tx.send(Ok(batch)).await.is_err() {
                debug!("prefetcher channel closed, stopping");
                break;
            }

            epoch += 1;
        }

        info!(epochs_produced = epoch, "prefetcher finished");
    }

    /// Receive the next pre-computed batch.
    ///
    /// Returns `None` when all epochs have been produced and consumed,
    /// or when the prefetcher has been stopped.
    pub async fn next_batch(&mut self) -> Option<Result<EpochBatch, PrefetchError>> {
        self.rx.recv().await
    }

    /// Request cancellation and discard buffered batches without waiting.
    /// Running samplers cooperate at checkpoints; use `shutdown` to join them.
    pub fn stop(self) {
        self.cancel.cancel();
    }

    /// Cancel queued/running sampling and wait for all owned sampling jobs.
    /// Allocator calls are not preemptible; this has no wall-clock guarantee.
    pub async fn shutdown(mut self) -> Result<(), tokio::task::JoinError> {
        self.cancel.cancel();
        self.rx.close();
        (&mut self.handle).await
    }

    /// Check whether the background task is still running.
    pub fn is_running(&self) -> bool {
        !self.handle.is_finished()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.rx.close();
    }
}

// ---------------------------------------------------------------------------
// One-time training data setup
// ---------------------------------------------------------------------------

/// Load graph from ArangoDB, split edges, and serialize to safetensors.
///
/// This is the one-time setup before training begins:
/// 1. Load the full graph (nodes + edges + embeddings) from ArangoDB.
/// 2. Split edges into train/val/test sets.
/// 3. Generate initial negative samples.
/// 4. Serialize everything to a safetensors file for Python to mmap.
///
/// Returns a [`TrainingData`] containing the graph and split (wrapped in
/// `Arc` for sharing with the prefetcher).
#[instrument(skip_all, fields(path = %output_path.display()))]
pub async fn prepare_training_data(
    pool: &ArangoPool,
    output_path: &Path,
    config: &SplitConfig,
) -> Result<TrainingData, PrefetchError> {
    info!("loading graph from ArangoDB");
    let schema = hades_core::graph::RuntimeSchema::load(pool)
        .await
        .map_err(|e| PrefetchError::SchemaLoad(e.to_string()))?;
    info!(
        from_db = schema.from_database,
        num_relations = schema.meta.num_relations,
        feature_dim = schema.meta.feature_dim,
        "loaded runtime schema"
    );
    let (graph, id_map) = hades_core::graph::load(pool, &schema).await?;

    info!(
        num_nodes = graph.num_nodes,
        num_edges = graph.num_edges,
        embedded = graph.embedded_count(),
        "graph loaded, preparing training data"
    );

    // Wrap in Arc now — avoids a full deep clone into spawn_blocking
    let graph = Arc::new(graph);

    // split + negative sample + serialize — CPU-bound
    let graph_ref = Arc::clone(&graph);
    let config_clone = config.clone();
    let path = output_path.to_path_buf();

    let split = tokio::task::spawn_blocking(move || -> Result<EdgeSplit, TensorError> {
        prepare_and_serialize(&path, &graph_ref, &config_clone)
    })
    .await
    .expect("serialization task panicked")?;

    info!(
        train = split.train_idx.len(),
        val = split.val_idx.len(),
        test = split.test_idx.len(),
        path = %output_path.display(),
        "training data prepared"
    );

    Ok(TrainingData {
        graph,
        id_map,
        split: Arc::new(split),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::split_edges;

    /// Jina V4 feature dimension — local to keep tests independent of the
    /// deleted `graph::schema` module.
    const JINA_DIM: usize = 2048;

    /// Build a small test graph matching tensor.rs tests.
    fn test_graph() -> GraphData {
        let num_nodes = 10;
        let mut graph = GraphData::with_capacity(num_nodes, 0);

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

        graph.collection_names = vec!["col_a".into(), "col_b".into()];
        for i in 0..num_nodes {
            graph.node_collections[i] = (i % 2) as u32;
        }

        let emb = vec![1.0f32; JINA_DIM];
        graph.set_node_features(0, &emb);
        graph.set_node_features(3, &emb);
        graph.set_node_features(7, &emb);

        graph
    }

    #[test]
    fn shutdown_joins_queued_samples_and_releases_graph() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let gate = tokio::task::spawn_blocking(move || {
                entered_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(std::time::Duration::from_secs(5));
            });
            entered_rx.await.unwrap();
            let graph = Arc::new(test_graph());
            let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
            let pf =
                Prefetcher::start(graph.clone(), split, PrefetchConfig::default(), None).unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while Arc::strong_count(&graph) != 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let shutdown = tokio::spawn(pf.shutdown());
            tokio::task::yield_now().await;
            release_tx.send(()).unwrap();
            gate.await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), shutdown)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(Arc::strong_count(&graph), 1);
        });
    }

    #[tokio::test]
    async fn shutdown_unblocks_full_buffer_and_drop_requests_cancellation() {
        for explicit in [true, false] {
            let graph = Arc::new(test_graph());
            let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
            let pf =
                Prefetcher::start(graph.clone(), split, PrefetchConfig::default(), None).unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while pf.rx.len() < 2 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            if explicit {
                tokio::time::timeout(std::time::Duration::from_secs(2), pf.shutdown())
                    .await
                    .unwrap()
                    .unwrap();
            } else {
                drop(pf);
            }
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while Arc::strong_count(&graph) != 1 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
    }

    #[test]
    fn test_prefetch_config_defaults() {
        let config = PrefetchConfig::default();
        assert_eq!(config.prefetch_depth, 2);
        assert!((config.neg_sampling_ratio - 1.0).abs() < f64::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_prefetch_config_validate_depth() {
        let bad = PrefetchConfig {
            prefetch_depth: 0,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        assert!(matches!(bad.validate(), Err(PrefetchError::InvalidDepth)));
    }

    #[test]
    fn test_prefetch_config_validate_neg_ratio() {
        for bad_ratio in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let config = PrefetchConfig {
                prefetch_depth: 2,
                neg_sampling_ratio: bad_ratio,
                seed: 0,
            };
            assert!(
                matches!(config.validate(), Err(PrefetchError::InvalidNegRatio(_))),
                "expected InvalidNegRatio for {bad_ratio}"
            );
        }
    }

    #[test]
    fn test_estimate_batch_bytes() {
        let config = PrefetchConfig {
            prefetch_depth: 2,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        // 100 train edges × 1.0 ratio = 100 neg, 10 val edges × 1.0 = 10 neg
        // Each neg: 2 vecs × len × 4 bytes = (100 + 10) × 2 × 4 = 880
        let bytes = config.estimate_batch_bytes(100, 10);
        assert_eq!(bytes, 880);
    }

    #[test]
    fn test_estimate_buffer_bytes() {
        let config = PrefetchConfig {
            prefetch_depth: 3,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };
        let batch = config.estimate_batch_bytes(100, 10);
        let buffer = config.estimate_buffer_bytes(100, 10);
        assert_eq!(buffer, batch * 3);
    }

    #[tokio::test]
    async fn dense_sampling_error_reaches_consumer() {
        let mut graph = GraphData::with_capacity(3, 0);
        for src in 0..3 {
            for dst in 0..3 {
                if src != dst {
                    graph.add_edge(src, dst, 0);
                }
            }
        }
        let split = Arc::new(EdgeSplit {
            train_idx: vec![0, 1],
            val_idx: vec![2, 3],
            test_idx: vec![4, 5],
        });
        let mut pf =
            Prefetcher::start(Arc::new(graph), split, PrefetchConfig::default(), Some(1)).unwrap();
        assert!(pf.next_batch().await.unwrap().is_err());
        assert!(pf.next_batch().await.is_none());
    }

    #[tokio::test]
    async fn test_prefetcher_produces_batches() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig::default();

        let mut pf = Prefetcher::start(graph.clone(), split.clone(), config, Some(5)).unwrap();

        let mut received = Vec::new();
        while let Some(batch) = pf.next_batch().await {
            let batch = batch.unwrap();
            received.push(batch);
        }

        assert_eq!(received.len(), 5, "should receive exactly 5 batches");

        // Epoch numbers are sequential
        for (i, batch) in received.iter().enumerate() {
            assert_eq!(batch.epoch, i);
        }

        // Each batch has non-empty negatives
        for batch in &received {
            assert!(!batch.train_neg.src.is_empty());
            assert_eq!(batch.train_neg.src.len(), batch.train_neg.dst.len());
            assert!(!batch.val_neg.src.is_empty());
            assert_eq!(batch.val_neg.src.len(), batch.val_neg.dst.len());
        }
    }

    #[tokio::test]
    async fn test_prefetcher_different_negatives_per_epoch() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig::default();

        let mut pf = Prefetcher::start(graph.clone(), split.clone(), config, Some(3)).unwrap();

        let b0 = pf.next_batch().await.unwrap().unwrap();
        let b1 = pf.next_batch().await.unwrap().unwrap();
        let b2 = pf.next_batch().await.unwrap().unwrap();

        assert_eq!(b0.val_neg.src, b1.val_neg.src);
        assert_eq!(b0.val_neg.dst, b1.val_neg.dst);
        assert_eq!(b1.val_neg.src, b2.val_neg.src);
        assert_eq!(b1.val_neg.dst, b2.val_neg.dst);

        // It's astronomically unlikely that two random samples are identical
        // on a 10-node graph with 20 edges. Check at least one pair differs.
        let all_same = b0.train_neg.src == b1.train_neg.src
            && b1.train_neg.src == b2.train_neg.src
            && b0.train_neg.dst == b1.train_neg.dst
            && b1.train_neg.dst == b2.train_neg.dst;

        assert!(
            !all_same,
            "negative samples should differ across epochs (re-randomized)"
        );
    }

    #[tokio::test]
    async fn test_prefetcher_stop_early() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig {
            prefetch_depth: 1,
            neg_sampling_ratio: 1.0,
            seed: 0,
        };

        // Request unbounded but stop after 2
        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            config,
            None, // unbounded
        )
        .unwrap();

        let _b0 = pf.next_batch().await.unwrap().unwrap();
        let _b1 = pf.next_batch().await.unwrap().unwrap();

        // Dropping should cancel the background task
        pf.stop();
    }

    #[tokio::test]
    async fn test_prefetcher_drop_cancels() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());

        let pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            PrefetchConfig::default(),
            None,
        )
        .unwrap();

        assert!(pf.is_running());

        // Drop should abort the background task
        drop(pf);

        // Give the runtime a moment to process the abort
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn test_prefetcher_custom_ratio() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());
        let config = PrefetchConfig {
            prefetch_depth: 1,
            neg_sampling_ratio: 2.0,
            seed: 0,
        };

        let expected_train_neg = (split.train_idx.len() as f64 * 2.0) as usize;

        let mut pf = Prefetcher::start(graph.clone(), split.clone(), config, Some(1)).unwrap();

        let batch = pf.next_batch().await.unwrap().unwrap();
        assert_eq!(batch.train_neg.src.len(), expected_train_neg);
    }

    #[tokio::test]
    async fn test_prefetcher_negatives_in_bounds() {
        let graph = Arc::new(test_graph());
        let split = Arc::new(split_edges(graph.num_edges, &SplitConfig::default()).unwrap());

        let mut pf = Prefetcher::start(
            graph.clone(),
            split.clone(),
            PrefetchConfig::default(),
            Some(3),
        )
        .unwrap();

        let num_nodes = graph.num_nodes as u32;

        while let Some(batch) = pf.next_batch().await {
            let batch = batch.unwrap();
            for (&s, &d) in batch.train_neg.src.iter().zip(&batch.train_neg.dst) {
                assert!(s < num_nodes, "train neg src {s} out of bounds");
                assert!(d < num_nodes, "train neg dst {d} out of bounds");
                assert_ne!(s, d, "train neg should not have self-loops");
            }
            for (&s, &d) in batch.val_neg.src.iter().zip(&batch.val_neg.dst) {
                assert!(s < num_nodes, "val neg src {s} out of bounds");
                assert!(d < num_nodes, "val neg dst {d} out of bounds");
                assert_ne!(s, d, "val neg should not have self-loops");
            }
        }
    }

    #[test]
    fn test_epoch_batch_clone() {
        let batch = EpochBatch {
            epoch: 0,
            train_neg: NegativeSamples {
                src: vec![0, 1],
                dst: vec![2, 3],
            },
            val_neg: NegativeSamples {
                src: vec![4],
                dst: vec![5],
            },
        };
        let cloned = batch.clone();
        assert_eq!(cloned.epoch, 0);
        assert_eq!(cloned.train_neg.src, vec![0, 1]);
    }

    #[test]
    fn test_training_data_fields() {
        let graph = test_graph();
        let split = split_edges(graph.num_edges, &SplitConfig::default()).unwrap();

        let data = TrainingData {
            graph: Arc::new(graph),
            id_map: IDMap::new(),
            split: Arc::new(split),
        };

        assert_eq!(data.graph.num_nodes, 10);
        assert_eq!(data.graph.num_edges, 20);
        assert!(data.id_map.is_empty());
        assert_eq!(
            data.split.train_idx.len() + data.split.val_idx.len() + data.split.test_idx.len(),
            20
        );
    }
}
