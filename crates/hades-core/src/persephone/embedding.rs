//! Persephone Embedding Client — OpenAI-compatible HTTP client for vector
//! embedding generation.
//!
//! HADES is **engine-agnostic** at the protocol layer: any embedding engine
//! that exposes the OpenAI `/v1/embeddings` surface (vLLM, HuggingFace TEI,
//! `hades-weaver-bridge`, llama.cpp-server, ollama where capable, the
//! upstream OpenAI/Anthropic APIs, etc.) is a valid backend. Engines speak
//! the same wire shape; HADES doesn't care which one is running.
//!
//! HADES is **model-bound** at the data layer to Jina V4 (or a future model
//! with the same capability profile: 2048d, 32k context, multimodal,
//! late-chunking-capable). Wrong model → invalidated stored vectors. The
//! engine is fungible, the model is not.
//!
//! Wire protocol: plain HTTP/1.1 JSON, OpenAI-compatible shape.
//!   - `GET  {base}/models`     → list of available models (used for [`info`])
//!   - `POST {base}/embeddings` → embedding generation (used for [`embed`])
//!
//! `task` and `batch_size` are sent as non-standard top-level fields. Engines
//! that don't recognize them ignore them (per JSON convention). Engines that
//! do (vLLM-serving-Jina, etc.) use them for retrieval-quality hints.
//!
//! Endpoint can be either an HTTP base URL (`http://localhost:8000/v1`) or a
//! Unix socket path (`/run/.../embedder.sock`). The Unix socket path is
//! intended for the `hades-weaver-bridge` adapter that exposes Weaver's gRPC
//! embedder via a local OpenAI-compatible HTTP surface.

use std::path::PathBuf;
use std::time::Duration;

use http::header::CONTENT_TYPE;
use http::{Method, Request, Uri};
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use hyperlocal::{UnixClientExt, UnixConnector};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// Default timeout for embedding requests (5 min for large batches).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the embedding client.
#[derive(Debug, Clone)]
pub struct EmbeddingClientConfig {
    /// Endpoint for the OpenAI-compatible embedding service.
    pub endpoint: EmbeddingEndpoint,
    /// Model identifier sent in every request (`model` field of the OpenAI
    /// embeddings request body). HADES is bound to Jina V4 capabilities;
    /// configure this to match whatever model your engine has loaded. Responses
    /// must name this model; the reference Jina registry ID also accepts an
    /// absolute local directory ending in `jinaai--jina-embeddings-v4`.
    pub model: String,
    /// Required vector width (default 2048). Never infer this from an untrusted
    /// response. Set explicitly when configuring another model or vector width.
    pub expected_dimension: u32,
    /// Request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the embedding service.
///
/// Both variants speak HTTP/1.1 JSON in the OpenAI-compatible shape; the
/// only difference is the transport. Unix is intended for the
/// `hades-weaver-bridge` adapter (Weaver coexistence mode); HTTP is the
/// default for everything else.
#[derive(Debug, Clone)]
pub enum EmbeddingEndpoint {
    /// Unix domain socket path. The server listening on this socket must
    /// expose `/v1/embeddings` and `/v1/models`.
    Unix(PathBuf),
    /// HTTP base URL including the `/v1` prefix
    /// (e.g. `http://localhost:8000/v1`). The client appends `/embeddings`
    /// and `/models` to form the full request URI.
    Tcp(String),
}

/// Default model identifier. Jina V4 is HADES's reference model; future
/// capability-equivalent models can be substituted by setting this.
const DEFAULT_MODEL: &str = "jinaai/jina-embeddings-v4";
/// Default endpoint: HADES-owned embedder on local URL. Port 8087 avoids
/// collisions with vLLM/uvicorn (8000) and weaver-serve LLM API (8080).
/// Override via config or `HADES_EMBEDDER_SOCKET` env var.
const DEFAULT_ENDPOINT_URL: &str = "http://localhost:8087/v1";

impl Default for EmbeddingClientConfig {
    fn default() -> Self {
        Self {
            endpoint: EmbeddingEndpoint::Tcp(DEFAULT_ENDPOINT_URL.to_string()),
            model: DEFAULT_MODEL.to_string(),
            expected_dimension: 2048,
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Error type for embedding client operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Transport/connection error.
    #[error("connection error: {0}")]
    Connection(String),

    /// HTTP error from the service.
    #[error("service error (HTTP {status}): {message}")]
    Http { status: u16, message: String },

    /// Invalid or unparseable response.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// Request timed out.
    #[error("request timed out after {0}s")]
    Timeout(u64),

    /// I/O error (socket not found, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl EmbeddingError {
    /// Whether this looks like a transient GPU out-of-memory condition that
    /// might succeed if retried with a smaller batch.
    ///
    /// Heuristic: an HTTP 500 whose body contains an OOM-shaped substring
    /// from the most common engines (PyTorch's "CUDA out of memory",
    /// generic "out of memory", or "OutOfMemoryError"). Conservative —
    /// false negatives just mean we don't halve.
    fn is_retriable_oom(&self) -> bool {
        match self {
            Self::Http {
                status: 500,
                message,
            } => {
                let m = message.to_ascii_lowercase();
                m.contains("out of memory")
                    || m.contains("outofmemoryerror")
                    || m.contains("cuda oom")
            }
            _ => false,
        }
    }
}

impl From<hyper_util::client::legacy::Error> for EmbeddingError {
    fn from(e: hyper_util::client::legacy::Error) -> Self {
        EmbeddingError::Connection(e.to_string())
    }
}

impl From<http::Error> for EmbeddingError {
    fn from(e: http::Error) -> Self {
        EmbeddingError::Connection(e.to_string())
    }
}

impl From<serde_json::Error> for EmbeddingError {
    fn from(e: serde_json::Error) -> Self {
        EmbeddingError::InvalidResponse(e.to_string())
    }
}

/// Provider info derived from the OpenAI `/v1/models` endpoint.
///
/// `device` and `dimension` are NOT part of the OpenAI standard; they're
/// engine-specific and may be `None` depending on the backend. `model_loaded`
/// is true iff the configured model appears in the engine's `/v1/models`
/// listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Configured model name (echoed from `EmbeddingClientConfig.model`).
    pub model_name: String,
    /// Device the model is loaded on (e.g. "cuda:0"). `None` for engines
    /// that don't expose this via a standard endpoint.
    #[serde(default)]
    pub device: Option<String>,
    /// Whether the configured model appears in the engine's model list.
    pub model_loaded: bool,
    /// Embedding dimension. `None` unless the engine advertises it (most
    /// don't via `/v1/models`); HADES expects 2048 for Jina V4 regardless.
    #[serde(default)]
    pub dimension: Option<u32>,
    /// Longest input this backend will accept, in tokens.
    ///
    /// A property of the load profile the backend is running, not of the model:
    /// the same weights serve 11,900 on a 16 GiB card and 32,768 on a 48 GiB
    /// one. A caller that packs windows has to read this rather than carry a
    /// constant, because a constant is wrong on one of the two cards and was
    /// wrong on both: the hardcoded 12,000-character budget it replaced packed
    /// at roughly 2,870 tokens and split 134 of 285 files in one real corpus.
    #[serde(default)]
    pub max_seq_length: Option<u32>,
    /// The backend's load profile name, when it reports one.
    #[serde(default)]
    pub profile: Option<String>,
}

/// Client for the embedding service.
///
/// Speaks HTTP/1.1 JSON over Unix domain socket or TCP.
#[derive(Clone)]
pub struct EmbeddingClient {
    response_limit: Option<usize>,
    config: EmbeddingClientConfig,
    unix_client: Option<Client<UnixConnector, Full<Bytes>>>,
    tcp_client: Option<Client<hyper_util::client::legacy::connect::HttpConnector, Full<Bytes>>>,
}

impl EmbeddingClient {
    /// Bound each HTTP response before deserializing it (for single-query search).
    pub fn with_response_limit(mut self, bytes: usize) -> Result<Self, EmbeddingError> {
        if bytes == 0 {
            return Err(EmbeddingError::InvalidResponse(
                "response limit must be positive".into(),
            ));
        }
        self.response_limit = Some(bytes);
        Ok(self)
    }

    /// Connect to the embedding service.
    ///
    /// For Unix sockets, validates the socket file exists. For TCP,
    /// validates the URL parses. The actual HTTP connection is made
    /// on the first request.
    #[instrument(skip_all)]
    pub async fn connect(config: EmbeddingClientConfig) -> Result<Self, EmbeddingError> {
        if config.model.trim().is_empty() || config.expected_dimension == 0 {
            return Err(EmbeddingError::Connection(
                "model must be nonempty and expected_dimension must be positive".into(),
            ));
        }
        match &config.endpoint {
            EmbeddingEndpoint::Unix(path) => {
                if !path.exists() {
                    return Err(EmbeddingError::Connection(format!(
                        "socket not found: {}",
                        path.display()
                    )));
                }
                debug!(socket = %path.display(), "embedding client targeting UDS");
                let client = Client::unix();
                info!("embedding client ready (Unix socket)");
                Ok(Self {
                    response_limit: None,
                    config,
                    unix_client: Some(client),
                    tcp_client: None,
                })
            }
            EmbeddingEndpoint::Tcp(addr) => {
                debug!(addr, "embedding client targeting TCP");
                let connector = hyper_util::client::legacy::connect::HttpConnector::new();
                let client = Client::builder(TokioExecutor::new())
                    .pool_idle_timeout(Duration::from_secs(90))
                    .build(connector);
                info!("embedding client ready (TCP)");
                Ok(Self {
                    response_limit: None,
                    config,
                    unix_client: None,
                    tcp_client: Some(client),
                })
            }
        }
    }

    /// Connect to the embedding service with default configuration.
    pub async fn connect_default() -> Result<Self, EmbeddingError> {
        Self::connect(EmbeddingClientConfig::default()).await
    }

    /// Connect to an embedding service at the given Unix socket path.
    pub async fn connect_unix_at(path: impl Into<PathBuf>) -> Result<Self, EmbeddingError> {
        let config = EmbeddingClientConfig {
            endpoint: EmbeddingEndpoint::Unix(path.into()),
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Connect to an embedding service at the given endpoint string,
    /// auto-detecting the transport from the prefix.
    ///
    /// - `http://...` or `https://...` → HTTP/TCP endpoint (the base URL,
    ///   typically including `/v1`)
    /// - `unix:///path/to/socket`      → Unix domain socket
    /// - `/path/to/socket`             → Unix domain socket (bare absolute path)
    ///
    /// Anything else is rejected with [`EmbeddingError::Connection`].
    pub async fn connect_at(endpoint_str: &str) -> Result<Self, EmbeddingError> {
        let endpoint = parse_endpoint(endpoint_str)?;
        let config = EmbeddingClientConfig {
            endpoint,
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Embed a batch of texts into vectors.
    ///
    /// When `batch_size` is set, the texts are split into chunks of that size
    /// and sent as multiple HTTP requests. This client-side chunking keeps
    /// each request small enough to fit within the embedder's VRAM headroom:
    /// the server's `batch_size` hint isn't always honored as a chunking
    /// boundary, so the client owns that responsibility. When `batch_size`
    /// is unset or zero, the entire input is sent in a single request.
    ///
    /// Returns one embedding vector per input text, in input order.
    #[instrument(skip(self, texts), fields(count = texts.len()))]
    pub async fn embed(
        &self,
        texts: &[String],
        task: &str,
        batch_size: Option<u32>,
    ) -> Result<EmbedResult, EmbeddingError> {
        if texts.is_empty() {
            return Ok(EmbedResult {
                embeddings: Vec::new(),
                model: self.config.model.clone(),
                dimension: 0,
                duration_ms: 0,
            });
        }

        let per_request = batch_size
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .unwrap_or(texts.len());

        // Work queue: (start_index_into_texts, slice). Halving on OOM splits
        // a slice into two and pushes both back onto the front of the queue.
        // The BTreeMap keyed by start index reassembles results in input order
        // regardless of the order in which sub-batches complete.
        use std::collections::{BTreeMap, VecDeque};
        let mut work: VecDeque<(usize, &[String])> = VecDeque::new();
        for (i, batch) in texts.chunks(per_request).enumerate() {
            work.push_back((i * per_request, batch));
        }

        let mut completed: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
        let mut model: Option<String> = None;
        let mut dimension = 0u32;
        let mut total_duration_ms = 0u64;

        while let Some((start, batch)) = work.pop_front() {
            match self.embed_single_request(batch, task, batch_size).await {
                Ok(result) => {
                    if model.as_ref().is_some_and(|name| name != &result.model) {
                        return Err(EmbeddingError::InvalidResponse(
                            "model identity changed across embedding batches".into(),
                        ));
                    }
                    model = Some(result.model);
                    if dimension == 0 {
                        dimension = result.dimension;
                    } else if result.dimension != dimension {
                        return Err(EmbeddingError::InvalidResponse(format!(
                            "dimension mismatch across batches: {dimension} vs {}",
                            result.dimension
                        )));
                    }
                    total_duration_ms = total_duration_ms.saturating_add(result.duration_ms);
                    completed.insert(start, result.embeddings);
                }
                Err(e) if e.is_retriable_oom() && batch.len() > 1 => {
                    // GPU OOM with a multi-chunk batch — halve and retry.
                    // Push the right half first so the left half pops first
                    // (front of queue); doesn't affect correctness since the
                    // BTreeMap reassembles by start index, but makes the work
                    // trace easier to read in logs.
                    let mid = batch.len() / 2;
                    debug!(
                        start,
                        batch_size = batch.len(),
                        "embed batch OOM — halving to {} and {}",
                        mid,
                        batch.len() - mid
                    );
                    work.push_front((start + mid, &batch[mid..]));
                    work.push_front((start, &batch[..mid]));
                }
                Err(e) => return Err(e),
            }
        }

        let all_embeddings: Vec<Vec<f32>> = completed.into_values().flatten().collect();

        Ok(EmbedResult {
            embeddings: all_embeddings,
            model: model.unwrap_or_else(|| self.config.model.clone()),
            dimension,
            duration_ms: total_duration_ms,
        })
    }

    /// Embed each input in ONE forward pass, pooling per supplied boundary.
    ///
    /// This is late chunking. Contrast [`Self::embed`], which sends chunks as
    /// separate inputs so each is encoded blind to the rest of its document.
    /// Here the whole text is encoded once and chunk vectors are pooled from
    /// that single pass, so every vector carries the surrounding context.
    ///
    /// `boundaries` are character ranges, one list per input. Pass AST
    /// definition spans or document sections to keep chunks aligned to
    /// meaningful units. The server maps them to token spans and returns the
    /// character ranges back, so symbol intersection still works.
    ///
    /// Pooling happens server-side deliberately: token-level embeddings are
    /// `seq_len x 2048`, roughly 120 MB per long document in float32, which
    /// should not cross the wire.
    ///
    /// Each input must fit the model's context window. Pre-chunking at
    /// section boundaries is the escape hatch for documents that exceed it.
    pub async fn embed_late_chunked(
        &self,
        texts: &[String],
        task: &str,
        boundaries: &[Vec<(usize, usize)>],
    ) -> Result<LateChunkEmbedResult, EmbeddingError> {
        if texts.is_empty() {
            return Ok(LateChunkEmbedResult {
                per_input: Vec::new(),
                model: self.config.model.clone(),
                dimension: 0,
                duration_ms: 0,
            });
        }
        if boundaries.len() != texts.len() {
            return Err(EmbeddingError::InvalidResponse(format!(
                "boundaries has {} entries for {} inputs",
                boundaries.len(),
                texts.len()
            )));
        }

        let mut per_input: Vec<Vec<LateChunkVector>> = vec![Vec::new(); texts.len()];
        let mut dimension = 0u32;
        // Taken from the response, not from the local config, because
        // `model_hash` keys incremental skip and selective re-embedding. The
        // configured name is a request hint and the served model is whatever
        // the backend loaded, so stamping the former puts two hashes on one
        // model as soon as any other path stamps the latter.
        let mut model: Option<String> = None;
        let started = std::time::Instant::now();

        // One request per input. Each yields its own number of chunks, and
        // batching them would only obscure which vectors belong to which text.
        for (i, text) in texts.iter().enumerate() {
            let bounds: Vec<[usize; 2]> = boundaries[i].iter().map(|(s, e)| [*s, *e]).collect();
            if bounds.is_empty() {
                continue;
            }
            let mut body = serde_json::json!({
                "model": self.config.model,
                "input": [text],
                "encoding_format": "float",
                "late_chunk": { "boundaries": bounds },
            });
            if !task.is_empty() {
                body["task"] = serde_json::json!(task);
            }

            let resp = self
                .request(Method::POST, "/embeddings", Some(&body))
                .await?;
            let response_model = validated_response_model(&resp, &self.config.model)?;
            if model.as_ref().is_some_and(|name| name != &response_model) {
                return Err(EmbeddingError::InvalidResponse(
                    "model identity changed across late-chunked inputs".into(),
                ));
            }
            model = Some(response_model);
            let data = resp["data"]
                .as_array()
                .ok_or_else(|| EmbeddingError::InvalidResponse("missing 'data' array".into()))?;

            for item in data {
                // Every field below is required rather than defaulted. A
                // default here is indistinguishable from a correct answer: an
                // engine that ignores the non-standard `late_chunk` field
                // returns plain embeddings with no chunk metadata, and
                // `unwrap_or(0)` would collapse every vector of the input onto
                // chunk 0 and write one embedding where the caller expected N.
                if item["index"].as_u64() != Some(0) {
                    return Err(EmbeddingError::InvalidResponse(
                        "late-chunked response index must be 0 for a single input".into(),
                    ));
                }
                let embedding =
                    validated_vector(&item["embedding"], self.config.expected_dimension)?;
                if dimension == 0 {
                    dimension = embedding.len() as u32;
                } else if embedding.len() as u32 != dimension {
                    // `embed_single_request` checks this and this path did not.
                    // A short vector otherwise passes the count, contiguity and
                    // drift checks and is stored under a dimension it does not
                    // have, which surfaces later as a vector search fault on a
                    // row the corpus reports as healthy.
                    return Err(EmbeddingError::InvalidResponse(format!(
                        "input {i}: inconsistent dimensions, {} then {}",
                        dimension,
                        embedding.len()
                    )));
                }
                let field = |name: &str| -> Result<usize, EmbeddingError> {
                    item[name].as_u64().map(|v| v as usize).ok_or_else(|| {
                        EmbeddingError::InvalidResponse(format!(
                            "late-chunked response item has no '{name}', so the server did \
                             not honour the 'late_chunk' request field and returned plain \
                             embeddings instead of pooled chunk vectors"
                        ))
                    })
                };
                per_input[i].push(LateChunkVector {
                    embedding,
                    chunk_index: field("chunk_index")?,
                    char_start: field("char_start")?,
                    char_end: field("char_end")?,
                });
            }
            // The server may return chunks out of order under concurrency;
            // sort so callers can rely on positional alignment.
            per_input[i].sort_by_key(|c| c.chunk_index);

            // One vector per boundary, or the caller is about to write fewer
            // embeddings than chunks and report success. This is the shape of
            // three separate defects already found in this pipeline, so it is
            // checked rather than assumed.
            if per_input[i].len() != boundaries[i].len() {
                return Err(EmbeddingError::InvalidResponse(format!(
                    "input {i}: sent {} boundaries and received {} vectors",
                    boundaries[i].len(),
                    per_input[i].len()
                )));
            }
            for (k, v) in per_input[i].iter().enumerate() {
                if v.chunk_index != k {
                    return Err(EmbeddingError::InvalidResponse(format!(
                        "input {i}: chunk indices are not contiguous from zero, \
                         expected {k} and found {}",
                        v.chunk_index
                    )));
                }
            }

            // The returned range is token-aligned, so it lands on the token
            // boundary at or before the requested character. A large gap means
            // this vector was labelled with a different chunk's range, which
            // is what happened when the server reconciled its span list
            // against its vector list by length after skipping one in the
            // middle.
            //
            // This does NOT catch a caller sending byte offsets for character
            // offsets. The server has no idea the caller meant bytes: it pools
            // the range it was given and echoes back where it pooled, so
            // requested and returned agree exactly while both point at the
            // wrong text. Verified against the live embedder. Sending
            // characters is the fix for that, and `byte_offsets_to_chars` on
            // the ingest side is what does it.
            const MAX_ALIGNMENT_DRIFT: usize = 64;
            for (k, v) in per_input[i].iter().enumerate() {
                let requested = boundaries[i][k].0;
                // Alignment only ever moves the start EARLIER: the server takes
                // the first token overlapping the range, and that token begins
                // at or before the requested character. A later start cannot
                // come from alignment, so it is not given any tolerance.
                if v.char_start > requested {
                    return Err(EmbeddingError::InvalidResponse(format!(
                        "input {i} chunk {k}: requested character {requested} and the \
                         server pooled from {}, which is later. Token alignment cannot \
                         move a start forward, so this vector carries another chunk's \
                         range",
                        v.char_start
                    )));
                }
                if requested - v.char_start > MAX_ALIGNMENT_DRIFT {
                    return Err(EmbeddingError::InvalidResponse(format!(
                        "input {i} chunk {k}: requested character {requested} and the \
                         server pooled from {}, a drift of {} that token alignment \
                         cannot explain",
                        v.char_start,
                        requested - v.char_start
                    )));
                }
            }
        }

        Ok(LateChunkEmbedResult {
            per_input,
            model: model.unwrap_or_else(|| self.config.model.clone()),
            dimension,
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// Send a single `POST /embeddings` request for the given texts.
    ///
    /// Sends `POST {base}/embeddings` in the OpenAI-compatible shape:
    /// `{"model": ..., "input": [...]}`. The HADES-specific `task` hint
    /// (e.g. `"retrieval.query"`, `"retrieval.passage"` for Jina V4) and
    /// `batch_size` are sent as non-standard top-level fields — engines
    /// that don't understand them ignore them.
    async fn embed_single_request(
        &self,
        texts: &[String],
        task: &str,
        batch_size: Option<u32>,
    ) -> Result<EmbedResult, EmbeddingError> {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        });
        if !task.is_empty() {
            body["task"] = serde_json::json!(task);
        }
        if let Some(bs) = batch_size
            && bs > 0
        {
            body["batch_size"] = serde_json::json!(bs);
        }

        let started = std::time::Instant::now();
        let resp = self
            .request(Method::POST, "/embeddings", Some(&body))
            .await?;
        let duration_ms = started.elapsed().as_millis() as u64;

        // OpenAI shape: { "object": "list", "data": [{"object":"embedding","embedding":[...],"index":N}], "model": "...", "usage": {...} }
        let data = resp["data"].as_array().ok_or_else(|| {
            EmbeddingError::InvalidResponse(
                "missing 'data' array in /v1/embeddings response".into(),
            )
        })?;

        // Sort by `index` to guarantee input-order alignment regardless of
        // server-side reordering (some engines parallelize and may reorder).
        if data.len() != texts.len() {
            return Err(EmbeddingError::InvalidResponse(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                data.len()
            )));
        }
        let mut embeddings = vec![None; texts.len()];
        for item in data {
            let index = item["index"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n < texts.len())
                .ok_or_else(|| {
                    EmbeddingError::InvalidResponse(
                        "embedding index must be an integer in the input range".into(),
                    )
                })?;
            if embeddings[index].is_some() {
                return Err(EmbeddingError::InvalidResponse(format!(
                    "duplicate embedding index {index}"
                )));
            }
            embeddings[index] = Some(validated_vector(
                &item["embedding"],
                self.config.expected_dimension,
            )?);
        }
        // Equal row count and unique in-range indices prove complete coverage.
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(Option::unwrap).collect();
        let dimension = self.config.expected_dimension;
        let model = validated_response_model(&resp, &self.config.model)?;

        debug!(
            count = embeddings.len(),
            dimension,
            duration_ms,
            model = %model,
            "embedding complete"
        );

        Ok(EmbedResult {
            embeddings,
            model,
            dimension,
            duration_ms,
        })
    }

    /// Embed a single text string.
    pub async fn embed_one(&self, text: &str, task: &str) -> Result<Vec<f32>, EmbeddingError> {
        let result = self.embed(&[text.to_string()], task, None).await?;
        Ok(result.embeddings.into_iter().next().unwrap())
    }

    /// Query the embedding provider's model availability via OpenAI's
    /// `/v1/models` endpoint.
    ///
    /// Returns `model_loaded = true` iff the configured model
    /// (`EmbeddingClientConfig.model`) appears in the engine's listing.
    /// `device` and `dimension` are not part of the OpenAI standard and
    /// will be `None` for engines that don't extend the response.
    #[instrument(skip(self))]
    pub async fn info(&self) -> Result<ProviderInfo, EmbeddingError> {
        let resp = self.request(Method::GET, "/models", None).await?;

        let configured_model = &self.config.model;

        // OpenAI shape: { "object": "list", "data": [{"id": "...", ...}, ...] }
        let data = resp["data"].as_array();
        let model_loaded = data
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["id"].as_str())
                    .any(|id| id == configured_model)
            })
            .unwrap_or(false);

        // Some engines (vLLM, llama.cpp-server) attach extra fields we can
        // opportunistically read. Standard says no, but if they're there we
        // surface them.
        // Find the entry describing what is loaded. An exact id match first, and
        // failing that the sole entry of a single-model listing.
        //
        // The fallback is load-bearing rather than lenient. This client is
        // configured with the alias `jinaai/jina-embeddings-v4` while the backend
        // serves a local filesystem path, so the id never matched and every
        // vendor field came back `None`: `max_seq_length` among them, which meant
        // the window budget silently fell back to a conservative constant on a
        // card that could hold three times as much. Finding #14 was the same
        // mismatch reached from the other side, where the configured name got
        // stamped onto stored data instead of the served one. A backend serving
        // one model is describing that model whatever it calls it.
        let entry = data.and_then(|arr| {
            arr.iter()
                .find(|item| item["id"].as_str() == Some(configured_model.as_str()))
                .or_else(|| if arr.len() == 1 { arr.first() } else { None })
        });
        let device = entry.and_then(|item| item["device"].as_str().map(String::from));
        let dimension = entry.and_then(|item| item["dimension"].as_u64().map(|n| n as u32));
        let max_seq_length =
            entry.and_then(|item| item["max_seq_length"].as_u64().map(|n| n as u32));
        let profile = entry
            .and_then(|item| item["profile"].as_str())
            .and_then(|p| {
                if p.is_empty() {
                    None
                } else {
                    Some(p.to_string())
                }
            });

        debug!(
            model = %configured_model,
            loaded = model_loaded,
            "provider info retrieved"
        );

        Ok(ProviderInfo {
            model_name: configured_model.clone(),
            device,
            model_loaded,
            dimension,
            max_seq_length,
            profile,
        })
    }

    /// Check if the service is reachable.
    ///
    /// Uses a short timeout so a stalled service returns `false` quickly
    /// rather than blocking for the full request timeout.
    pub async fn health_check(&self) -> bool {
        match tokio::time::timeout(Duration::from_secs(5), self.info()).await {
            Ok(Ok(_)) => true,
            Ok(Err(e)) => {
                warn!(error = %e, "embedding service health check failed");
                false
            }
            Err(_) => {
                warn!("embedding service health check timed out");
                false
            }
        }
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &EmbeddingEndpoint {
        &self.config.endpoint
    }

    // -----------------------------------------------------------------------
    // HTTP transport
    // -----------------------------------------------------------------------

    /// Send an HTTP request to the embedding service.
    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, EmbeddingError> {
        let uri = self.build_uri(path)?;

        let body_bytes = match body {
            Some(v) => serde_json::to_vec(v)?,
            None => Vec::new(),
        };

        let mut builder = Request::builder().method(method).uri(uri);
        if body.is_some() {
            builder = builder.header(CONTENT_TYPE, "application/json");
        }

        let req = builder.body(Full::new(Bytes::copy_from_slice(&body_bytes)))?;

        let timeout = self.config.timeout;
        let response_future = if let Some(ref client) = self.unix_client {
            client.request(req)
        } else if let Some(ref client) = self.tcp_client {
            client.request(req)
        } else {
            return Err(EmbeddingError::Connection(
                "no transport configured".to_string(),
            ));
        };

        let deadline = tokio::time::Instant::now() + timeout;
        let response = tokio::time::timeout_at(deadline, response_future)
            .await
            .map_err(|_| EmbeddingError::Timeout(timeout.as_secs()))??;

        let status = response.status();
        let body = Limited::new(
            response.into_body(),
            self.response_limit.unwrap_or(usize::MAX),
        );
        let resp_bytes = tokio::time::timeout_at(deadline, body.collect())
            .await
            .map_err(|_| EmbeddingError::Timeout(timeout.as_secs()))?
            .map_err(|e| EmbeddingError::InvalidResponse(format!("response body rejected: {e}")))?
            .to_bytes();

        if !status.is_success() {
            let message = String::from_utf8_lossy(&resp_bytes).into_owned();
            return Err(EmbeddingError::Http {
                status: status.as_u16(),
                message,
            });
        }

        serde_json::from_slice(&resp_bytes).map_err(|e| {
            EmbeddingError::InvalidResponse(format!("failed to parse response JSON: {e}"))
        })
    }

    /// Build a URI for the given path, using Unix socket or TCP.
    ///
    /// For Unix endpoints, the path is appended directly (e.g.
    /// `/v1/embeddings`). For HTTP endpoints, the path is appended to the
    /// configured base URL (e.g. `http://localhost:8000/v1` + `/embeddings`).
    fn build_uri(&self, path: &str) -> Result<Uri, EmbeddingError> {
        match &self.config.endpoint {
            EmbeddingEndpoint::Unix(socket) => {
                // Bridge servers on the Unix socket expose the full /v1/...
                // path; we add the /v1 prefix here so `path` argument stays
                // bare (`/embeddings`, `/models`).
                let full = format!("/v1{}", path);
                Ok(hyperlocal::Uri::new(socket, &full).into())
            }
            EmbeddingEndpoint::Tcp(base) => {
                // HTTP endpoints already include /v1 in the configured base.
                let url = format!("{}{}", base.trim_end_matches('/'), path);
                url.parse()
                    .map_err(|e| EmbeddingError::Connection(format!("invalid URI: {e}")))
            }
        }
    }
}

/// Validate the wire vector before any values are associated with a chunk (#16).
fn validated_vector(value: &serde_json::Value, expected: u32) -> Result<Vec<f32>, EmbeddingError> {
    let values = value
        .as_array()
        .ok_or_else(|| EmbeddingError::InvalidResponse("embedding must be an array".into()))?;
    if values.is_empty() || values.len() != expected as usize {
        return Err(EmbeddingError::InvalidResponse(format!(
            "embedding dimension {}, expected {expected}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|n| n as f32)
                .filter(|n| n.is_finite())
                .ok_or_else(|| {
                    EmbeddingError::InvalidResponse(
                        "embedding values must be finite float32 numbers".into(),
                    )
                })
        })
        .collect()
}

fn validated_response_model(
    resp: &serde_json::Value,
    expected: &str,
) -> Result<String, EmbeddingError> {
    let served = resp["model"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            EmbeddingError::InvalidResponse("embedding response must identify its model".into())
        })?;
    // The reference backend serves a local model directory while requests name
    // the registry ID. Accept that documented alias, not an arbitrary sole model
    // from /models. A different vector space must never silently pass as Jina.
    let reference_id = |name: &str| {
        name == DEFAULT_MODEL
            || (std::path::Path::new(name).is_absolute()
                && std::path::Path::new(name)
                    .file_name()
                    .and_then(|s| s.to_str())
                    == Some("jinaai--jina-embeddings-v4"))
    };
    if served != expected && !(reference_id(served) && reference_id(expected)) {
        return Err(EmbeddingError::InvalidResponse(format!(
            "response model '{served}' is incompatible with requested model '{expected}'"
        )));
    }
    Ok(served.to_string())
}

/// Parse an endpoint string into an [`EmbeddingEndpoint`].
///
/// Recognized prefixes:
/// - `http://`, `https://`           → HTTP/TCP base URL (must include `/v1`)
/// - `unix:///path/to/socket`        → Unix domain socket
/// - `/path/to/socket` (bare path)   → Unix domain socket
fn parse_endpoint(endpoint_str: &str) -> Result<EmbeddingEndpoint, EmbeddingError> {
    if endpoint_str.starts_with("http://") || endpoint_str.starts_with("https://") {
        Ok(EmbeddingEndpoint::Tcp(endpoint_str.to_string()))
    } else if let Some(path) = endpoint_str.strip_prefix("unix://") {
        Ok(EmbeddingEndpoint::Unix(PathBuf::from(path)))
    } else if endpoint_str.starts_with('/') {
        Ok(EmbeddingEndpoint::Unix(PathBuf::from(endpoint_str)))
    } else {
        Err(EmbeddingError::Connection(format!(
            "endpoint must start with http://, https://, unix://, or be an absolute path; got '{endpoint_str}'"
        )))
    }
}

/// One late-chunked vector and where it came from in the source text.
#[derive(Debug, Clone)]
pub struct LateChunkVector {
    /// The pooled, L2-normalized embedding.
    pub embedding: Vec<f32>,
    /// Position of this chunk within its input.
    pub chunk_index: usize,
    /// Character range over the original input.
    ///
    /// Character ranges rather than token indices, because callers intersect
    /// chunks with symbol spans and token positions cannot express that.
    pub char_start: usize,
    pub char_end: usize,
}

/// Result of a late-chunked embed: per input, the chunks it produced.
#[derive(Debug, Clone)]
pub struct LateChunkEmbedResult {
    /// One entry per input text, in input order.
    pub per_input: Vec<Vec<LateChunkVector>>,
    pub model: String,
    pub dimension: u32,
    pub duration_ms: u64,
}

/// Result of an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Embedding vectors, one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Model identifier used.
    pub model: String,
    /// Embedding dimension.
    pub dimension: u32,
    /// Wall-clock time in milliseconds.
    pub duration_ms: u64,
}

impl std::fmt::Debug for EmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingClient")
            .field("endpoint", &self.config.endpoint)
            .field("model", &self.config.model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_endpoint_http() {
        let ep = parse_endpoint("http://localhost:8000/v1").unwrap();
        assert!(matches!(ep, EmbeddingEndpoint::Tcp(ref u) if u == "http://localhost:8000/v1"));
    }

    #[test]
    fn parse_endpoint_https() {
        let ep = parse_endpoint("https://api.openai.com/v1").unwrap();
        assert!(matches!(ep, EmbeddingEndpoint::Tcp(ref u) if u == "https://api.openai.com/v1"));
    }

    #[test]
    fn parse_endpoint_unix_scheme() {
        let ep = parse_endpoint("unix:///run/foo/bar.sock").unwrap();
        match ep {
            EmbeddingEndpoint::Unix(p) => assert_eq!(p, PathBuf::from("/run/foo/bar.sock")),
            _ => panic!("expected Unix variant"),
        }
    }

    #[test]
    fn parse_endpoint_bare_path() {
        let ep = parse_endpoint("/run/foo/bar.sock").unwrap();
        match ep {
            EmbeddingEndpoint::Unix(p) => assert_eq!(p, PathBuf::from("/run/foo/bar.sock")),
            _ => panic!("expected Unix variant"),
        }
    }

    #[test]
    fn parse_endpoint_rejects_unknown() {
        assert!(parse_endpoint("relative/path").is_err());
        assert!(parse_endpoint("ftp://example.com").is_err());
        assert!(parse_endpoint("").is_err());
    }
}
