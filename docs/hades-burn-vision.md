# HADES-Burn: Vision Document

**Date:** 2026-03-28
**Status:** Exploratory — no code yet, working system (HADES) continues in parallel
**Author:** Todd Bucy

---

## 1. Motivation

### 1.1 Security

Python's runtime is fundamentally permissive in ways that create an irreducible attack surface:

- **`.pth` file persistence** — arbitrary code execution on every Python interpreter startup, even without importing the compromised package. The TeamPCP/LiteLLM attack (March 2026) used `litellm_init.pth` to harvest SSH keys, cloud credentials, Kubernetes configs, and crypto wallets from every Python process on the machine.
- **`setup.py` execution on install** — `pip install` runs arbitrary code before you've even imported anything.
- **Import side effects** — `__init__.py` runs on import; `sys.path` manipulation is unrestricted.
- **Deep dependency trees** — HADES currently pulls in hundreds of transitive dependencies, each a potential vector.

These aren't bugs — they're language design decisions. Mitigation (pinning, hashing, SLSA) helps but fights the runtime's own architecture. A compiled static binary eliminates the entire class.

**Recent precedent (Q1 2026):**
- TeamPCP cascading attack: Trivy (security scanner) → Checkmarx KICS → LiteLLM (95M monthly downloads)
- GlassWorm: 72+ malicious VS Code extensions
- 454,600 new malicious packages identified in 2025 alone (Sonatype)
- Cumulative: 1.23M known malicious packages across npm, PyPI, Maven, NuGet, Hugging Face

### 1.2 Performance

HADES is evolving toward serving as a graph-aware DataLoader for nightly autonomous model training. In a training loop, database access latency compounds across millions of iterations:

| Path | Estimated round-trip |
|---|---|
| Python → JSON → Go proxy → Unix socket → ArangoDB | ~2-5ms |
| Rust → HTTP/2 Unix socket direct | ~0.1-0.3ms |
| Rust with prefetch + connection pool | ~0.05-0.15ms |
| Rust with shared-memory cache (hit) | ~0.001-0.01ms |

Over 1M training steps: 2-5ms = **33-83 minutes** in DB access alone. 0.1ms = **1.5 minutes**.

Additional Python performance costs:
- **GIL** — no real parallelism for I/O coordination alongside compute
- **Startup time** — 300-800ms per CLI invocation (importing dependency tree)
- **Memory** — each worker process duplicates the interpreter
- **Serialization overhead** — JSON round-trips on every DB call
- **Jitter** — GC pauses and GIL contention create unpredictable latency, which is worse than consistently high latency for GPU batch scheduling

### 1.3 Deployability

Current HADES requires: Python 3.11+, Poetry, a virtual environment, system-level dependencies for Docling/torch, and careful management of dev vs prod venvs.

A Rust binary: `scp hades-burn user@machine:/usr/local/bin/`. Done.

---

## 2. Architecture

### 2.1 Core Principle

**HADES doesn't need to know what generates embeddings.** It needs vectors and metadata. Separate the infrastructure (storage, graph, search, orchestration) from the ML (inference, extraction) with a clean protocol boundary.

### 2.2 Two Systems, One Protocol

```
┌──────────────────────────────────────────────────────┐
│  HADES-Burn (100% Rust)                              │
│  ├── CLI (clap)                                      │
│  ├── ArangoDB client (HTTP/2 direct Unix socket)     │
│  ├── Graph engine (traversal, semantic search, cache) │
│  ├── Document chunker (text splitting — no ML)       │
│  ├── ArXiv sync + metadata                           │
│  ├── Config system (hades.yaml, env overrides)       │
│  ├── Training prefetcher (tokio async, batch queue)  │
│  │                                                   │
│  └── Provider API ← gRPC / Unix socket               │
│        Contract: text in → vector out                │
│        Doesn't care who implements it                │
└──────────┬───────────────────────────────────────────┘
           │
           │  gRPC / protobuf (well-defined, versionable)
           │
           │  Embed(text) → Vec<f32>
           │  EmbedBatch(texts) → Vec<Vec<f32>>
           │  Extract(bytes) → Document
           │  Info() → ProviderInfo
           │
┌──────────▼───────────────────────────────────────────┐
│  Persephone (Python — separate process)              │
│  "The intermediary between worlds"                   │
│                                                      │
│  Embedding providers (pluggable):                    │
│  ├── persephone-jina        ← our default            │
│  ├── persephone-ollama      ← community / local      │
│  ├── persephone-openai      ← API-based              │
│  ├── persephone-vllm        ← self-hosted            │
│  ├── persephone-st          ← sentence-transformers  │
│  └── (future) native Burn   ← pure Rust ONNX        │
│                                                      │
│  Extraction providers (pluggable):                   │
│  ├── persephone-docling     ← our default            │
│  ├── persephone-marker      ← alternative            │
│  └── persephone-custom      ← bring your own         │
│                                                      │
│  Also retains:                                       │
│  ├── Task management (Persephone kanban)             │
│  └── Lifecycle governance                            │
└──────────────────────────────────────────────────────┘
```

### 2.3 Security Properties

- **HADES-Burn has zero Python in its dependency tree.** No `.pth` files, no import hooks, no `setup.py`, no runtime code loading. The entire Python attack surface is eliminated from the infrastructure layer.
- **Persephone runs as a separate, sandboxable process.** It has no access to HADES credentials, database sockets, or config. It receives text, returns vectors. Even if a compromised Python dependency executes code, there's nothing to steal.
- **The protocol boundary is the security boundary.** gRPC over Unix socket with well-defined message types. No arbitrary code crossing the boundary.

### 2.4 Mythology

| Name | Role | Mythological parallel |
|---|---|---|
| **HADES** | Infrastructure — the underworld of data | God of the underworld, keeper of all souls (data) |
| **HADES-Burn** | The Rust rewrite | Cremation — the Python is burned away |
| **Persephone** | Python intermediary between HADES and ML | Lives between underworld and surface world |
| **Acheron** | Archived/dead code | River of woe, boundary of the underworld |
| **Bident** | Project's self-knowledge database | HADES' two-pronged weapon |

Persephone mediates between the underworld (HADES — deep infrastructure, storage, graph) and the surface (ML models, inference engines, the living computation). Different users bring different offerings to the intermediary.

---

## 3. Nightly Training Cycle (End-State Vision)

The long-term goal: an autonomous nightly cycle where the model trains over the knowledge graph.

```
00:00  systemd timer fires
00:01  hades arxiv sync (Rust — fast HTTP, concurrent downloads)
00:15  hades ingest --batch (Rust orchestration → Persephone for embeddings)
00:45  hades graph update (Rust — recompute edges, update indices)
01:00  Training begins
        │
        │  Python training loop imports hades_client
        │  (thin Python wrapper around gRPC, or shared-memory mmap)
        │
        │  for epoch in epochs:
        │      batch = hades.graph_batch(
        │          root_nodes=sample(new_papers),
        │          depth=2,
        │          include_edges=True,
        │      )
        │      # batch contains:
        │      #   node embeddings (zero-copy via shared memory)
        │      #   edge types + weights
        │      #   graph structure (adjacency)
        │      #   text chunks (for generation loss)
        │      loss = model(batch)
        │      loss.backward()
        │
06:00  Training complete. Model checkpointed.
06:01  Model wakes up smarter.
```

**Key performance requirements:**
- Prefetcher keeps 3-4 batches queued ahead of GPU consumption
- Graph traversal and batch assembly happen in Rust (tokio async)
- Zero-copy handoff to PyTorch via shared memory (mmap) or DLPack
- GPU never stalls waiting on data

---

## 4. Rust Crate Stack

| Component | Crate | Purpose |
|---|---|---|
| Async runtime | `tokio` | Connection pool, prefetcher, batch pipeline |
| HTTP/2 + Unix socket | `hyper` + `hyper-util` + `hyperlocal` | Direct ArangoDB connection |
| gRPC | `tonic` + `prost` | Provider protocol (HADES ↔ Persephone) |
| CLI | `clap` | Argument parsing, subcommands |
| Config | `serde` + `serde_yaml` | Drop-in for existing hades.yaml |
| JSON | `serde_json` | ArangoDB protocol |
| Concurrent cache | `moka` | Lock-free LRU with TTL |
| Connection pool | `bb8` | Pooled persistent connections |
| Shared memory | `memmap2` | Zero-copy tensor handoff for training |
| Logging | `tracing` | Structured logging, compatible with JSON output |

### Repo Layout

```
HADES-Burn/
├── crates/
│   ├── hades-core/          # DB client, graph traversal, cache, config
│   ├── hades-cli/           # Binary entry point (clap)
│   ├── hades-proto/         # gRPC/protobuf definitions
│   └── hades-prefetch/      # Async batch prefetcher for training
├── proto/
│   ├── embedding.proto      # Embed service definition
│   └── extraction.proto     # Extract service definition
├── Cargo.toml               # Workspace manifest
├── CLAUDE.md
└── README.md
```

---

## 5. Migration Strategy

**Approach:** Strangler fig. The Rust binary grows around the existing Python system, replacing modules one at a time. HADES (Python) continues working throughout.

### Phase 1 — Rust CLI Shell
- Rust binary handles CLI parsing and config loading
- Dispatches to existing Python HADES via subprocess for actual work
- **Immediate wins:** fast startup (<10ms), single binary, no `.pth` attack surface for outer layer
- **Effort:** Low. Mostly `clap` setup and subprocess management.

### Phase 2 — Rust ArangoDB Client
- Replace Go socks proxy + Python HTTP client with direct Rust HTTP/2 over Unix socket
- Implement core queries: insert, get, query, traverse, search
- AQL query passthrough
- **Wins:** Eliminate two hops from the data path. Connection pooling. Real concurrency.
- **Effort:** Medium. Well-defined protocol (ArangoDB HTTP API), but thorough work.

### Phase 3 — Rust Orchestration
- Pipeline control (extract → embed → store) moves to Rust
- Persephone provider protocol defined and implemented
- Python reduced to: Persephone providers (Jina, Docling) running as gRPC services
- ArXiv sync moves to Rust (pure HTTP, no ML)
- **Wins:** Batch ingest with real parallelism. Python fully isolated.
- **Effort:** Medium-high. Protocol design is the hard part.

### Phase 4 — Training Integration
- Shared-memory tensor bridge
- Graph-aware batch prefetcher
- Training DataLoader interface
- **Wins:** Sub-millisecond DB access in training loop. GPU saturation.
- **Effort:** Medium. Depends on Phase 2 client being solid.

### Phase 5 (Horizon) — Pure Rust ML
- Evaluate Burn framework for ONNX model inference
- If viable: `persephone-native` provider compiled into the binary
- Zero Python deployment for users who don't need custom models
- **Wins:** Single static binary for everything. Ultimate simplicity.
- **Effort:** Depends on Burn ecosystem maturity. Not blocking.

---

## 6. Open Source Positioning

```bash
# Zero Python — bring your own vectors
cargo install hades-burn
hades init --database mydb
hades ingest paper.pdf --embedding-endpoint http://localhost:8080/embed

# With our default Jina stack
pip install persephone-jina
hades config set provider.embedding persephone-jina

# With Ollama (community)
pip install persephone-ollama
hades config set provider.embedding ollama
hades config set provider.endpoint http://localhost:11434

# With OpenAI API
pip install persephone-openai
hades config set provider.embedding openai
hades config set provider.api_key $OPENAI_KEY
```

HADES-Burn is useful to anyone with ArangoDB and vectors, regardless of their ML stack. Persephone providers are optional, pluggable, and community-extensible.

---

## 7. What Stays in HADES (Python)

The existing HADES repo doesn't die — it evolves:
- Becomes the reference Persephone provider implementation
- Houses Jina V4 late chunking, Docling extraction, training scripts
- Continues working as-is during the entire migration
- Eventually slims down to just ML code with a `persephone` entry point

---

## 8. Decision Log

| Decision | Rationale |
|---|---|
| Rust over Go | PyO3 available if needed, no GC pauses, zero-copy FFI, Todd has Rust experience |
| Separate processes over in-process FFI | Cleaner security boundary, easier to sandbox Python, simpler deployment |
| gRPC over REST | Typed contracts, streaming support, efficient binary serialization, code generation |
| New repo over incremental rewrite | Can't Rustify inside a Poetry project cleanly; clean separation of concerns |
| Persephone as intermediary | Mythologically perfect; makes ML layer pluggable; isolates Python attack surface |
| Strangler fig migration | Working system continues; each phase delivers value independently |

---

## 9. NLM Integration: The Endgame

The ultimate architecture eliminates all external embedding models. The NLM (from the Hecate/NestedLearning project) becomes its own embedder and reasoning engine, with HADES as its long-term memory.

### 9.1 The Recursive Flashcard Loop

```
Round 1: NLM processes [input] → raw hidden states (2048-d)
         → HADES-Burn receives vector over Unix socket
         → AQL: vector similarity + shallow graph traversal (broad)
         → enriched context returned to NLM

Round 2: NLM processes [input + enriched context] → better hidden states
         → HADES-Burn: deeper graph traversal from Round 1 seeds
         → more specific context returned

Round 3: NLM processes [input + all context] → final output → decode
```

Two Rust binaries connected by a Unix socket. No Python in the loop.

### 9.2 Why This Works Dimensionally

The NLM target model is **2048-d** — the same dimensionality as current Jina V4 embeddings in ArangoDB. Smaller models (768-d, 1024-d) serve as testbeds.

| NLM scale | Dimensions | Role |
|---|---|---|
| Test model | 768-d | Prove the concept (fast, cheap) |
| Mid model | 1024-d | Validate scaling behavior |
| Production | 2048-d | Drop-in replacement for Jina vectors |

The 2048-d production model's vectors go in the **same field, same index, same AQL queries** as Jina vectors today. From ArangoDB's perspective, nothing changes.

### 9.3 Vector Source: NLM Hidden States

The NLM's raw output before decoding includes:
- **Final hidden states** (d_model-dimensional) — natural embedding candidates
- **Memory states M** (d_model × d_model) — richest representation but needs projection
- **Logits** (~49,152-d) — too large for efficient search

Hidden states are the natural fit. If memory states prove more semantically rich, a learned projection layer (d_model² → d_model) compresses them — one linear layer in the NLM, negligible cost.

### 9.4 Dual-Index Migration Path

ArangoDB supports multiple vector indexes on the same collection. Transition strategy:

```
Phase A (Jina-only):     embedding: [2048 floats from Jina]  ← single FAISS index
Phase B (dual):          embedding: [Jina]  +  embedding_nlm: [NLM]  ← two indexes
Phase C (NLM-only):      embedding_nlm: [NLM]  ← Jina index dropped
```

No big-bang re-embedding. NLM populates `embedding_nlm` incrementally. Queries use whichever index has coverage. Switch over when complete.

### 9.5 The Socket Protocol

Between NLM and HADES-Burn (Unix socket, raw binary — no JSON, no HTTP):

```
NLM → HADES-Burn:
  [4 bytes: vector_len as u32]
  [N × 4 bytes: f32 values]
  [4 bytes: top_k as u32]
  [4 bytes: depth as u32]

HADES-Burn → NLM:
  [4 bytes: result_count as u32]
  [for each result: length-prefixed text + metadata]
```

HADES-Burn converts f32 slice to JSON only at the ArangoDB boundary. The NLM never sees JSON.

### 9.6 AQL: Vector Search + Graph Traversal in One Query

```aql
// Stage 1: Vector similarity (find neighborhood)
LET seeds = (
  FOR doc IN embeddings
    LET score = APPROX_NEAR_COSINE(doc.embedding_nlm, @query_vector, @top_k, @n_probe)
    SORT score DESC
    LIMIT @seed_count
    RETURN { key: doc.chunk_key, score }
)
// Stage 2: Graph traversal from seeds
FOR seed IN seeds
  FOR v, e, p IN 1..@depth OUTBOUND
    CONCAT("chunks/", seed.key) edges
    RETURN { seed: seed, vertex: v, edge: e, depth: LENGTH(p.edges) }
```

One round-trip. Parameterized per-round:

| Round | seed_count | depth | Strategy |
|---|---|---|---|
| 1 | 20 | 1 | Broad — find the neighborhood |
| 2 | 5 | 3 | Deep — follow edges from best matches |
| 3 | 3 | 2 | Focused — precise context for final decode |

### 9.7 Latency Budget (3-Round Loop)

| Step | Estimated | Notes |
|---|---|---|
| NLM forward pass | ~5-20ms | Depends on sequence length, GPU |
| Socket write + read | ~0.01ms | Unix socket, raw bytes |
| HADES-Burn serialize + AQL | ~0.1ms | JSON only at ArangoDB boundary |
| ArangoDB FAISS + graph | ~0.5-2ms | Single AQL query |
| **Per round** | **~6-22ms** | |
| **Three rounds** | **~18-66ms** | Acceptable for training and interactive |

### 9.8 The Self-Sustaining Flywheel

Once the NLM is trained:
1. **Ingest**: NLM encodes new documents → 2048-d vectors → ArangoDB
2. **Train overnight**: HADES-Burn prefetches graph batches → NLM trains → encoder improves
3. **Re-embed**: Better encoder → better vectors → write back to ArangoDB
4. **Next cycle**: Better vectors → better training data → better model

Jina becomes a **bootstrap dependency only** — needed for initial embeddings before the NLM exists.

### 9.9 The Full System (Endgame)

```
nl_hecate    — Rust binary. Processes tokens, writes raw output between rounds.
hades-burn   — Rust binary. Receives vectors, queries ArangoDB, returns context.
ArangoDB     — Stores everything. Similarity + graph in one query.
Unix socket  — Connects them.

No embedding model. No Python. No third-party ML dependencies.
The model IS the embedding model.
The database IS the long-term memory.
The flashcard loop IS the reasoning engine.
```

### 9.10 Open Questions for Hecate

See companion document: `docs/hecate-prompt-nlm-embedder.md`

Key unknowns that Hecate needs to evaluate:
- Does the NLM architecture produce usable fixed-dimensional embeddings from its hidden states?
- Does the embedding space drift with each training cycle, or stabilize?
- Can the NLM handle code modality, or only text?
- What constraints from the NL papers govern self-referential improvement loops?

---

## 10. Risks and Open Questions

- **`arangors` crate maturity** — may need to build ArangoDB client on raw `hyper`. Needs evaluation.
- **Late chunking in Rust** — the chunking strategy (document-level encoding before segmentation) is currently tied to Jina's Python SDK. The protocol needs to support "encode full document, then chunk" as a provider capability.
- **Embedding dimension compatibility** — different providers produce different vector dimensions. HADES needs to handle this (per-collection metadata, or enforce at config time).
- **Burn framework timeline** — Phase 5 depends on Burn's ONNX support maturing. Not blocking, but worth tracking.
- **Shared memory for training** — mmap-based tensor handoff needs careful design to avoid copies while maintaining safety. DLPack is the likely interface.
- **NLM embedding space stability** — if training shifts the vector space each cycle, the FAISS index rebuild cost could be significant. Dual-index strategy mitigates but doesn't eliminate.
- **Code modality** — the NLM may not produce meaningful embeddings for code. Jina or a separate code-specific model may remain necessary for code ingest.
