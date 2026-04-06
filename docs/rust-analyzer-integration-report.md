# Rust-Analyzer Integration — Hecate Briefing

## What Changed

HADES can now semantically analyze Rust codebases during `codebase ingest`. Three PRs merged to `main`:

| PR | Scope |
|----|-------|
| #139 | LSP JSON-RPC transport + rust-analyzer session manager |
| #140 | Symbol extractor + edge resolver |
| #141 | Codebase ingest integration (Rust alongside Python) |

## New Files

```
core/analyzers/
├── __init__.py
├── lsp_client.py               # Generic LSP JSON-RPC transport
├── rust_analyzer_client.py     # rust-analyzer session lifecycle
├── rust_symbol_extractor.py    # Symbol/impl/call extraction per file
└── rust_edge_resolver.py       # Materializes symbol nodes + typed edges
```

## Modified Files

- `core/cli/commands/codebase.py` — Discovers `.rs` files, groups by crate, runs rust-analyzer, stores results
- `core/database/codebase_collections.py` — Added `codebase_symbols` collection

## New ArangoDB Collection

**`codebase_symbols`** — Method-level nodes derived from file node data. One document per symbol (struct, function, trait, constant, etc.).

Key format: `{file_key}__{qualified_name}` (e.g., `src_model_rs__Model__forward`)

Each symbol doc contains:
- `name`, `qualified_name`, `kind` (function, struct, trait, constant, etc.)
- `visibility` (pub, pub(crate), private)
- `file_path` (repo-relative)
- `start_line`, `end_line`
- `signature` (for functions)
- `derives` (for structs/enums)
- `is_pyo3`, `pyo3_name` (PyO3 boundary detection)
- `is_ffi` (extern "C" / #[no_mangle] detection)
- `calls` (outgoing call targets with file + position)

## New Edge Types in `codebase_edges`

| Type | From → To | Meaning |
|------|-----------|---------|
| `defines` | codebase_files → codebase_symbols | File contains this symbol |
| `calls` | codebase_symbols → codebase_symbols | Function calls function |
| `implements` | codebase_symbols → codebase_symbols | Struct implements trait |
| `pyo3_exposes` | codebase_symbols → codebase_files | Symbol exposed to Python via PyO3 |
| `ffi_exposes` | codebase_symbols → codebase_files | Symbol exposed via C FFI |

## How the Data Gets There

```
hades codebase ingest /path/to/project
```

1. `git ls-files *.rs` discovers Rust files
2. Groups files by nearest `Cargo.toml` (crate roots)
3. Starts an ephemeral rust-analyzer session per crate
4. For each `.rs` file: extracts symbols, impl blocks, call hierarchy, PyO3/FFI annotations
5. Stores full analysis as `rust_analyzer` attribute on the file node in `codebase_files`
6. `RustEdgeResolver` materializes `codebase_symbols` nodes + `codebase_edges` from that data
7. rust-analyzer shuts down — all data is now in ArangoDB

**Incremental**: On re-ingest, only changed files (by SHA-256 content hash) trigger re-analysis. When any file in a crate changes, the entire crate is re-analyzed (Rust analysis is crate-scoped).

**Graceful degradation**: If `rust-analyzer` binary is not installed, ingest continues without semantic analysis. No failure.

## How to Use It (Hecate's Workflow)

### Step 1: Ingest the NL Rust Crate

```bash
poetry run hades codebase ingest /path/to/nestedlearning --database NestedLearning
```

Or via MCP tool:
```
hades_codebase_ingest(path="/path/to/nestedlearning", database="NestedLearning")
```

### Step 2: Query Symbols via AQL

**Find all PyO3-exposed symbols:**
```aql
FOR s IN codebase_symbols
  FILTER s.is_pyo3 == true
  RETURN { name: s.name, file: s.file_path, kind: s.kind, pyo3_name: s.pyo3_name }
```

**Find all FFI boundary symbols (CUDA dispatch points):**
```aql
FOR s IN codebase_symbols
  FILTER s.is_ffi == true
  RETURN { name: s.name, file: s.file_path, signature: s.signature }
```

**Trace calls from a Python-facing method:**
```aql
FOR v, e IN 1..5 OUTBOUND 'codebase_symbols/src_model_rs__GpuStackedModel__forward'
  codebase_edges
  FILTER e.type == 'calls'
  RETURN { callee: v.name, file: v.file_path, depth: LENGTH(e) }
```

**Find all symbols in a file:**
```aql
FOR s IN codebase_symbols
  FILTER s.file_path == 'src/model.rs'
  RETURN { name: s.name, kind: s.kind, visibility: s.visibility, lines: [s.start_line, s.end_line] }
```

**Find what implements a trait:**
```aql
FOR v, e IN 1..1 INBOUND 'codebase_symbols/src_traits_rs__Forward'
  codebase_edges
  FILTER e.type == 'implements'
  RETURN { implementor: v.name, file: v.file_path }
```

### Step 3: Link Symbols to Code Smells

Method-level compliance edges:
```bash
poetry run hades link codebase_symbols/src_model_rs__forward --smell CS-32 --enforcement behavioral
```

Or via MCP:
```
hades_link(source="codebase_symbols/src_model_rs__forward", smell="CS-32", enforcement="behavioral")
```

### Step 4: Access Raw Analyzer Data on File Nodes

The file node's `rust_analyzer` attribute has the full extraction output:
```aql
FOR f IN codebase_files
  FILTER f.rel_path == 'src/model.rs'
  RETURN f.rust_analyzer
```

This includes `symbols`, `impl_blocks`, `pyo3_exports`, `ffi_boundaries`, and `analyzed_at`.

## Architecture: File Node is Source of Truth

```
codebase_files (file node)
  └── rust_analyzer: { symbols: [...], impl_blocks: [...], ... }
        │
        ├── codebase_symbols (materialized child nodes)
        │     ├── src_model_rs__Model
        │     ├── src_model_rs__Model__forward
        │     └── src_model_rs__Model__new
        │
        └── codebase_edges (materialized edges)
              ├── defines: file → symbol
              ├── calls: symbol → symbol
              ├── implements: symbol → symbol
              ├── pyo3_exposes: symbol → file
              └── ffi_exposes: symbol → file
```

Re-ingest is idempotent: update file node → delete old children → regenerate from fresh data.

## What This Enables for the PyO3 Boundary Audit

1. **Enumerate the boundary**: Query `is_pyo3 == true` to find every Rust method callable from Python
2. **Trace execution depth**: Follow `calls` edges from boundary methods into internal Rust code
3. **Identify CUDA dispatch**: Follow `calls` edges until hitting `is_ffi == true` symbols
4. **Method-level compliance**: Link individual methods to code smells (not just files)
5. **Impl verification**: Check which structs implement which traits via `implements` edges
6. **Dynamic depth**: Start at file granularity, drill into method granularity as needed

## Detected Annotations

| Pattern | Detection |
|---------|-----------|
| `#[pyclass]`, `#[pyclass(name = "...")]` | `is_pyo3 = true` on struct/enum |
| `#[pymethods]` | `is_pyo3 = true` propagated to all methods in impl block |
| `#[pyfunction]` | `is_pyo3 = true` on function |
| `#[pyo3(name = "...")]` | `pyo3_name` field populated |
| `extern "C"` | `is_ffi = true` |
| `#[no_mangle]` | `is_ffi = true` |
| `#[unsafe(no_mangle)]` | `is_ffi = true` (Rust 2024 edition) |
| `#[export_name = "..."]` | `is_ffi = true` |
