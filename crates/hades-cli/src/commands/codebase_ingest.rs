//! Native Rust implementation of the `hades codebase ingest` command.
//!
//! Walks a directory (or single file), detects language, runs AST
//! analysis, chunks at function/class boundaries, embeds chunks via
//! the Persephone embedder, and stores everything in dedicated codebase
//! collections.
//!
//! Supports:
//! - Recursive directory traversal (respects common ignore patterns)
//! - Language auto-detection from file extension
//! - Incremental ingestion via symbol_hash comparison
//! - Python import graph resolution (file→file edges)
//! - Rust and Go semantic enrichment through language servers
//! - Per-file error isolation in batch mode

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use serde_json::{Value, json};
use tracing::{debug, error, info, warn};

use hades_core::HadesConfig;
use hades_core::chunking::ChunkingStrategy;
use hades_core::code::lsp::go_symbols::GoSymbolExtractor;
use hades_core::code::lsp::symbols::FileExtraction;
use hades_core::code::lsp::{
    EdgeKind, GoplsSession, LspEdgeResolver, RustAnalyzerSession, RustSymbolExtractor,
    group_files_by_crate, group_files_by_go_module,
};
use hades_core::code::{
    self, AnalysisOptions, AnalysisTier, AnalyzerOutcome, AstChunking, Language, Symbol, SymbolKind,
};
use hades_core::code::{cpp_edges, python_calls, rust_imports, tree_sitter_edges};
use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::keys;
use hades_core::db::query::ExecutionTarget;
use hades_core::db::{ArangoErrorKind, ArangoPool};
use hades_core::ingest_routing::{self, Route};
use hades_core::persephone::embedding::EmbeddingClient;

use super::output::{self, OutputFormat};

/// Hard floor for directory exclusions — applied even when the project has
/// no `.gitignore`, no `.ignore`, and no `.hadesignore`. The `ignore` crate's
/// standard filters (gitignore + hidden-file skipping) already handle most
/// real repos; this list catches the unfortunate case of a flat directory
/// dropped onto disk without any ignore files.
const SKIP_DIRS: &[&str] = &[
    "__pycache__",
    "node_modules",
    "target",
    "venv",
    "dist",
    "build",
];

/// Per-file result for JSON output.
#[derive(serde::Serialize)]
struct FileResult {
    path: String,
    success: bool,
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_symbols: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_embeddings: Option<usize>,
    /// Embedding preparation failed; the previous committed graph is retained.
    /// The summary keeps this diagnostic separate from database write failures.
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skipped: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    duration_ms: u64,
}

/// Codebase ingest command failed with partial results.
#[derive(Debug, thiserror::Error)]
#[error("{failed} of {total} files failed to ingest")]
pub struct CodebaseIngestFailure {
    pub total: usize,
    pub failed: usize,
}

/// Accumulators for cross-file import resolution.
///
/// Collects per-file import data during the ingest loop so that
/// import edges can be resolved in a batch pass after all files are processed.
#[derive(Default)]
struct ImportContext {
    /// Python: rel_path → list of import symbols (with metadata for resolution).
    python_imports: HashMap<String, Vec<Symbol>>,
    /// Python: rel_path → all definition symbols (for building the resolution index).
    python_file_symbols: HashMap<String, Vec<Symbol>>,
    /// Rust: rel_path → list of expanded use-paths.
    rust_imports: HashMap<String, Vec<String>>,
    /// Rust: rel_path → all symbols (for building the resolution index).
    rust_file_symbols: HashMap<String, Vec<Symbol>>,
    /// C/C++/CUDA: rel_path → semantic symbols and resolved call metadata.
    cpp_file_symbols: HashMap<String, Vec<Symbol>>,
    /// Lower-fidelity files used for syntax-only relationship resolution.
    structural_file_symbols: HashMap<String, Vec<Symbol>>,
    /// Inbound edges remapped inside acknowledged file transactions.
    repointed_edges: u64,
    committed_revisions: HashMap<String, String>,
    remapped_symbols: usize,
}

/// One half of an ingest, as data rather than as printed output.
///
/// The unified `hades ingest` runs the code phase and the document phase against
/// one root and emits a single envelope, so neither phase may print its own. The
/// failure travels beside the summary instead of replacing it: a run that
/// ingested 180 files and lost its enrichment has to report both.
pub struct PhaseOutcome {
    /// The phase's summary, as it would have appeared in its own envelope.
    pub data: Value,
    /// Set when the phase finished but its outcome is a failure.
    pub failure: Option<anyhow::Error>,
}

/// Run the codebase ingest command, printing its own envelope.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    batch: bool,
    unparsed_ext: &[String],
    compile_commands: Option<&Path>,
    force: bool,
    allow_analysis_downgrade: bool,
) -> Result<()> {
    let outcome = run_phase(
        config,
        path,
        language,
        batch,
        unparsed_ext,
        compile_commands,
        force,
        allow_analysis_downgrade,
    )
    .await?;
    output::print_output("codebase.ingest", outcome.data, &OutputFormat::Json);
    match outcome.failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The code phase itself.
// TODO: support --batch to enable parallel/batched ingestion
#[allow(clippy::too_many_arguments)]
pub async fn run_phase(
    config: &HadesConfig,
    path: PathBuf,
    language: Option<&str>,
    batch: bool,
    unparsed_ext: &[String],
    compile_commands: Option<&Path>,
    force: bool,
    allow_analysis_downgrade: bool,
) -> Result<PhaseOutcome> {
    let cmd_start = Instant::now();

    // Resolve the ingest root before anything derives from it.
    let path = resolve_ingest_root(&path)?;

    let unparsed_set = normalize_unparsed_ext(unparsed_ext);
    let lang_override = parse_language_arg(language)?;

    // Connect to services.
    let db = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // Embedding is optional — ingest proceeds without vectors if the service is unavailable.
    let embedder = match EmbeddingClient::connect_at(&config.embedding.service.socket).await {
        Ok(client) => {
            info!("connected to embedding service");
            Some(client)
        }
        Err(e) => {
            warn!(error = %e, "embedding service unavailable — ingesting without vectors");
            None
        }
    };

    // Ask the backend what it will accept, once per run. The answer is a
    // property of the load profile the embedder happens to be running, so it
    // cannot be a constant and must not be read per file.
    let window_chars = embed_window_chars(embedder.as_ref()).await;

    // Ensure codebase collections exist.
    ensure_collections(&db).await?;

    // Discover source files.
    let files = discover_files(&path, lang_override, &unparsed_set)?;
    if files.is_empty() {
        return Ok(PhaseOutcome {
            data: json!({ "total": 0, "message": "no supported source files found" }),
            failure: None,
        });
    }

    info!(file_count = files.len(), "discovered source files");

    // Compute base path for relative paths.
    let base = ingest_base_path(&path);
    let namespace = base.to_str().context("ingest root must be valid UTF-8")?;
    preflight_file_identities(&db, &base, &files).await?;

    // ── Analyzer preflight (#164/#167) ─────────────────────────────────
    // Resolve each needed analyzer (config/env override wins over PATH) and
    // probe it FROM the ingest base, because the rustup shim resolves
    // per-directory. This runs BEFORE any file is touched: `--force` purges a
    // file's semantic edges on the assumption enrichment will rebuild them,
    // so an analyzer that cannot run must stop the ingest up front — after
    // the purge is too late (#164). `--allow-analysis-downgrade` is the
    // explicit override, matching its existing fidelity semantics.
    let needs_rust = files
        .iter()
        .any(|f| is_semantic_target(f, lang_override, &unparsed_set, Language::Rust, "rs"));
    let needs_go = files
        .iter()
        .any(|f| is_semantic_target(f, lang_override, &unparsed_set, Language::Go, "go"));
    let rust_analyzer_cmd = if needs_rust {
        preflight_or_bail(
            "rust-analyzer",
            config.analyzers.rust_analyzer.as_deref(),
            &base,
            allow_analysis_downgrade,
        )?
    } else {
        None
    };
    let gopls_cmd = if needs_go {
        preflight_or_bail(
            "gopls",
            config.analyzers.gopls.as_deref(),
            &base,
            allow_analysis_downgrade,
        )?
    } else {
        None
    };

    // Process each file with per-file error isolation.
    let mut results: Vec<FileResult> = Vec::with_capacity(files.len());
    // File keys whose symbol set was rebuilt this run — the inputs to the
    // post-run dangling-edge sweep (#183).
    let mut rewritten_file_keys: Vec<String> = Vec::new();
    // Accumulators for cross-file import resolution.
    let mut imports = ImportContext {
        python_imports: HashMap::new(),
        python_file_symbols: HashMap::new(),
        rust_imports: HashMap::new(),
        rust_file_symbols: HashMap::new(),
        cpp_file_symbols: HashMap::new(),
        structural_file_symbols: HashMap::new(),
        repointed_edges: 0,
        committed_revisions: HashMap::new(),
        remapped_symbols: 0,
    };
    // Collect absolute paths for Rust files — used for rust-analyzer post-loop phase.
    let mut rust_abs_paths: Vec<PathBuf> = Vec::new();
    // Go starts with Tree-sitter and is semantically enriched by gopls after
    // the full module file set is available.
    let mut go_abs_paths: Vec<PathBuf> = Vec::new();

    // Auto-activate batch mode for large input sets.
    let batch_mode = batch || files.len() > 5;

    let total_files = files.len();
    for (idx, file_path) in files.iter().enumerate() {
        if batch_mode {
            let progress = json!({
                "type": "progress",
                "current": idx + 1,
                "total": total_files,
                "percent": ((idx + 1) as f64 / total_files as f64 * 100.0),
            });
            eprintln!("{}", serde_json::to_string(&progress).unwrap_or_default());
        }

        let item_start = Instant::now();
        let rel_path = rel_path_for(&base, file_path);

        // Route unparsed-allowlisted files (e.g. CUDA `.cu`) through the
        // parser-free fallback: line/size chunk + embed, no AST (#121).
        let file_ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        // The unparsed allowlist is orthogonal to `--language`: an allowlisted
        // extension with no recognized parser (e.g. `.cu`) always takes the
        // parser-free path, even when `--language` is set for other files.
        // `discover_files` likewise includes these regardless of the override.
        // Extensionless scripts are classified by shebang: a `#!…python3` file
        // gets the Python analyzer, a `#!/bin/bash` one has no analyzer and
        // takes the raw-text path rather than erroring out (#183).
        let shebang = if file_ext.is_none() {
            shebang_of(file_path)
        } else {
            None
        };
        let shebang_lang = shebang.and_then(|(_, lang)| lang);
        let has_shebang = shebang.is_some();
        // An explicit `--language` still wins; shebang only fills the gap where
        // there was previously no signal at all.
        let effective_override = lang_override.or(shebang_lang);
        // The unparsed allowlist stays orthogonal to `--language`: an
        // allowlisted extension with no parser takes the raw-text path even when
        // `--language` is set for other files. Only the shebang clause is gated,
        // and only on a shebang-derived language — gating the whole predicate on
        // `effective_override` would send `--unparsed-ext sh` files through
        // whatever `--language` names, and diverge from `is_semantic_target`,
        // which is the split #164 exists to prevent.
        let is_unparsed = Language::from_path(&rel_path).is_none()
            && (file_ext
                .as_deref()
                .is_some_and(|e| unparsed_set.contains(e))
                || (has_shebang && shebang_lang.is_none() && lang_override.is_none()));

        // Track Rust/Go files for post-loop semantic enrichment (parsed only).
        // Uses the SAME predicate as the preflight gate above, so the set the
        // gate protects and the set the phases process cannot diverge — a
        // divergence here is exactly how `--language rust` on non-.rs files
        // would skip the preflight and silently lose enrichment (#164).
        if is_semantic_target(
            file_path,
            lang_override,
            &unparsed_set,
            Language::Rust,
            "rs",
        ) {
            rust_abs_paths.push(file_path.clone());
        }
        if is_semantic_target(file_path, lang_override, &unparsed_set, Language::Go, "go") {
            go_abs_paths.push(file_path.clone());
        }

        let result = if is_unparsed {
            ingest_unparsed_file(
                &db,
                embedder.as_ref(),
                config,
                file_path,
                &rel_path,
                None,
                "no registered language or grammar",
                force,
                allow_analysis_downgrade,
                namespace,
            )
            .await
        } else {
            ingest_file(
                &db,
                embedder.as_ref(),
                config,
                file_path,
                &rel_path,
                effective_override,
                &mut imports,
                compile_commands,
                force,
                allow_analysis_downgrade,
                gopls_cmd.is_some(),
                window_chars,
                namespace,
            )
            .await
        };

        let duration = item_start.elapsed().as_millis() as u64;
        match result {
            Ok(r) => {
                // A file that was actually rewritten (not skipped) had its symbol
                // set purged and rebuilt, so a symbol another file points at may
                // have disappeared. Remember it for the post-run dangling sweep.
                if r.skipped != Some(true) && r.success {
                    rewritten_file_keys.push(keys::scoped_file_key(namespace, &rel_path));
                }
                results.push(FileResult {
                    duration_ms: duration,
                    ..r
                })
            }
            Err(e) => {
                error!(path = %rel_path, error = %e, "ingest failed");
                results.push(FileResult {
                    path: rel_path,
                    success: false,
                    language: None,
                    num_symbols: None,
                    num_chunks: None,
                    num_embeddings: None,
                    embedding_error: None,
                    skipped: None,
                    error: Some(e.to_string()),
                    duration_ms: duration,
                });
            }
        }
    }

    // Resolve Python import graph edges (file→symbol where possible, file→file fallback).
    let py_symbol_index = build_python_symbol_index_scoped(&imports.python_file_symbols, namespace);
    let py_import_edges = resolve_python_imports_scoped(
        &imports.python_imports,
        &imports.python_file_symbols,
        &py_symbol_index,
        namespace,
    );

    // Resolve Python call graph edges (symbol → symbol). Uses the calls + parent_symbol
    // metadata attached during AST extraction; reuses the bare-name index already built
    // for imports as the Strategy-3 fallback.
    let py_qualified_index =
        python_calls::build_qualified_index_scoped(&imports.python_file_symbols, namespace);
    let py_call_edges = python_calls::resolve_python_calls_scoped(
        &imports.python_file_symbols,
        &py_qualified_index,
        &py_symbol_index,
        namespace,
    );

    // Resolve compiler-grade C/C++/CUDA calls, including CUDA kernel launches.
    // libclang records target USRs and definition spans during each file parse;
    // this batch phase maps them onto HADES's cross-file span keys.
    let cpp_call_edges =
        cpp_edges::resolve_cpp_calls_scoped(&base, &imports.cpp_file_symbols, namespace);

    let structural_edges =
        tree_sitter_edges::resolve_scoped(&imports.structural_file_symbols, namespace);

    // Resolve Rust import graph edges (file → symbol).
    let rust_symbol_index =
        rust_imports::build_symbol_index_scoped(&imports.rust_file_symbols, namespace);
    let rs_import_edges = rust_imports::resolve_rust_imports_scoped(
        &imports.rust_imports,
        &rust_symbol_index,
        namespace,
    );

    // This separate enrichment stage is all-or-nothing. File replacements
    // preceding it remain committed, but failures cannot masquerade as success.
    let relationship_error = if results.iter().any(|result| !result.success) {
        Some(
            "relationship stage deferred because file preparation failed; retry ingestion"
                .to_owned(),
        )
    } else {
        super::codebase_persist::store_relationships(
        &db,
        std::mem::take(&mut imports.committed_revisions),
        vec![
            (CODEBASE.imports_edges, py_import_edges.clone()),
            (CODEBASE.calls_edges, py_call_edges.clone()),
            (CODEBASE.calls_edges, cpp_call_edges.clone()),
            (CODEBASE.calls_edges, structural_edges.calls.clone()),
            (CODEBASE.imports_edges, structural_edges.imports.clone()),
            (CODEBASE.imports_edges, rs_import_edges.clone()),
        ],
    ).await.err().map(|error| format!("failed to atomically store cross-file relationships; earlier file replacements remain committed: {error}"))
    };
    let stored_relationships = usize::from(relationship_error.is_none());

    let total_import_edges =
        py_import_edges.len() + rs_import_edges.len() + structural_edges.imports.len();

    // ── rust-analyzer deep analysis ────────────────────────────────────
    // When Rust files were ingested, optionally use rust-analyzer for richer
    // symbol extraction: qualified names, call hierarchy, impl-trait edges,
    // PyO3/FFI detection. This enrichment phase runs after the syn-based loop.
    let ra_stats = if relationship_error.is_none()
        && !rust_abs_paths.is_empty()
        && rust_analyzer_cmd.is_some()
    {
        match run_rust_analyzer_phase(&db, &base, &rust_abs_paths, rust_analyzer_cmd.as_deref())
            .await
        {
            Ok(stats) => {
                info!(
                    symbols = stats.symbols,
                    edges = stats.edges,
                    crates = stats.workspaces,
                    store_errors = stats.store_errors,
                    "rust-analyzer enrichment complete"
                );
                stats
            }
            Err(e) => {
                // `{e:#}` prints the full anyhow context chain — plain `{e}`
                // shows only the outermost context and swallows the underlying
                // cause (#180).
                warn!(error = %format!("{e:#}"), "rust-analyzer enrichment failed, syn-based data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    // Total enrichment failure is loud, not an info line (#164). The preflight
    // passed, Rust files were ingested, and the phase produced nothing — that
    // is the exact state that previously reported success while the graph
    // silently lost its calls/implements layer. Only the downgrade flag makes
    // proceeding an explicit choice.
    if !rust_abs_paths.is_empty()
        && rust_analyzer_cmd.is_some()
        && ra_stats.workspaces == 0
        && !ra_stats.store_failed
        && !allow_analysis_downgrade
    {
        anyhow::bail!(
            "rust-analyzer enrichment produced nothing across {} Rust file(s) \
             (crates_analyzed = 0) despite a passing preflight. The graph would \
             keep syn symbols but lose calls/implements edges. Investigate the \
             analyzer session logs above, or pass --allow-analysis-downgrade to \
             accept the loss explicitly.",
            rust_abs_paths.len()
        );
    }

    // A store failure is a distinct stage from analysis failure (#180): the
    // analyzer did its job, but the results never reached ArangoDB. Attribute
    // it correctly so operators don't chase analyzer session logs for a
    // database error.
    if ra_stats.store_failed && !allow_analysis_downgrade {
        anyhow::bail!(
            "rust-analyzer analyzed {} crate(s) but storing the enrichment to \
             ArangoDB failed (see the store warnings above for the database \
             error). The graph keeps syn symbols but loses calls/implements \
             edges. Fix the store error and re-run, or pass \
             --allow-analysis-downgrade to accept the loss explicitly.",
            ra_stats.workspaces
        );
    }

    let gopls_stats = if relationship_error.is_none()
        && !go_abs_paths.is_empty()
        && gopls_cmd.is_some()
    {
        match run_gopls_phase(&db, &base, &go_abs_paths, gopls_cmd.as_deref()).await {
            Ok(stats) => {
                info!(
                    symbols = stats.symbols,
                    edges = stats.edges,
                    modules = stats.workspaces,
                    "gopls semantic enrichment complete"
                );
                stats
            }
            Err(error) => {
                warn!(error = %format!("{error:#}"), "gopls enrichment failed; Tree-sitter Go data retained");
                SemanticLspStats::default()
            }
        }
    } else {
        SemanticLspStats::default()
    };

    // Enrichment failures are recorded rather than raised here, and returned
    // after the summary JSON is printed. `codebase ingest` writes the graph
    // before these checks run, so unwinding early exits non-zero having done
    // the work and printed nothing on stdout — which reads as "ingest failed,
    // nothing written" to anything parsing the output (#194 review). The
    // existing `CodebaseIngestFailure` at the end of this function already
    // follows print-then-fail; these now match it.
    let mut enrichment_failure: Option<String> = None;

    // Same store-vs-analysis attribution as the rust-analyzer path (#180):
    // gopls analyzed its modules but the results never reached ArangoDB, so
    // exiting 0 would silently lose the Go calls/implements layer.
    if gopls_stats.store_failed && !allow_analysis_downgrade {
        enrichment_failure.get_or_insert_with(|| {
            format!(
                "gopls analyzed {} module(s) but storing the enrichment to ArangoDB \
             failed (see the store warnings above for the database error). The \
             graph keeps Tree-sitter symbols but loses calls/implements edges. \
             Fix the store error and re-run, or pass --allow-analysis-downgrade \
             to accept the loss explicitly.",
                gopls_stats.workspaces
            )
        });
    }

    // Same loud-zero rule as rust-analyzer (#164), widened to partial failure.
    //
    // A passing preflight with Go files ingested and zero modules analyzed is
    // silent semantic loss. So is *some* modules analyzed — but only when this
    // run actually rewrote files in a module that failed. The fidelity guard
    // stands aside for `.go` files whenever the phase is scheduled (#193), so a
    // failed module means the files it owns were purged and rebuilt at
    // `structural` with nothing restoring their semantic layer.
    //
    // Gated on real loss, not on the failure alone: `go_abs_paths` comes from
    // discovery and includes files this run skipped as unchanged. Those were
    // never purged, so the previous run's gopls symbols and edges are still in
    // the graph and there is nothing to report. Bailing there would assert a
    // loss that did not happen, on the common case of re-running ingest over an
    // unchanged tree with one perpetually-unhappy module.
    let rewritten_go_under_failed_module: Vec<&str> = if gopls_stats.failed_workspaces.is_empty() {
        Vec::new()
    } else {
        results
            .iter()
            .filter(|r| !r.skipped.unwrap_or(false) && r.success && r.path.ends_with(".go"))
            .filter(|r| {
                let absolute = base.join(&r.path);
                gopls_stats
                    .failed_workspaces
                    .iter()
                    .any(|root| absolute.starts_with(root))
            })
            .map(|r| r.path.as_str())
            .collect()
    };

    if !go_abs_paths.is_empty() && gopls_cmd.is_some() && !allow_analysis_downgrade {
        if gopls_stats.workspaces == 0 {
            enrichment_failure.get_or_insert_with(|| {
                format!(
                    "gopls enrichment produced nothing across {} Go file(s) despite a \
                 passing preflight. Investigate the session logs above, or pass \
                 --allow-analysis-downgrade to accept the loss explicitly.",
                    go_abs_paths.len()
                )
            });
        } else if !rewritten_go_under_failed_module.is_empty() {
            enrichment_failure.get_or_insert_with(|| {
                format!(
                    "gopls analyzed {} of {} Go module(s); {} failed to start a session \
                 (see the warnings above). {} file(s) under the failed module(s) were \
                 re-ingested this run, so their gopls symbols and calls/implements \
                 edges are gone — the fidelity guard stands aside for Go on the \
                 expectation that this phase re-supplies them. Fix the failing \
                 module(s) and re-run, or pass --allow-analysis-downgrade to accept \
                 the loss explicitly.",
                    gopls_stats.workspaces,
                    gopls_stats.workspaces_attempted,
                    gopls_stats.failed_workspaces.len(),
                    rewritten_go_under_failed_module.len()
                )
            });
        } else if !gopls_stats.failed_workspaces.is_empty() {
            warn!(
                analyzed = gopls_stats.workspaces,
                attempted = gopls_stats.workspaces_attempted,
                failed = gopls_stats.failed_workspaces.len(),
                "gopls could not start a session for some module(s); no file under \
                 them was rewritten this run, so nothing was lost"
            );
        }
    }

    // Report inbound edges left dangling by this run's rebuilds.
    //
    // Purging a file removes only its *outgoing* edges, by design — inbound ones
    // are owned by other source files this run may not have touched. A rebuild
    // that drops a symbol (rename, re-qualification, analyzer change) leaves
    // those pointing at nothing, which fails the `imports_edge_endpoints`
    // invariant in `codebase validate` (#183). Counted, not deleted: the edge
    // records a real dependency, and removing it would erase the only signal
    // that the dependent needs re-ingesting. Runs after the enrichment phases so
    // a symbol rust-analyzer/gopls recreates is not counted as gone.
    // File transactions already re-pointed moved symbols (#9). A symbol that kept
    // its name and changed only its line is the common case, and its dependents
    // were skipped precisely because nothing about them changed -- so the edge is
    // still correct about the dependency and wrong only about the key.
    let repointed = imports.repointed_edges;
    if repointed > 0 {
        info!(
            edges = repointed,
            symbols = imports.remapped_symbols,
            "re-pointed inbound edges onto symbols that moved without being renamed"
        );
    }

    let dangling_inbound = count_dangling_inbound(&db, &rewritten_file_keys).await;
    if dangling_inbound > 0 {
        warn!(
            edges = dangling_inbound,
            "inbound edges point at symbols this run removed and could not be \
             re-pointed: the symbol was renamed or dropped, or its qualified name \
             gained or lost a sibling, which makes position stop identifying it; \
             re-run `codebase ingest --force <the same ingest root>` to re-resolve \
             them (--force because the dependents' own content_hash is unchanged, \
             and the original root because keys are relative to it -- a narrower \
             path re-bases them and writes duplicate nodes), or run \
             `hades codebase prune-orphans` to drop them"
        );
    }

    // Output summary.
    let total = results.len();
    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results
        .iter()
        .filter(|r| !r.success && r.skipped != Some(true))
        .count();
    let skipped = results.iter().filter(|r| r.skipped == Some(true)).count();
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let files_embedded = results
        .iter()
        .filter(|r| r.num_embeddings.is_some_and(|n| n > 0))
        .count();
    let total_embeddings: usize = results.iter().filter_map(|r| r.num_embeddings).sum();

    // Files preserved because embedding preparation failed (e.g. embedder
    // GPU OOM, timeout). Their per-file `embedding_error` carries the message;
    // we surface a count + the affected paths here so a green-looking "completed"
    // doesn't hide silent vector loss.
    let embedding_failures: Vec<&str> = results
        .iter()
        .filter(|r| r.embedding_error.is_some())
        .map(|r| r.path.as_str())
        .collect();

    let result_data = json!({
        "total": total,
        "completed": succeeded,
        "failed": failed,
        "skipped": skipped,
        "embedding": {
            "service_connected": embedder.is_some(),
            "files_embedded": files_embedded,
            "total_embeddings": total_embeddings,
            "files_with_embedding_failures": embedding_failures.len(),
            "embedding_failure_paths": embedding_failures,
        },
        // Inbound edges now pointing at symbols this run removed. Surfaced in
        // the JSON contract, not just stderr, so an agent parsing stdout sees
        // that the graph needs attention rather than only "completed: N".
                        "dangling_inbound_edges": dangling_inbound,
        // Inbound edges re-pointed onto symbols that moved without being
        // renamed (#9), as opposed to the ones above, which could not be.
        "repointed_inbound_edges": repointed,
        "relationship_error": relationship_error,
        "import_edges": total_import_edges * stored_relationships,
        "python_import_edges": py_import_edges.len() * stored_relationships,
        "rust_import_edges": rs_import_edges.len() * stored_relationships,
        "python_call_edges": py_call_edges.len() * stored_relationships,
        "cpp_call_edges": cpp_call_edges.len() * stored_relationships,
        "structural_call_edges": structural_edges.calls.len() * stored_relationships,
        "structural_import_edges": structural_edges.imports.len() * stored_relationships,
        "rust_analyzer": {
            "symbols": ra_stats.symbols,
            "edges": ra_stats.edges,
            "crates_analyzed": ra_stats.workspaces,
            "store_errors": ra_stats.store_errors,
            "store_failed": ra_stats.store_failed,
        },
        "gopls": {
            "symbols": gopls_stats.symbols,
            "edges": gopls_stats.edges,
            "modules_analyzed": gopls_stats.workspaces,
            "store_errors": gopls_stats.store_errors,
            "store_failed": gopls_stats.store_failed,
        },
        "results": results,
        "duration_ms": duration_ms,
    });

    // The failure travels with the summary rather than replacing it, so the
    // caller can emit both. Enrichment loss outranks per-file failures because
    // it means the graph is missing a whole layer of edges.
    let failure = if let Some(message) = relationship_error {
        Some(anyhow::anyhow!("{message}"))
    } else if let Some(message) = enrichment_failure {
        Some(anyhow::anyhow!("{message}"))
    } else if failed > 0 {
        Some(CodebaseIngestFailure { total, failed }.into())
    } else {
        None
    };

    Ok(PhaseOutcome {
        data: result_data,
        failure,
    })
}
/// Escape hatch for the late-chunking path, read once per process.
static LATE_CHUNKING_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("HADES_DISABLE_LATE_CHUNKING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

/// Character budget for one embed window, derived from the backend's ceiling.
///
/// The window is packed by characters because chunk boundaries are character
/// offsets, while the ceiling is in tokens, so the conversion needs a
/// chars-per-token figure. It has to be a *floor*, not an average: pack by the
/// average and the densest files overshoot the ceiling.
///
/// 2.0 is measured, not guessed. Across 285 files of real Rust, Python, CUDA and
/// markdown the lowest ratio observed was 2.18 chars per token, the 5th
/// percentile 3.68 and the median 4.18. So 2.0 leaves headroom under the densest
/// file in that corpus, and a typical file still fills roughly half the window.
///
/// The constant this replaced was 12,000 characters, justified by a comment
/// claiming dense Rust runs 1.5 chars per token. Measurement puts it at 4.19, so
/// that budget packed about 2,870 tokens against a 32,768-token card and split
/// 134 of those 285 files, producing 672 windows where 313 suffice. Every extra
/// window is a seam where a chunk's vector loses the surrounding file, which is
/// the whole point of late chunking.
const CHARS_PER_TOKEN_FLOOR: f64 = 2.0;

/// Fallback budget when the backend does not report `max_seq_length`.
///
/// The previous hardcoded value, kept for a backend that predates the field:
/// under-packing is a quality loss, and overshooting an unknown ceiling is a
/// refused window, so the conservative number is the right guess when there is
/// nothing to read.
const FALLBACK_WINDOW_CHARS: usize = 12_000;

/// Ask the backend what it will accept and convert it to a character budget.
async fn embed_window_chars(embedder: Option<&EmbeddingClient>) -> usize {
    let Some(client) = embedder else {
        return FALLBACK_WINDOW_CHARS;
    };
    match client.info().await {
        Ok(info) => match info.max_seq_length {
            Some(ceiling) => {
                let budget = (f64::from(ceiling) * CHARS_PER_TOKEN_FLOOR) as usize;
                info!(
                    ceiling_tokens = ceiling,
                    window_chars = budget,
                    profile = info.profile.as_deref().unwrap_or("unreported"),
                    "embed window sized from the backend's ceiling"
                );
                budget
            }
            None => {
                warn!(
                    fallback = FALLBACK_WINDOW_CHARS,
                    "backend does not report max_seq_length, using the conservative budget"
                );
                FALLBACK_WINDOW_CHARS
            }
        },
        Err(e) => {
            warn!(error = %e, fallback = FALLBACK_WINDOW_CHARS,
                  "could not read the backend's ceiling, using the conservative budget");
            FALLBACK_WINDOW_CHARS
        }
    }
}

/// One unit of text handed to the embedder in a single forward pass.
///
/// A file smaller than the model's context window is one window. A larger file
/// is split into several, which is the pre-chunking escape hatch: chunk
/// boundaries inside a window keep their AST alignment, so only the seams
/// between windows lose cross-chunk context.
struct EmbedWindow {
    /// The window's source text, sent to the embedder whole.
    text: String,
    /// Chunk boundaries as character ranges relative to `text`, not the file.
    boundaries: Vec<(usize, usize)>,
    /// File-order index of each boundary, positionally parallel to
    /// `boundaries`.
    ///
    /// Not a first index plus an offset. The packing loop skips a chunk too large
    /// for any window, so a window's boundaries are not necessarily consecutive
    /// in file order, and `first_chunk_index + position` then named the wrong
    /// chunk for every boundary after the gap: each vector would have been stored
    /// under a later chunk's key, with the count still matching and nothing to
    /// show it. Carrying the indices makes the mapping explicit.
    chunk_indices: Vec<usize>,
}

/// Finalize one embed window: slice its text and convert its boundaries.
///
/// Skips the window rather than panicking when the offsets do not describe a
/// valid slice. `TextChunk` offsets come from AST spans and from line
/// arithmetic that can drift, since `str::lines()` strips `\r` and so a CRLF
/// source undercounts by one byte per line. Before late chunking these offsets
/// were only ever stored as metadata, so drift was harmless. Slicing with them
/// makes it fatal, and `&str[a..b]` panics on an index inside a multibyte
/// sequence.
#[allow(clippy::too_many_arguments)]
fn push_embed_window(
    windows: &mut Vec<EmbedWindow>,
    source: &str,
    start: usize,
    end: usize,
    byte_boundaries: Vec<(usize, usize)>,
    chunk_indices: Vec<usize>,
    rel_path: &str,
) {
    let end = end.min(source.len());
    let Some(text) = source.get(start..end) else {
        warn!(
            path = rel_path,
            start, end, "chunk offsets do not land on character boundaries, window skipped"
        );
        return;
    };
    debug_assert_eq!(
        byte_boundaries.len(),
        chunk_indices.len(),
        "every boundary needs the file-order index of the chunk it came from"
    );
    windows.push(EmbedWindow {
        boundaries: byte_offsets_to_chars(text, &byte_boundaries),
        text: text.to_string(),
        chunk_indices,
    });
}

/// Convert byte offsets into `text` to character offsets.
///
/// `TextChunk::start_char` and `end_char` are byte offsets despite their
/// names. The embedding server maps boundaries against the tokenizer's offset
/// mapping, which is character-indexed, so passing bytes straight through made
/// the two sides disagree on every file containing a multibyte character, with
/// the gap widening through the file. Nothing failed, because the pooling was
/// correct over whatever range it was given.
///
/// An offset landing inside a multibyte sequence rounds down to the character
/// containing it, which is the only interpretation that keeps ranges ordered.
fn byte_offsets_to_chars(text: &str, offsets: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut wanted: Vec<usize> = offsets.iter().flat_map(|(s, e)| [*s, *e]).collect();
    wanted.sort_unstable();
    wanted.dedup();

    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut w = 0usize;
    for (char_idx, (byte_idx, _)) in text.char_indices().enumerate() {
        while w < wanted.len() && wanted[w] < byte_idx {
            map.insert(wanted[w], char_idx.saturating_sub(1));
            w += 1;
        }
        if w < wanted.len() && wanted[w] == byte_idx {
            map.insert(wanted[w], char_idx);
            w += 1;
        }
    }
    // Anything at or past the end maps to the character count, so an exclusive
    // end offset of `text.len()` stays exclusive.
    let total_chars = text.chars().count();
    while w < wanted.len() {
        map.insert(wanted[w], total_chars);
        w += 1;
    }

    offsets.iter().map(|(s, e)| (map[s], map[e])).collect()
}

// ── Collection setup ────────────────────────────────────────────────────

/// Ensure all codebase collections, named graph, and indices exist.
///
/// Creation order (per ontology spec §7.1):
/// 1. Document collections (files, chunks, embeddings, symbols)
/// 2. Edge collections (defines, calls, implements, imports)
/// 3. Named graph `codebase_graph` via Gharial API
/// 4. Persistent indices on document collections
async fn ensure_collections(db: &ArangoPool) -> Result<()> {
    // Step 1–2: Create collections.
    let existing = crud::list_collections(db, false)
        .await
        .context("failed to list collections")?;
    let existing_names: Vec<&str> = existing.iter().map(|c| c.name.as_str()).collect();

    for (name, col_type) in CODEBASE.all_collections() {
        if !existing_names.contains(&name) {
            info!(collection = name, col_type, "creating collection");
            crud::create_collection(db, name, Some(col_type))
                .await
                .with_context(|| format!("failed to create collection: {name}"))?;
        }
    }

    // Step 3: Create named graph (idempotent — 409 means it already exists).
    ensure_named_graph(db).await?;

    // Step 4: Ensure persistent indices.
    ensure_indices(db).await?;

    Ok(())
}

/// The named graph name.
const CODEBASE_GRAPH: &str = "codebase_graph";

/// Create the `codebase_graph` named graph via the Gharial API.
///
/// The named graph enforces `_from`/`_to` vertex constraints at insert
/// time — an edge with `_from` pointing to the wrong collection is
/// rejected by ArangoDB rather than silently corrupting the graph.
async fn ensure_named_graph(db: &ArangoPool) -> Result<()> {
    let body = json!({
        "name": CODEBASE_GRAPH,
        "edgeDefinitions": [
            {
                "collection": CODEBASE.defines_edges,
                "from": [CODEBASE.files],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.calls_edges,
                "from": [CODEBASE.symbols],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.implements_edges,
                "from": [CODEBASE.symbols],
                "to": [CODEBASE.symbols],
            },
            {
                "collection": CODEBASE.imports_edges,
                "from": [CODEBASE.files],
                "to": [CODEBASE.files, CODEBASE.symbols],
            },
        ],
        "orphanCollections": [CODEBASE.chunks, CODEBASE.embeddings],
    });

    match db.writer().post("gharial", &body).await {
        Ok(_) => {
            info!(graph = CODEBASE_GRAPH, "created named graph");
        }
        Err(e) if e.kind() == ArangoErrorKind::Conflict => {
            debug!(graph = CODEBASE_GRAPH, "named graph already exists");
        }
        Err(e)
            if matches!(
                e.kind(),
                ArangoErrorKind::Forbidden
                    | ArangoErrorKind::NotFound
                    | ArangoErrorKind::Unavailable
            ) =>
        {
            // Non-fatal: edge collections work for AQL traversals without
            // a named graph wrapper. Forbidden/NotFound typically mean the
            // Metis proxy or RBAC blocks the gharial management endpoint;
            // Unavailable means the endpoint is temporarily down.
            warn!(
                graph = CODEBASE_GRAPH,
                error = %e,
                kind = ?e.kind(),
                "failed to create named graph (non-fatal — edges still work)"
            );
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e).context("failed to create named graph"));
        }
    }
    Ok(())
}

/// Ensure persistent indices exist on codebase document collections.
///
/// ArangoDB's `ensureIndex` is idempotent — if an index with the same
/// fields and type already exists, it returns the existing index.
async fn ensure_indices(db: &ArangoPool) -> Result<()> {
    let indices: &[(&str, &[&str])] = &[
        (CODEBASE.chunks, &["file_key"]),
        (CODEBASE.chunks, &["symbols[*]"]),
        (CODEBASE.embeddings, &["file_key"]),
        (CODEBASE.embeddings, &["chunk_key"]),
        (CODEBASE.symbols, &["file_key"]),
        (CODEBASE.symbols, &["kind"]),
    ];

    for (collection, fields) in indices {
        let path = format!("index?collection={collection}");
        let body = json!({
            "type": "persistent",
            "fields": fields,
        });
        db.writer()
            .post(&path, &body)
            .await
            .with_context(|| format!("failed to ensure index on {collection} {fields:?}"))?;
    }

    debug!("ensured {} persistent indices", indices.len());
    Ok(())
}

// ── File discovery ──────────────────────────────────────────────────────

/// Discover source files to ingest from a path.
///
/// If `path` is a file, returns just that file (if it matches the language
/// filter). If a directory, walks recursively, skipping common non-source
/// directories.
/// Normalize an `--unparsed-ext` allowlist: trim, strip a leading dot, lowercase.
///
/// Shared with `codebase drift` so the same flag value always produces the same
/// discovery set. This is not cosmetic: `discover_files` matches a file's
/// lowercased, dot-less extension against this set, so an entry of `.md` or
/// ` md ` silently matches nothing. If drift normalized differently from ingest,
/// files ingest *did* pick up would be absent from drift's disk set, get reported
/// as `stale`, and — piped into `codebase retire` — have their live nodes
/// deleted. Parity here has to be structural, not conventional.
pub(crate) fn normalize_unparsed_ext(unparsed_ext: &[String]) -> std::collections::HashSet<String> {
    unparsed_ext
        .iter()
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect()
}

/// Parse a `--language` override, accepting the word forms as well as extensions.
///
/// Shared with `codebase drift` for the same reason as
/// [`normalize_unparsed_ext`]: drift documents that its flags must match the
/// ingest invocation, so it has to accept exactly what ingest accepts. Parsing
/// via `Language::from_extension` alone would reject `rust`, `python`, `golang`,
/// and `cuda` — all of which ingest takes.
pub(crate) fn parse_language_arg(language: Option<&str>) -> Result<Option<Language>> {
    let Some(l) = language else { return Ok(None) };
    let lang = match l.to_lowercase().as_str() {
        "python" | "py" => Language::Python,
        "rust" | "rs" => Language::Rust,
        "c" | "cpp" | "c++" | "cuda" | "cu" => Language::Cpp,
        "go" | "golang" => Language::Go,
        other => {
            bail!("unsupported language: {other}. Supported: python, rust, go, c/c++/cuda")
        }
    };
    Ok(Some(lang))
}

/// Field on a `codebase_files` node naming the ingest root it was built from.
///
/// Read by `codebase drift` to scope its graph query. The root is also part of
/// the file identity, so distinct trees retain independent documents (#13).
pub(crate) const INGEST_ROOT_FIELD: &str = "ingest_root";

/// The base directory that ingest strips to form a file node's `rel_path`.
///
/// A file node's `_key` is `keys::scoped_file_key(namespace, rel_path)` where `rel_path` is the
/// path relative to this base — so anything comparing graph keys against the
/// working tree (e.g. `codebase drift`) MUST derive keys through this same
/// function, or every key mismatches and the comparison is meaningless.
/// Resolve the ingest root: it must exist, and it is canonicalized.
///
/// Canonical because everything downstream mixes two derivations of the same
/// tree. [`ingest_base_path`] canonicalizes the base, while `discover_files`
/// returned paths exactly as the operator typed them, so a relative argument
/// left the two in different spaces. [`rel_path_for`] then stripped an absolute
/// base off a relative path, failed, and fell back to the whole path as given.
///
/// Measured on `crates/hades-proto`, three files: an absolute argument keyed
/// them `build_rs`, `src_lib_rs`, `tests_proto_types_rs`, and a relative
/// argument keyed the same three files
/// `crates_hades-proto_build_rs` and so on. Six file nodes for three files,
/// each with its own chunks, symbols and embeddings, and both sets stamped with
/// the same `ingest_root`, so `codebase drift` could not tell them apart.
///
/// The rust-analyzer phase broke from the other end of the same cause. Its
/// `rust_abs_paths` were relative, so the crate-root walk in
/// `find_workspace_root` popped to an empty path (`Cargo.toml` existing in the
/// process cwd), and spawning a session with an empty working directory failed
/// with ENOENT. Every call and implements edge in the run was lost, which #164's
/// guard then correctly refused to accept silently.
pub(crate) fn resolve_ingest_root(path: &Path) -> Result<PathBuf> {
    if !path.exists() {
        bail!("path not found: {}", path.display());
    }
    path.canonicalize()
        .with_context(|| format!("failed to resolve ingest path: {}", path.display()))
}

pub(crate) fn ingest_base_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.parent()
            .map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

/// The `rel_path` a file node records: the path relative to the ingest base.
///
/// Single home for this derivation. The ingest loop and `codebase drift` both
/// call it, so drift cannot silently disagree with ingest about which file a key
/// refers to.
pub(crate) fn rel_path_for(base: &Path, file_path: &Path) -> String {
    file_path
        .strip_prefix(base)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string()
}

/// Compute the graph `_key` for a discovered file, relative to `base`.
pub(crate) fn file_key_for(base: &Path, file_path: &Path) -> String {
    keys::scoped_file_key(&base.to_string_lossy(), &rel_path_for(base, file_path))
}

/// Fail closed before replacing any legacy or conflicting file graph.
async fn preflight_file_identities(db: &ArangoPool, base: &Path, files: &[PathBuf]) -> Result<()> {
    let namespace = base.to_str().context("ingest root must be valid UTF-8")?;
    let legacy = hades_core::db::query::query(
        db,
        "FOR f IN @@files FILTER f.file_key_version != 2 LIMIT 1 RETURN f._key",
        Some(&json!({"@files": CODEBASE.files})),
        Some(1),
        false,
        ExecutionTarget::Reader,
    )
    .await?;
    if !legacy.results.is_empty() {
        bail!(
            "legacy code-file identities require explicit migration; ingest into an isolated new database and follow docs/code-file-identities.md; existing data was not purged"
        );
    }
    let mut identities = HashMap::new();
    for file in files {
        file.to_str()
            .context("source paths must be valid UTF-8; refusing lossy identity conversion")?;
        let relative = rel_path_for(base, file);
        let key = keys::scoped_file_key(namespace, &relative);
        if let Some(previous) = identities.insert(key.clone(), relative.clone())
            && previous != relative
        {
            bail!("file identity conflict between {previous} and {relative}; no files were purged");
        }
    }
    let entries: Vec<_> = identities
        .into_iter()
        .map(|(key, path)| json!({"key":key,"path":path}))
        .collect();
    for batch in entries.chunks(500) {
        let rows = hades_core::db::query::query(db,
            "FOR item IN @items LET f = DOCUMENT(@@files, item.key) RETURN f == null OR (f.file_key_version == 2 AND f.path == item.path AND f.ingest_root == @root)",
            Some(&json!({"@files":CODEBASE.files,"items":batch,"root":namespace})), Some(500), false, ExecutionTarget::Reader).await?;
        if rows.results.len() != batch.len()
            || rows.results.iter().any(|row| row.as_bool() != Some(true))
        {
            bail!("file identity conflict in ingest batch; refusing to purge existing data");
        }
    }
    Ok(())
}

async fn verify_file_identity(
    db: &ArangoPool,
    namespace: &str,
    relative: &str,
    key: &str,
) -> Result<()> {
    let rows = hades_core::db::query::query(db,
        "LET f = DOCUMENT(@@files, @key) RETURN f == null OR (f.file_key_version == 2 AND f.path == @path AND f.ingest_root == @root)",
        Some(&json!({"@files": CODEBASE.files, "key": key, "path": relative, "root": namespace})),
        Some(1), false, ExecutionTarget::Reader).await?;
    if rows.results.first().and_then(Value::as_bool) != Some(true) {
        bail!(
            "file identity conflict for {relative}; stored root/path/version do not match; refusing to purge existing data"
        );
    }
    Ok(())
}

/// A file under the ingest root that discovery deliberately did not pick up.
///
/// Recorded rather than dropped so `codebase drift` can report a third bucket.
/// A file that is neither ingested nor reportable is a silent hole: the pair
/// (ingest, drift) otherwise reports a clean sweep over a partially-covered
/// tree, which is a false green rather than a visible gap (#183).
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct UnhandledFile {
    pub path: String,
    pub reason: &'static str,
}

/// The outcome of a discovery walk: what will be ingested, and what will not.
pub(crate) struct Discovery {
    pub files: Vec<PathBuf>,
    pub unhandled: Vec<UnhandledFile>,
}

/// The first line of a file, if it is readable as UTF-8.
///
/// Used only for shebang sniffing, so a binary (invalid UTF-8) simply yields
/// `None` and stays out of discovery.
///
/// The read is capped: `read_line` alone allocates until the first newline, so
/// an extensionless blob with none near the start (a compiled `a.out`, a
/// checked-in artifact, a minified single-line bundle) would be pulled entirely
/// into memory during a directory walk that previously never opened it.
/// A shebang lives in the first handful of bytes or not at all.
fn first_line(path: &Path) -> Option<String> {
    use std::io::{BufRead, BufReader, Read};
    /// Longest plausible shebang line; anything beyond cannot be one.
    const SHEBANG_PROBE_BYTES: u64 = 256;
    let file = std::fs::File::open(path).ok()?;
    let mut line = String::new();
    BufReader::new(file.take(SHEBANG_PROBE_BYTES))
        .read_line(&mut line)
        .ok()?;
    Some(line)
}

/// The analyzer language implied by a file's shebang, plus whether it had one.
///
/// Only consulted for extensionless files: an extension is cheaper and more
/// reliable when present.
/// Only extensionless files are sniffed. An extension is cheaper and more
/// reliable when present, and — critically — the ingest loop applies the same
/// restriction, so admitting an extension-bearing script here would let it pass
/// discovery and then fail with "cannot detect language" instead of the
/// actionable "unsupported file type … use --language or --unparsed-ext".
pub(crate) fn shebang_of(path: &Path) -> Option<(bool, Option<Language>)> {
    if path.extension().is_some() {
        return None;
    }
    let line = first_line(path)?;
    let trimmed = line.trim_end();
    if !Language::is_shebang(trimmed) {
        return None;
    }
    Some((true, Language::from_shebang(trimmed)))
}

/// Every file under `root`, classified by which pipeline claims it.
///
/// One walk, one routing decision per file, so a unified ingest can report what
/// happened to everything rather than leaving a caller to infer it from two
/// separate commands' counts. Honors the same ignore rules as code discovery:
/// `.gitignore`, `.ignore`, `.hadesignore` and [`SKIP_DIRS`].
///
/// Extensionless files with a shebang are classified as code, matching
/// `discover_files_detailed`, which includes them so they are visible to ingest
/// and drift instead of disappearing (#183).
///
/// `unparsed_set` are extensions the operator asked to embed without a parser.
/// They count as code here because that is the phase which ingests them, and a
/// report that called them unrouted while the code phase was storing them would
/// be wrong in the direction operators trust.
pub(crate) fn discover_by_route(
    root: &Path,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<RouteDiscovery> {
    let mut found = RouteDiscovery::default();

    let unparsed = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| unparsed_set.contains(&e.to_lowercase()))
    };

    if root.is_file() {
        match ingest_routing::route_for(root) {
            Route::Code(_) => found.code.push(root.to_path_buf()),
            Route::Document => found.documents.push(root.to_path_buf()),
            Route::Unrouted => {
                if unparsed(root) || shebang_of(root).is_some() {
                    found.code.push(root.to_path_buf());
                } else {
                    found.unrouted.push(UnhandledFile {
                        path: root.to_string_lossy().to_string(),
                        reason: "no handler for extension",
                    });
                }
            }
        }
        return Ok(found);
    }

    let walker = WalkBuilder::new(root)
        .follow_links(false)
        .add_custom_ignore_filename(".hadesignore")
        .filter_entry(|entry| {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && let Some(name) = entry.file_name().to_str()
            {
                return !SKIP_DIRS.contains(&name);
            }
            true
        })
        .build();

    for entry in walker {
        let entry = entry.context("error walking directory")?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        match ingest_routing::route_for(path) {
            Route::Code(_) => found.code.push(path.to_path_buf()),
            Route::Document => found.documents.push(path.to_path_buf()),
            Route::Unrouted => {
                if unparsed(path) || (path.extension().is_none() && shebang_of(path).is_some()) {
                    found.code.push(path.to_path_buf());
                } else {
                    found.unrouted.push(UnhandledFile {
                        path: path.to_string_lossy().to_string(),
                        reason: if path.extension().is_some() {
                            "no handler for extension"
                        } else {
                            "no extension and no shebang"
                        },
                    });
                }
            }
        }
    }

    found.code.sort();
    found.documents.sort();
    found.unrouted.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(found)
}

/// What one walk found, split by route.
#[derive(Default)]
pub(crate) struct RouteDiscovery {
    pub(crate) code: Vec<PathBuf>,
    pub(crate) documents: Vec<PathBuf>,
    pub(crate) unrouted: Vec<UnhandledFile>,
}

pub(crate) fn discover_files(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<Vec<PathBuf>> {
    Ok(discover_files_detailed(path, lang_override, unparsed_set)?.files)
}

/// Walk the tree, returning both the files ingest will process and the ones it
/// will not, each with a reason.
pub(crate) fn discover_files_detailed(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<Discovery> {
    // Whether a path's (lowercased) extension is in the unparsed allowlist.
    let ext_allowed = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| unparsed_set.contains(&e.to_lowercase()))
    };

    if path.is_file() {
        let path_str = path.to_string_lossy();
        if lang_override.is_some()
            || Language::from_path(&path_str).is_some()
            || ext_allowed(path)
            || shebang_of(path).is_some()
        {
            return Ok(Discovery {
                files: vec![path.to_path_buf()],
                unhandled: Vec::new(),
            });
        }
        bail!(
            "unsupported file type: {}. Use --language or --unparsed-ext to override.",
            path.display()
        );
    }

    let mut files = Vec::new();
    let mut unhandled = Vec::new();
    let walker = WalkBuilder::new(path)
        .follow_links(false)
        .add_custom_ignore_filename(".hadesignore")
        .filter_entry(|entry| {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && let Some(name) = entry.file_name().to_str()
            {
                return !SKIP_DIRS.contains(&name);
            }
            true
        })
        .build();

    for entry in walker {
        let entry = entry.context("error walking directory")?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let entry_path = entry.path();
        let path_str = entry_path.to_string_lossy();
        let has_ext = entry_path.extension().is_some();

        let (include, reason) = if Language::from_path(&path_str).is_some() {
            // File has a recognized source extension — always include.
            (true, "")
        } else if ext_allowed(entry_path) {
            // Extension is in the unparsed allowlist (e.g. cu,cuh) — include
            // for the parser-free embedding fallback (#121).
            (true, "")
        } else if lang_override.is_some() && !has_ext {
            // Language override active: include extensionless files only
            // (skip .md, .json, images, etc.).
            (true, "")
        } else if !has_ext && shebang_of(entry_path).is_some() {
            // Extensionless script identified by its shebang. `--unparsed-ext`
            // is extension-keyed and so can never name these (#183); without
            // this branch they are invisible to both ingest and drift.
            (true, "")
        } else if has_ext {
            (false, "no handler for extension")
        } else {
            (false, "no extension and no shebang")
        };

        if include {
            files.push(entry_path.to_path_buf());
        } else {
            unhandled.push(UnhandledFile {
                path: path_str.to_string(),
                reason,
            });
        }
    }

    files.sort();
    unhandled.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Discovery { files, unhandled })
}

// ── Per-file ingest ─────────────────────────────────────────────────────

/// Ingest a single source file: analyze → chunk → embed → store.
#[allow(clippy::too_many_arguments)]
async fn ingest_file(
    db: &ArangoPool,
    embedder: Option<&EmbeddingClient>,
    config: &HadesConfig,
    file_path: &Path,
    rel_path: &str,
    lang_override: Option<Language>,
    imports: &mut ImportContext,
    compile_commands: Option<&Path>,
    force: bool,
    allow_analysis_downgrade: bool,
    gopls_scheduled: bool,
    window_chars: usize,
    namespace: &str,
) -> Result<FileResult> {
    // Read source.
    let fkey = keys::scoped_file_key(namespace, rel_path);
    let expected_revision = super::codebase_persist::revision(db.writer(), &fkey).await?;
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;

    // Detect language.
    let lang = lang_override
        .or_else(|| Language::from_path(rel_path))
        .ok_or_else(|| anyhow::anyhow!("cannot detect language for {rel_path}"))?;

    let reenriched_this_run = reenrichment_hatch(file_path, rel_path, gopls_scheduled);

    // Analyze.
    // Pass the absolute path to the analyzer so libclang resolves C/C++ includes
    // consistently regardless of cwd. Keys and logging still use rel_path.
    let options = AnalysisOptions {
        compilation_database: compile_commands.map(Path::to_path_buf),
    };
    let mut analysis =
        match code::analyze_with_fallback(&source, lang, &file_path.to_string_lossy(), &options) {
            AnalyzerOutcome::Success(analysis) => analysis,
            AnalyzerOutcome::Failed { analyzer, reason } => {
                warn!(
                    path = rel_path,
                    analyzer,
                    reason,
                    "semantic and structural analysis unavailable; using raw text fallback"
                );
                return ingest_unparsed_file(
                    db,
                    embedder,
                    config,
                    file_path,
                    rel_path,
                    Some(lang.name()),
                    &reason,
                    force,
                    allow_analysis_downgrade,
                    namespace,
                )
                .await;
            }
        };
    info!(
        path = rel_path,
        analyzer = analysis.analyzer,
        analysis_tier = %analysis.analysis_tier,
        fallback_reason = analysis.fallback_reason.as_deref(),
        "selected code analyzer"
    );

    // Check for incremental skip via content_hash.
    //
    // **Not `symbol_hash`, which is name-only and cannot see a comment edit.**
    // That was the gate until #7: `5c84d44` in one corpus changed two files in
    // doc comments alone, touching no symbol-declaring line, so the names hashed
    // identically, both files were reported `skipped`, and their stored chunks
    // kept text the corpus had retired -- served by `db_query` as current while
    // the symbol half, which rust-analyzer re-reads every run, moved on. The two
    // halves of one file disagreed and only one of them is what search reads.
    //
    // The two hashes are not independent, so this is a replacement and not an
    // addition: a byte-identical file has identical symbols, so `content_hash`
    // unchanged implies `symbol_hash` unchanged. It is the strictly stronger
    // predicate, and the weaker one was letting real changes through.
    //
    // Nor can the decision be split per artifact -- chunks on content, symbols
    // on names. `symbol_key` hashes the line number, so a comment that adds
    // eight lines moves every later symbol's key, and chunk documents reference
    // those keys in `overlapping_symbols`. Refreshing chunks while leaving
    // symbols would point fresh chunks at keys that no longer exist. Within one
    // file the two move together or not at all.
    //
    // `symbol_hash` keeps its real job, which is cross-file: whether a file's
    // *dependents* need their import and call edges re-resolved. See the
    // `dangling_inbound` warning in the caller, which is about exactly that.
    //
    // Only skip if the content is unchanged AND embeddings aren't needed (either
    // already present or no embedder available to backfill). `--force` bypasses
    // this entirely, re-ingesting in place (#145) — the per-file purge below
    // touches only this file's own symbols and outbound edges, so inbound
    // authored bridge edges are preserved (unlike a cascading `db purge`).
    verify_file_identity(db, namespace, rel_path, &fkey).await?;
    if preserve_higher_fidelity(
        db,
        &fkey,
        analysis.analysis_tier,
        allow_analysis_downgrade,
        reenriched_this_run,
    )
    .await?
    {
        warn!(
            path = rel_path,
            incoming_tier = %analysis.analysis_tier,
            "preserving higher-fidelity stored analysis"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: Some("higher-fidelity stored analysis preserved".to_string()),
            duration_ms: 0,
        });
    }
    let content_hash = hades_core::code::compute_content_hash(&source);
    if !force && check_unchanged(db, &fkey, &content_hash, embedder.is_some()).await? == Some(true)
    {
        debug!(
            path = rel_path,
            "unchanged (same content_hash, embeddings present), skipping"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang.name().to_string()),
            num_symbols: Some(analysis.symbols.len()),
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Snapshot prior identities before preparing a replacement. Purging happens
    // only inside the later atomic store, after analysis/embedding succeeds.
    // Symbol/edge inserts are overwrite-by-key only, so without this a renamed
    // or deleted symbol would leave an orphaned row that later inflates
    // `symbol_count` and dangles in the graph (#126). We only reach here when
    // the file actually changed (the unchanged-skip returned above), so an
    // unchanged file — which has no orphans — is never needlessly purged.
    // RA enrichment runs afterward and *augments* the freshly-written syn set.
    // Read the symbols about to be destroyed, so the rewrite can be paired with
    // its predecessor (#9). After the purge a key cannot be reversed into a
    // name, so this is the only moment both sides are knowable.
    let previous_symbols = existing_symbol_identities(db, &fkey).await?;

    // Chunk with AST-aligned chunking.
    let chunker = AstChunking::new(analysis.top_level_defs.clone());
    let chunks = chunker.chunk(&source);

    // Build file document (embedding_count populated after embed step below).
    let num_sym = analysis.symbols.len();
    let num_chk = chunks.len();

    // Build line→byte offset table for symbol-chunk interval intersection.
    let line_offsets = build_line_offsets(&source);

    // Build chunk documents with symbol context.
    let chunk_docs: Vec<Value> = chunks
        .iter()
        .map(|c| {
            // Find symbols whose span overlaps this chunk (interval intersection).
            let overlapping_symbols: Vec<String> = analysis
                .symbols
                .iter()
                .filter(|s| s.kind.is_primitive())
                .filter_map(|s| {
                    let sym_start = line_offsets
                        .get(s.start_line.saturating_sub(1))
                        .copied()
                        .unwrap_or(0);
                    let sym_end = line_offsets
                        .get(s.end_line)
                        .copied()
                        .unwrap_or(source.len());
                    if c.start_char < sym_end && sym_start < c.end_char {
                        Some(keys::symbol_key(&fkey, &s.qualified_name(), s.start_line))
                    } else {
                        None
                    }
                })
                .collect();

            let ckey = keys::chunk_key(&fkey, c.chunk_index);
            json!({
                "_key": ckey,
                "file_key": fkey,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "symbols": overlapping_symbols,
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
            })
        })
        .collect();

    // Build symbol documents (primitives only — imports and impl blocks are not vertices).
    let symbol_docs: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let qname = s.qualified_name();
            let skey = keys::symbol_key(&fkey, &qname, s.start_line);
            json!({
                "_key": skey,
                "file_key": fkey,
                "file_path": rel_path,
                "name": s.name,
                "qualified_name": qname,
                "kind": s.kind.universal_kind().unwrap(),
                "lang_kind": s.kind.lang_kind(),
                "start_line": s.start_line,
                "end_line": s.end_line,
                "metadata": s.metadata,
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
            })
        })
        .collect();

    // Build defines edges (file → symbol) for primitives only.
    let define_edges: Vec<Value> = analysis
        .symbols
        .iter()
        .filter(|s| s.kind.is_primitive())
        .map(|s| {
            let skey = keys::symbol_key(&fkey, &s.qualified_name(), s.start_line);
            let edge_key = keys::edge_key(&fkey, "defines", &skey);
            json!({
                "_key": edge_key,
                "_from": format!("{}/{}", CODEBASE.files, fkey),
                "_to": format!("{}/{}", CODEBASE.symbols, skey),
                "file_path": rel_path,
                "symbol_name": s.name,
                "analysis_tier": analysis.analysis_tier.as_str(),
                "analyzer": analysis.analyzer,
                "resolution": if analysis.analysis_tier == AnalysisTier::Semantic { "semantic" } else { "syntactic" },
            })
        })
        .collect();

    // Embed with LATE CHUNKING: encode each window of the file in one pass and
    // pool per AST boundary, so every chunk vector is conditioned on the code
    // around it. Embedding chunks independently (the previous behaviour) threw
    // that context away — the model computes token-level states either way, so
    // it cost nothing and gained nothing.
    //
    // Files larger than the model's context window are pre-chunked first: the
    // AST chunks are grouped into windows that fit, and each window is encoded
    // whole. Boundaries within a window keep their AST alignment, so symbol
    // intersection is unaffected. This is the escape hatch, not the norm.
    // Escape hatch, and what makes the two strategies comparable: with this
    // set, chunks are embedded independently as before. Useful if late
    // chunking ever misbehaves, and required to A/B the two on one corpus.
    // Read once for the process rather than once per ingested file.
    let late_chunking_disabled = *LATE_CHUNKING_DISABLED;

    let (embedding_docs, embedding_error): (Vec<Value>, Option<String>) = match embedder {
        Some(emb) if !chunks.is_empty() && late_chunking_disabled => {
            let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
            match emb
                .embed(&chunk_texts, "code", Some(config.embedding.batch.size))
                .await
            {
                Ok(r) => (
                    r.embeddings
                        .iter()
                        .enumerate()
                        .map(|(i, vec)| {
                            let ckey = keys::chunk_key(&fkey, i);
                            json!({
                                "_key": keys::embedding_key(&ckey),
                                "chunk_key": ckey,
                                "file_key": fkey,
                                "embedding": vec,
                                "model": r.model,
                                "model_hash": keys::model_hash(&r.model),
                                "dimension": r.dimension,
                            })
                        })
                        .collect::<Vec<Value>>(),
                    None,
                ),
                Err(e) => {
                    warn!(path = rel_path, error = %e, "embedding preparation failed; retaining committed graph");
                    (Vec::new(), Some(e.to_string()))
                }
            }
        }
        Some(emb) if !chunks.is_empty() => {
            // Window budget comes from the backend's reported ceiling, computed
            // once per run in `embed_window_chars`. A constant here was wrong on
            // both cards: see that function for the measurement.

            let mut windows: Vec<EmbedWindow> = Vec::new();
            let mut cur: Vec<(usize, usize)> = Vec::new();
            let mut cur_indices: Vec<usize> = Vec::new();
            let mut cur_start = 0usize;
            // Chunks too large to share a window with anything, including
            // themselves. `AstChunking` caps a chunk at 8,000 characters by
            // splitting at line boundaries, but `split_at_lines` emits one whole
            // line when its accumulator is empty, so a minified or generated file
            // with a single very long line produces a chunk of unbounded size.
            //
            // Sent whole to `embed_late_chunked`, such a chunk exceeds the
            // backend's ceiling, is refused, and takes the file's other windows
            // down with it, because that call returns on the first error. They go
            // through the plain path instead: one vector each with no surrounding
            // context, which is worse than late chunking and far better than the
            // file losing every vector it had.
            let mut oversized: Vec<usize> = Vec::new();
            for (i, c) in chunks.iter().enumerate() {
                if c.end_char.saturating_sub(c.start_char) > window_chars {
                    oversized.push(i);
                    continue;
                }
                if cur.is_empty() {
                    cur_start = c.start_char;
                }
                let would_span = c.end_char.saturating_sub(cur_start);
                if !cur.is_empty() && would_span > window_chars {
                    // Derived from the window's own boundaries rather than by
                    // indexing `first_index + cur.len()`: a skipped oversized
                    // chunk makes those two disagree.
                    let end = cur_start + cur.last().map(|(_, e)| *e).unwrap_or(0);
                    push_embed_window(
                        &mut windows,
                        &source,
                        cur_start,
                        end,
                        std::mem::take(&mut cur),
                        std::mem::take(&mut cur_indices),
                        rel_path,
                    );
                    cur_start = c.start_char;
                }
                // Boundaries are relative to the window, not the file, and are
                // byte offsets at this point. `push_embed_window` converts.
                cur.push((
                    c.start_char.saturating_sub(cur_start),
                    c.end_char.saturating_sub(cur_start),
                ));
                cur_indices.push(i);
            }
            if !cur.is_empty() {
                let end = cur_start + cur.last().map(|(_, e)| *e).unwrap_or(0);
                push_embed_window(
                    &mut windows,
                    &source,
                    cur_start,
                    end,
                    cur,
                    cur_indices,
                    rel_path,
                );
            }
            if windows.len() > 1 {
                debug!(
                    path = rel_path,
                    windows = windows.len(),
                    "file exceeds the context window, pre-chunked before late chunking"
                );
            }

            let texts: Vec<String> = windows.iter().map(|w| w.text.clone()).collect();
            let bounds: Vec<Vec<(usize, usize)>> =
                windows.iter().map(|w| w.boundaries.clone()).collect();

            // One vector per chunk, keyed by the chunk's file-order index, so a
            // window that fails costs its own chunks and not the file's.
            let mut by_index: std::collections::BTreeMap<usize, Vec<f32>> =
                std::collections::BTreeMap::new();
            let mut model = String::new();
            let mut dimension = 0u32;
            let mut first_error: Option<String> = None;

            let take = |result: hades_core::persephone::embedding::LateChunkEmbedResult,
                        offset: usize,
                        by_index: &mut std::collections::BTreeMap<usize, Vec<f32>>,
                        model: &mut String,
                        dimension: &mut u32| {
                *model = result.model.clone();
                *dimension = result.dimension;
                for (w, vecs) in result.per_input.iter().enumerate() {
                    let indices = &windows[offset + w].chunk_indices;
                    for v in vecs {
                        // Looked up rather than computed. A position the window
                        // never sent would otherwise mint an embedding under some
                        // other chunk's key, which reads as a healthy row.
                        match indices.get(v.chunk_index) {
                            Some(&i) => {
                                by_index.insert(i, v.embedding.clone());
                            }
                            None => warn!(
                                path = rel_path,
                                window = offset + w,
                                position = v.chunk_index,
                                boundaries = indices.len(),
                                "embedder returned a chunk position the window did not send"
                            ),
                        }
                    }
                }
            };

            if !windows.is_empty() {
                match emb.embed_late_chunked(&texts, "code", &bounds).await {
                    Ok(result) => take(result, 0, &mut by_index, &mut model, &mut dimension),
                    Err(e) => {
                        // The batched call returns on its first failing window and
                        // discards the vectors of every window that already
                        // succeeded, so a 500-window file that trips on window 400
                        // used to be stored with nothing. Retried one window at a
                        // time, a bad window costs its own chunks.
                        warn!(
                            path = rel_path,
                            error = %e,
                            windows = windows.len(),
                            "batched late-chunk embed failed, retrying window by window"
                        );
                        first_error = Some(e.to_string());
                        for (w, (text, bound)) in texts.iter().zip(bounds.iter()).enumerate() {
                            match emb
                                .embed_late_chunked(
                                    std::slice::from_ref(text),
                                    "code",
                                    std::slice::from_ref(bound),
                                )
                                .await
                            {
                                Ok(result) => {
                                    take(result, w, &mut by_index, &mut model, &mut dimension)
                                }
                                Err(e) => warn!(
                                    path = rel_path,
                                    window = w,
                                    error = %e,
                                    "window failed on retry, its chunks keep no vector"
                                ),
                            }
                        }
                    }
                }
            }

            // Chunks no window could hold, embedded blind rather than dropped.
            for &i in &oversized {
                match emb
                    .embed(std::slice::from_ref(&chunks[i].text), "code", None)
                    .await
                {
                    Ok(r) => {
                        if let Some(v) = r.embeddings.into_iter().next() {
                            if model.is_empty() {
                                model = r.model.clone();
                                dimension = r.dimension;
                            }
                            by_index.insert(i, v);
                        }
                    }
                    Err(e) => {
                        warn!(
                            path = rel_path,
                            chunk = i,
                            error = %e,
                            "oversized chunk failed the plain embed path too"
                        );
                        if first_error.is_none() {
                            first_error = Some(e.to_string());
                        }
                    }
                }
            }

            if by_index.len() != chunks.len() {
                warn!(
                    path = rel_path,
                    chunks = chunks.len(),
                    vectors = by_index.len(),
                    "fewer vectors than chunks, storing what was produced"
                );
                if first_error.is_none() {
                    first_error = Some(format!(
                        "{} chunks produced {} vectors",
                        chunks.len(),
                        by_index.len()
                    ));
                }
            }

            let docs = by_index
                .into_iter()
                .map(|(i, vec)| {
                    let ckey = keys::chunk_key(&fkey, i);
                    json!({
                        "_key": keys::embedding_key(&ckey),
                        "chunk_key": ckey,
                        "file_key": fkey,
                        "embedding": vec,
                        "model": model,
                        "model_hash": keys::model_hash(&model),
                        "dimension": dimension,
                    })
                })
                .collect::<Vec<Value>>();
            (docs, first_error)
        }
        _ => (Vec::new(), None),
    };
    if let Some(error) = &embedding_error {
        return Ok(FileResult {
            path: rel_path.to_owned(),
            success: false,
            language: Some(lang.name().to_owned()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: Some(error.clone()),
            skipped: Some(false),
            error: Some("embedding preparation failed; committed file graph retained".into()),
            duration_ms: 0,
        });
    }
    let num_embeddings_written = embedding_docs.len();

    // Build file document (after embedding so we can record embedding_count).
    // symbol_count reflects primitives only (no imports, no impl blocks), and is
    // counted from the DEDUPLICATED stored set (distinct `_key`) rather than the
    // pre-dedup primitive list — the upsert collapses any same-keyed symbols, so
    // counting the input would over-report and break `symbol_count_consistency`
    // (#113). With qualified-name keying collisions should not occur, but this
    // keeps the denorm honest regardless.
    // Pair the old keys with the new ones, for symbols that moved without being
    // renamed. Recorded on the shared context and applied after every file has
    // been written, because a dependent may itself be rewritten later in the run.
    let symbol_key_remap = pair_moved_symbol_keys(&previous_symbols, &symbol_docs);

    let primitive_count = symbol_docs
        .iter()
        .filter_map(|d| d["_key"].as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();
    let file_doc = json!({
        "_key": fkey,
        "file_key_version": 2,
        "ingest_root": namespace,
        "path": rel_path,
        "kind": "file",
        "language": lang.name(),
        "metrics": analysis.metrics,
        "symbol_hash": analysis.symbol_hash,
        // Full-source digest. **This is what gates incremental re-ingest**
        // (#7) and what `codebase drift` compares. `symbol_hash` above is
        // deliberately name-only (see compute_symbol_hash), so a rewritten
        // body, changed signature, or edited comment leaves it identical; it
        // gated the skip until #7 and let exactly those edits through, leaving
        // stale chunks that semantic search served as current. `symbol_hash` is
        // still stored, for the cross-file question of whether dependents need
        // re-resolution (#183).
        "content_hash": content_hash,
        "symbol_count": primitive_count,
        "relationships_pending": true,
        "chunk_count": num_chk,
        "embedding_count": num_embeddings_written,
        "total_lines": analysis.metrics.total_lines,
        "status": "PROCESSED",
        "analysis_tier": analysis.analysis_tier.as_str(),
        "analyzer": analysis.analyzer,
        "fallback_reason": analysis.fallback_reason,
        "ingested_at": chrono::Utc::now().to_rfc3339(),
    });

    let remapped_symbols = symbol_key_remap.len();
    let stored = super::codebase_persist::Replacement {
        key: fkey.clone(),
        expected_revision,
        chunks: chunk_docs,
        symbols: symbol_docs,
        embeddings: embedding_docs,
        defines: define_edges,
        file: file_doc,
        purge_symbols: true,
        merge_file: false,
        symbol_remap: symbol_key_remap,
    }
    .store(db)
    .await
    .context("failed to atomically replace file graph")?;
    imports.repointed_edges += stored.moved_edges;
    imports
        .committed_revisions
        .insert(fkey.clone(), stored.revision);
    imports.remapped_symbols += remapped_symbols;

    // Collect Python import symbols for later edge resolution.
    if lang == Language::Python {
        let py_import_syms: Vec<Symbol> = analysis
            .symbols
            .iter()
            .filter(|s| s.kind == hades_core::code::SymbolKind::Import)
            .cloned()
            .collect();
        if !py_import_syms.is_empty() {
            imports
                .python_imports
                .insert(rel_path.to_string(), py_import_syms);
        }
    }

    // Collect Rust use-paths for later import edge resolution.
    // Symbol transfer into the index is deferred until after all uses of analysis.symbols.
    if lang == Language::Rust {
        let use_paths = rust_imports::collect_use_paths(&analysis.symbols);
        if !use_paths.is_empty() {
            imports.rust_imports.insert(rel_path.to_string(), use_paths);
        }
    }

    // Transfer symbols into relationship indexes.
    if analysis.analysis_tier == AnalysisTier::Structural {
        imports
            .structural_file_symbols
            .insert(rel_path.to_string(), analysis.symbols.clone());
    }
    match lang {
        Language::Rust if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .rust_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Python if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .python_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Cpp if uses_semantic_relationship_resolver(lang, analysis.analysis_tier) => {
            imports
                .cpp_file_symbols
                .insert(rel_path.to_string(), std::mem::take(&mut analysis.symbols));
        }
        Language::Cpp => {}
        Language::Go => {}
        _ => {}
    }

    info!(
        path = rel_path,
        language = lang.name(),
        symbols = num_sym,
        chunks = num_chk,
        embeddings = num_embeddings_written,
        "ingested"
    );

    Ok(FileResult {
        path: rel_path.to_string(),
        success: true,
        language: Some(lang.name().to_string()),
        num_symbols: Some(num_sym),
        num_chunks: Some(num_chk),
        num_embeddings: Some(num_embeddings_written),
        embedding_error,
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

fn uses_semantic_relationship_resolver(language: Language, tier: AnalysisTier) -> bool {
    matches!(language, Language::Rust | Language::Python | Language::Cpp)
        && tier == AnalysisTier::Semantic
}

// ── Unparsed-language fallback (#121) ────────────────────────────────────

/// Map an unparsed file extension to a language label for the file node.
fn unparsed_language_label(rel_path: &str) -> &'static str {
    let ext = Path::new(rel_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    match ext.as_deref() {
        Some("cu") | Some("cuh") => "cuda",
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hh") | Some("hxx") => "cpp",
        // `.h` is shared by C and C++; labelled "c" as an intentional
        // simplification. The label is for human display / RGCN features only,
        // not parsing, so the ambiguity is harmless here.
        Some("c") | Some("h") => "c",
        Some("go") => "go",
        _ => "other",
    }
}

/// Ingest a file whose language has no parser: size-chunk the raw text, embed
/// the chunks as node features, and attach them to the file node — WITHOUT
/// symbol/edge extraction. The file node is *merged* (existing fields
/// preserved), not overwritten, so a pre-existing node's metadata survives
/// (e.g. an externally-created stub's `note`/`source`). See #121.
#[allow(clippy::too_many_arguments)]
async fn ingest_unparsed_file(
    db: &ArangoPool,
    embedder: Option<&EmbeddingClient>,
    config: &HadesConfig,
    file_path: &Path,
    rel_path: &str,
    language_label: Option<&str>,
    fallback_reason: &str,
    force: bool,
    allow_analysis_downgrade: bool,
    namespace: &str,
) -> Result<FileResult> {
    let fkey = keys::scoped_file_key(namespace, rel_path);
    let expected_revision = super::codebase_persist::revision(db.writer(), &fkey).await?;
    let source = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;
    verify_file_identity(db, namespace, rel_path, &fkey).await?;
    let lang_label = language_label.unwrap_or_else(|| unparsed_language_label(rel_path));

    if preserve_higher_fidelity(
        db,
        &fkey,
        AnalysisTier::Text,
        allow_analysis_downgrade,
        false,
    )
    .await?
    {
        warn!(
            path = rel_path,
            fallback_reason, "raw fallback skipped to preserve higher-fidelity stored analysis"
        );
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang_label.to_string()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: Some("higher-fidelity stored analysis preserved".to_string()),
            duration_ms: 0,
        });
    }
    let content_hash = code::compute_content_hash(&source);
    if !force && check_unchanged(db, &fkey, &content_hash, embedder.is_some()).await? == Some(true)
    {
        debug!(path = rel_path, "unchanged raw text, skipping");
        return Ok(FileResult {
            path: rel_path.to_string(),
            success: true,
            language: Some(lang_label.to_string()),
            num_symbols: Some(0),
            num_chunks: None,
            num_embeddings: None,
            embedding_error: None,
            skipped: Some(true),
            error: None,
            duration_ms: 0,
        });
    }

    // Parser-free chunking: empty defs => whole file, split at line boundaries
    // to stay under the max chunk size.
    let chunker = AstChunking::new(Vec::new());
    let chunks = chunker.chunk(&source);
    let num_chk = chunks.len();

    // Chunk documents — no symbol overlap (unparsed files have no symbols).
    let chunk_docs: Vec<Value> = chunks
        .iter()
        .map(|c| {
            let ckey = keys::chunk_key(&fkey, c.chunk_index);
            json!({
                "_key": ckey,
                "file_key": fkey,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "text": c.text,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "symbols": [],
                "analysis_tier": "text",
                "analyzer": "raw-text",
            })
        })
        .collect();

    // Embed chunks (skipped if embedder unavailable).
    //
    // NOT the same path as parsed files any more. The parsed path encodes whole
    // windows and pools per AST boundary, so each vector carries its file's
    // context. This one still sends chunks as separate inputs, so every chunk
    // is encoded blind to the rest of its document.
    //
    // That leaves markdown and every unsupported language on the old
    // behaviour, which includes the specs and PRDs the late-chunking work was
    // undertaken for. These chunks have offsets too, so the same treatment
    // applies. Not done here because it is a second change and this one is
    // already under review.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let (embedding_docs, embedding_error): (Vec<Value>, Option<String>) = match embedder {
        Some(emb) if !chunk_texts.is_empty() => {
            match emb
                .embed(&chunk_texts, "code", Some(config.embedding.batch.size))
                .await
            {
                Ok(embed_result) => {
                    let docs = embed_result
                        .embeddings
                        .iter()
                        .enumerate()
                        .map(|(i, vec)| {
                            let ckey = keys::chunk_key(&fkey, i);
                            let ekey = keys::embedding_key(&ckey);
                            json!({
                                "_key": ekey,
                                "chunk_key": ckey,
                                "file_key": fkey,
                                "embedding": vec,
                                "model": embed_result.model,
                                "model_hash": keys::model_hash(&embed_result.model),
                                "dimension": embed_result.dimension,
                            })
                        })
                        .collect::<Vec<Value>>();
                    (docs, None)
                }
                Err(e) => {
                    warn!(path = rel_path, error = %e, "embedding preparation failed; retaining committed graph");
                    (Vec::new(), Some(e.to_string()))
                }
            }
        }
        _ => (Vec::new(), None),
    };
    if let Some(error) = &embedding_error {
        return Ok(FileResult {
            path: rel_path.to_owned(),
            success: false,
            language: Some(lang_label.to_owned()),
            num_symbols: None,
            num_chunks: None,
            num_embeddings: None,
            embedding_error: Some(error.clone()),
            skipped: Some(false),
            error: Some("embedding preparation failed; committed file graph retained".into()),
            duration_ms: 0,
        });
    }
    let num_embeddings_written = embedding_docs.len();

    // Merge the file node — preserve any pre-existing fields, set only ours,
    // create if absent.
    let total_lines = source.lines().count();
    let fields = json!({
        "file_key_version": 2,
        "ingest_root": namespace,
        "path": rel_path,
        "rel_path": rel_path,
        "kind": "file",
        "language": lang_label,
        // The parser-free path has no symbols, so its change-detection digest is
        // already the full-source hash. Recorded under both names so drift has a
        // single uniform column across parsed and unparsed files.
        "symbol_hash": content_hash,
        "content_hash": content_hash,
        "symbol_count": 0,
        "chunk_count": num_chk,
        "embedding_count": num_embeddings_written,
        "total_lines": total_lines,
        "status": "PROCESSED",
        "analysis_tier": "text",
        "analyzer": "raw-text",
        "fallback_reason": fallback_reason,
        "ingested_at": chrono::Utc::now().to_rfc3339(),
    });
    super::codebase_persist::Replacement {
        key: fkey.clone(),
        expected_revision,
        chunks: chunk_docs,
        symbols: Vec::new(),
        embeddings: embedding_docs,
        defines: Vec::new(),
        file: fields,
        purge_symbols: allow_analysis_downgrade,
        merge_file: true,
        symbol_remap: Vec::new(),
    }
    .store(db)
    .await
    .context("failed to atomically replace fallback file graph")?;

    info!(
        path = rel_path,
        language = lang_label,
        chunks = num_chk,
        embeddings = num_embeddings_written,
        "ingested (unparsed)"
    );

    Ok(FileResult {
        path: rel_path.to_string(),
        success: true,
        language: Some(lang_label.to_string()),
        num_symbols: Some(0),
        num_chunks: Some(num_chk),
        num_embeddings: Some(num_embeddings_written),
        embedding_error,
        skipped: None,
        error: None,
        duration_ms: 0,
    })
}

/// Merge-write a `codebase_files` node: PATCH (preserving existing fields) when
/// it exists, otherwise insert. Lets the unparsed fallback attach to a
/// pre-existing file node without clobbering its metadata (#121).
#[cfg(test)]
async fn upsert_merge_file_node(db: &ArangoPool, fkey: &str, fields: Value) -> Result<()> {
    match crud::update_document(db, CODEBASE.files, fkey, &fields).await {
        Ok(_) => Ok(()),
        Err(e) if e.is_not_found() => {
            let mut doc = fields;
            doc["_key"] = json!(fkey);
            crud::insert_documents(db, CODEBASE.files, &[doc], true)
                .await
                .context("failed to insert file document")?;
            Ok(())
        }
        Err(e) => Err(anyhow::Error::new(e).context("failed to merge file document")),
    }
}

// ── Line-offset table ─────────────────────────────────────────────────

/// Build a byte-offset table for each line in `source`.
///
/// `offsets[i]` is the byte position where line `i` starts (0-based line
/// numbering). An extra sentinel entry for `offsets[line_count]` equals
/// `source.len()`, so callers can use `offsets[end_line]` to get the byte
/// position just past the last line of a span without bounds checks.
fn build_line_offsets(source: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets.push(source.len());
    offsets
}

// ── Stale embedding cleanup ────────────────────────────────────────────

/// Remove existing embedding documents for a file.
///
/// Called before (re-)embedding to ensure stale vectors from a previous
/// run don't linger when the embedder is unavailable or fails.
#[cfg(test)]
async fn delete_file_embeddings(db: &ArangoPool, file_key: &str) {
    if let Err(e) = hades_core::db::query::remove_docs_by_fields(
        db,
        CODEBASE.embeddings,
        &["file_key"],
        file_key,
    )
    .await
    {
        debug!(file_key, error = %e, "failed to clean up old embeddings (non-fatal)");
    }
}

/// Delete all chunk documents for a file.
///
/// Called before re-chunking on **both** the parsed and unparsed paths so that a
/// re-ingest which produces fewer chunks leaves no orphaned high-index chunk docs
/// behind (overwrite-by-key only updates the chunks that still exist). The parsed
/// path was missing this call until #159.
#[cfg(test)]
async fn delete_file_chunks(db: &ArangoPool, file_key: &str) {
    if let Err(e) =
        hades_core::db::query::remove_docs_by_fields(db, CODEBASE.chunks, &["file_key"], file_key)
            .await
    {
        debug!(file_key, error = %e, "failed to clean up old chunks (non-fatal)");
    }
}

/// Count inbound edges left dangling by this run's rebuilds. **Read-only.**
///
/// Deliberately reports rather than deletes. An inbound edge belongs to a file
/// this run may not have touched, and it encodes a real relationship: `b.py`
/// imports something from `a.py`. When a rebuild of `a.py` renames the target
/// symbol, deleting that edge destroys the only record that `b.py` depends on
/// `a.py` — and nothing re-derives it, because `b.py` is itself unchanged and
/// every later ingest skips it. The graph would then pass `validate` and `drift`
/// while silently missing a true relation, which is the same class of false
/// green this change set exists to remove. Re-ingesting the dependent (or an
/// explicit `codebase prune-orphans`) is the honest repair, so the count is
/// surfaced in the JSON summary and the operator decides.
///
/// Scoped two ways so the number means something: the target key must carry the
/// `{file_key}__` prefix of a file this run actually rewrote, AND the target
/// must genuinely not resolve — a symbol recreated by enrichment is not counted.
///
/// Note: symbols of files whose key exceeds the 254-byte budget carry a
/// *truncated* file_key prefix and fall outside this filter; `prune-orphans`
/// remains the global backstop.
async fn count_dangling_inbound(db: &ArangoPool, file_keys: &[String]) -> u64 {
    if file_keys.is_empty() {
        return 0;
    }
    let mut dangling = 0u64;
    for edges in [
        CODEBASE.imports_edges,
        CODEBASE.calls_edges,
        CODEBASE.implements_edges,
    ] {
        // Prefix test first: it is a cheap string comparison that eliminates
        // almost every edge, whereas DOCUMENT() is an unindexable per-edge
        // lookup. Ordering it last made this a full scan of all three edge
        // collections on every run.
        let aql = "\
            LET prefixes = (FOR fk IN @keys RETURN CONCAT(@symbols_name, '/', fk, '__')) \
            RETURN LENGTH( \
                FOR e IN @@edges \
                    FILTER LENGTH(FOR p IN prefixes FILTER STARTS_WITH(e._to, p) LIMIT 1 RETURN 1) > 0 \
                      AND DOCUMENT(e._to) == null \
                    RETURN 1)";
        let bind = json!({
            "@edges": edges,
            "symbols_name": CODEBASE.symbols,
            "keys": file_keys,
        });
        match hades_core::db::query::query(
            db,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        {
            Ok(rows) => {
                dangling += rows
                    .results
                    .first()
                    .and_then(|v| v.as_u64())
                    .unwrap_or_default();
            }
            Err(e) => {
                warn!(collection = edges, error = %e, "failed to check for dangling inbound edges after re-ingest (non-fatal)");
            }
        }
    }
    dangling
}

/// The symbols a file holds right now, as `(qualified_name, start_line, key)`.
///
/// Read immediately before a purge so the rewrite can be paired with what it
/// replaced (#9). Ordered by line, because the pairing is positional among
/// symbols that share a qualified name.
async fn existing_symbol_identities(
    db: &ArangoPool,
    file_key: &str,
) -> Result<Vec<(String, u64, String)>> {
    let aql = "FOR s IN @@symbols FILTER s.file_key == @key \
               SORT s.start_line, s._key \
               RETURN [s.qualified_name, s.start_line, s._key]";
    let bind = json!({ "@symbols": CODEBASE.symbols, "key": file_key });
    let result =
        hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer)
            .await?;
    result
        .results
        .iter()
        .map(|row| {
            let values = row.as_array().context("invalid prior symbol identity")?;
            Ok((
                values
                    .first()
                    .and_then(Value::as_str)
                    .context("missing prior symbol name")?
                    .to_owned(),
                values
                    .get(1)
                    .and_then(Value::as_u64)
                    .context("missing prior symbol line")?,
                values
                    .get(2)
                    .and_then(Value::as_str)
                    .context("missing prior symbol key")?
                    .to_owned(),
            ))
        })
        .collect()
}

/// Pair each old symbol key with the key that replaced it.
///
/// Matched on `(qualified_name, position among symbols sharing that name)`,
/// which is stable when a file's text moves but its symbol set does not -- the
/// case #9 is about. Names are not unique within a file (618 colliding groups in
/// one real corpus), so the position is what disambiguates, exactly as the line
/// does in the key itself.
///
/// **A name whose count changed is deliberately not paired.** If a file gained
/// or lost one of three `Config::new`s, position no longer identifies the same
/// symbol, and a wrong pairing would silently re-point a dependency at the wrong
/// definition. An unpaired symbol leaves its inbound edges dangling, which is
/// true and visible, and `count_dangling_inbound` still reports it.
fn pair_moved_symbol_keys(
    previous: &[(String, u64, String)],
    written: &[Value],
) -> Vec<(String, String)> {
    if previous.is_empty() {
        return Vec::new();
    }
    let mut now: Vec<(String, u64, String)> = written
        .iter()
        .filter_map(|d| {
            Some((
                d["qualified_name"].as_str()?.to_string(),
                d["start_line"].as_u64().unwrap_or(0),
                d["_key"].as_str()?.to_string(),
            ))
        })
        .collect();
    now.sort_by(|a, b| (a.1, &a.2).cmp(&(b.1, &b.2)));
    // Two symbol documents can collide on one `_key`, which the upsert then
    // collapses -- the same reason `primitive_count` counts the deduplicated
    // set. Counting the pre-dedup docs here would make the after-count exceed
    // what the database holds, trip the count-changed guard, and silently drop a
    // remap that was correct.
    now.dedup_by(|a, b| a.2 == b.2);

    let mut by_name_before: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, _, key) in previous {
        by_name_before.entry(name.as_str()).or_default().push(key);
    }
    let mut by_name_after: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, _, key) in &now {
        by_name_after.entry(name.as_str()).or_default().push(key);
    }

    let mut pairs = Vec::new();
    for (name, before) in by_name_before {
        let Some(after) = by_name_after.get(name) else {
            continue; // the name is gone: a real removal, not a move
        };
        if before.len() != after.len() {
            continue; // the count changed: position no longer identifies
        }
        for (old, new) in before.iter().zip(after.iter()) {
            if old != new {
                pairs.push(((*old).to_string(), (*new).to_string()));
            }
        }
    }
    pairs
}

/// Re-point inbound edges from a symbol's old key onto its replacement.
///
/// Returns how many edges were moved.
///
/// **Written under the canonical key, not updated in place.** An edge's `_key`
/// is `edge_key(from, kind, to)`, so changing `_to` with an `UPDATE` leaves the
/// key describing the old target. That is not merely untidy: the rust-analyzer
/// and gopls phases re-resolve cross-file `calls` and `implements` edges for
/// *every* file in the run, skipped ones included, and write them under the
/// canonical new key. An in-place update therefore left two documents for one
/// relation -- the phase's correct one and this one, stale-keyed -- which
/// traversals and neighbour counts both double. Inserting at the canonical key
/// collapses with whatever the phase wrote, and removing the old document
/// leaves exactly one edge. Found by review of #11 before it merged.
///
/// **Every read completes before any write.** The remap can chain: two symbols
/// sharing a qualified name can move such that one's new key is another's old
/// key (`A -> B` and `B -> C` in the same file). Resolving edges against a
/// collection that is being written in the same query would drag an edge for the
/// first symbol onto the third position, silently attaching a dependency to a
/// definition nobody wrote. Reading first fixes each edge's destination from its
/// pre-move target, which is the only reading that means anything.
#[cfg(test)]
async fn repoint_inbound_edges(db: &ArangoPool, remap: &[(String, String)]) -> u64 {
    let remap = remap.to_vec();
    let collections = CODEBASE
        .all_collections()
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    hades_core::db::transaction::run(db, collections, move |client| async move {
        super::codebase_persist::remap_inbound(&client, &remap).await
    })
    .await
    .unwrap()
}

/// Purge a file's existing symbols and the **source-owned (outgoing) edges**
/// the file's own ingest will recreate, before re-writing. Symbol/edge inserts
/// are overwrite-by-key only, so without this a renamed/deleted symbol leaves
/// an orphaned row — which inflates `symbol_count` and dangles in the graph
/// (#126). This makes `codebase_symbols` authoritative on re-ingest.
///
/// Only edges with `_from` in this file (the file node for `defines`, or one of
/// its symbols for `calls`/`implements`/`imports`) are removed — those are
/// rebuilt by this file's ingest. **Incoming** edges (`_to` in this file) are
/// owned by *other* source files and are NOT touched here: deleting them would
/// drop valid edges that a skipped source file never rebuilds. Incoming edges
/// left dangling by a rename/delete are cleaned by `hades codebase prune`.
///
/// The `ids` list is snapshotted in-query from the current symbols plus the
/// file `_id`, so the edge filter is consistent even under concurrent inserts.
#[cfg(test)]
async fn purge_file_symbols_and_edges(db: &ArangoPool, file_key: &str) {
    let aql = "\
        LET ids = APPEND( \
            (FOR s IN @@symbols FILTER s.file_key == @key RETURN s._id), \
            [CONCAT(@files_name, '/', @key)]) \
        LET syms = (FOR d IN @@symbols FILTER d.file_key == @key REMOVE d IN @@symbols RETURN 1) \
        LET defs = (FOR e IN @@defines FILTER e._from IN ids REMOVE e IN @@defines RETURN 1) \
        LET calls = (FOR e IN @@calls FILTER e._from IN ids REMOVE e IN @@calls RETURN 1) \
        LET impls = (FOR e IN @@implements FILTER e._from IN ids REMOVE e IN @@implements RETURN 1) \
        LET imps = (FOR e IN @@imports FILTER e._from IN ids REMOVE e IN @@imports RETURN 1) \
        RETURN 1";
    let bind = json!({
        "@symbols": CODEBASE.symbols,
        "@defines": CODEBASE.defines_edges,
        "@calls": CODEBASE.calls_edges,
        "@implements": CODEBASE.implements_edges,
        "@imports": CODEBASE.imports_edges,
        "files_name": CODEBASE.files,
        "key": file_key,
    });
    if let Err(e) =
        hades_core::db::query::query(db, aql, Some(&bind), None, false, ExecutionTarget::Writer)
            .await
    {
        warn!(file_key, error = %e, "failed to purge stale symbols/edges before re-ingest");
    }
}

// ── Incremental check ───────────────────────────────────────────────────

/// May the fidelity guard stand aside for this file, because a post-loop LSP
/// phase will re-supply what the rewrite drops?
///
/// True for Go alone, and only when the gopls phase will actually run. Go has no
/// per-file semantic analyzer, so `ingest_file` can never reproduce a stored
/// `Semantic` on its own and the guard would pin the node forever (#193).
///
/// Rust is deliberately NOT in this set. `rust_ast` produces `Semantic` per file
/// and rust-analyzer only augments it, so a stored `Semantic` is something
/// `ingest_file` normally reproduces. Releasing the guard there would let a file
/// `syn` cannot parse — a mid-edit save, or syntax newer than the pinned `syn` —
/// fall to tree-sitter and silently overwrite semantic artifacts nothing would
/// restore.
///
/// Keyed on the path's own extension rather than on `ingest_file`'s resolved
/// language, because the resolved language *is* `--language` when that flag is
/// given: `--language go` makes every file in the tree resolve to Go, which
/// would hand the hatch to `.rs`, `.py` and `.h` files whose stored `Semantic`
/// gopls cannot rebuild. The override can force how a file is parsed; it cannot
/// make gopls able to re-enrich it.
///
/// Note what `gopls_scheduled` does and does not promise. It means the preflight
/// found a runnable binary, so the phase will run — not that it will succeed for
/// this file's module. Releasing the guard is therefore a bet that re-enrichment
/// follows the purge, and the bet is covered in two places.
///
/// A module resolution is required here, not just a `.go` extension.
/// `group_files_by_go_module` silently drops any path with no `go.mod`/`go.work`
/// ancestor, so a module-less `.go` file is one gopls will never be handed: it
/// would land in neither `workspaces` nor `workspaces_attempted`, leaving the
/// caller's partial-failure check blind to it while the guard had already been
/// released. Requiring the same resolution the grouping uses keeps the hatch a
/// strict subset of what the phase will actually attempt.
///
/// The remaining case — a module gopls is handed but cannot start a session on —
/// is caught by `run_ingest`, which compares `workspaces` against
/// `workspaces_attempted` and refuses to exit 0 over files it purged and could
/// not rebuild.
fn reenrichment_hatch(file_path: &Path, rel_path: &str, gopls_scheduled: bool) -> bool {
    gopls_scheduled
        && Language::from_path(rel_path) == Some(Language::Go)
        && hades_core::code::lsp::gopls::find_go_module_root(file_path).is_some()
}

/// Return true when replacing the stored artifacts would lower fidelity.
///
/// `reenriched_this_run` is the escape hatch for languages whose semantic
/// artifacts come from a post-loop LSP phase rather than from the per-file
/// analyzer. Go has no semantic analyzer, so `ingest_file` can only ever offer
/// `Structural`; the gopls phase supplies the semantic symbols and edges
/// afterwards, in this same run. Blocking the rewrite there protects nothing —
/// the purge is immediately followed by re-enrichment — while permanently
/// pinning every `.go` node against re-ingest (#193). C++ and Python have no
/// such phase, so a genuine downgrade there is permanent and still blocked, and
/// neither does Rust: `rust_ast` produces `Semantic` per file and
/// rust-analyzer only augments it, so the caller must not set this for Rust.
///
/// The hatch never applies to an incoming `Text`. That tier is only reached
/// when every parser failed, and a file that defeated tree-sitter defeats the
/// LSP phase too, so nothing re-supplies what the purge would drop. Enforced
/// here rather than left to callers, because it is the one case where being
/// wrong costs a silent, permanent loss of structure.
fn should_preserve_tier(
    stored: Option<AnalysisTier>,
    incoming: AnalysisTier,
    allow_downgrade: bool,
    reenriched_this_run: bool,
) -> bool {
    let hatch = reenriched_this_run && incoming > AnalysisTier::Text;
    !allow_downgrade && !hatch && stored.is_some_and(|tier| tier > incoming)
}

/// Enforce monotonic analyzer fidelity before any destructive per-file write.
///
/// Compares against the stored file node's `analysis_tier`, which records the
/// analysis that produced that node's `symbol_hash` and chunks — the LSP phases
/// deliberately leave it alone (see `store_lsp_extractions`).
async fn preserve_higher_fidelity(
    db: &ArangoPool,
    file_key: &str,
    incoming: AnalysisTier,
    allow_downgrade: bool,
    reenriched_this_run: bool,
) -> Result<bool> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            let stored = doc["analysis_tier"].as_str().and_then(AnalysisTier::parse);
            let preserve =
                should_preserve_tier(stored, incoming, allow_downgrade, reenriched_this_run);
            if preserve && doc["relationships_pending"] == true {
                anyhow::bail!(
                    "relationship recovery requires the stored analyzer tier; restore its prerequisites or explicitly allow analysis downgrade"
                );
            }
            Ok(preserve)
        }
        Err(e) if e.is_not_found() => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Check if a file can be skipped during incremental ingest.
///
/// Returns `Some(true)` if the file should be skipped:
/// - Content hash matches, relationships are complete, and embeddings need no backfill
///
/// Returns `Some(false)` if re-processing is needed:
/// - Content hash differs, relationships are pending, or embeddings need backfill
///
/// Returns `None` if the file is not in the database (first ingest).
async fn check_unchanged(
    db: &ArangoPool,
    file_key: &str,
    new_content_hash: &str,
    embedder_available: bool,
) -> Result<Option<bool>> {
    match crud::get_document(db, CODEBASE.files, file_key).await {
        Ok(doc) => {
            if doc["relationships_pending"] == true {
                return Ok(Some(false));
            }

            // `content_hash` and not `symbol_hash` (#7). A row written before
            // that field existed has none, so the empty string never matches a
            // real digest and the file is re-processed once to record one --
            // the same "no stored hash means changed" rule the document half
            // uses, rather than a skip that could never be revisited.
            let stored_hash = doc["content_hash"].as_str().unwrap_or("");
            if stored_hash != new_content_hash {
                return Ok(Some(false)); // content changed, must re-process
            }
            // Code unchanged. Skip only if embeddings aren't needed or already complete.
            if embedder_available {
                // Files with no chunks have nothing to embed — always skip.
                let chunk_count = doc["chunk_count"].as_u64().unwrap_or(0);
                if chunk_count == 0 {
                    return Ok(Some(true));
                }
                // Skip only if every chunk has an embedding. A partial-success
                // state (e.g. from a prior ingest that hit embedder OOM on some
                // batches) must NOT be treated as "done" — the missing chunks
                // need backfill on this run. Previously this checked only
                // `embedding_count > 0`, which left partial-success files stuck.
                let embedding_count = doc["embedding_count"].as_u64().unwrap_or(0);
                Ok(Some(embedding_count >= chunk_count))
            } else {
                Ok(Some(true)) // no embedder → nothing to backfill → skip
            }
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ── semantic language-server enrichment ───────────────────────────────

/// Stats returned from the rust-analyzer enrichment phase.
///
/// Analysis and storage are reported separately (#180): `workspaces` counts
/// crates/modules the analyzer processed, while `store_errors`/`store_failed`
/// describe what happened when writing the results to ArangoDB. A store
/// failure must not be conflated with `workspaces == 0` — that signature is
/// reserved for the analyzer itself producing nothing (#164).
#[derive(Default)]
struct SemanticLspStats {
    /// Symbol documents actually stored (created + updated).
    symbols: usize,
    /// Edge documents actually stored (created + updated).
    edges: usize,
    /// Crates (rust-analyzer) or modules (gopls) successfully analyzed.
    workspaces: usize,
    /// Crates/modules the phase *attempted*. Greater than `workspaces` means
    /// some were skipped after their language-server session failed to start,
    /// which the caller must treat as loss rather than success (#194 review).
    ///
    /// Supplied by the phase rather than derived inside `store_lsp_extractions`:
    /// that function has four exits, and deriving it there let them disagree
    /// (two returned `Default`, i.e. zero, while `workspaces` was non-zero).
    /// A `saturating_sub` against a zero placeholder never fires, so the check
    /// this field exists for would have been silently dead on the second phase
    /// to adopt it.
    workspaces_attempted: usize,
    /// Documents ArangoDB rejected individually (illegal key, bad body).
    /// Retained for output compatibility; atomic stores report store_failed instead.
    store_errors: usize,
    /// The store stage failed wholesale (request-level error). Analysis
    /// results exist but none of them reached the database.
    store_failed: bool,
    /// Workspace roots whose language-server session failed to start. Their
    /// files were handed to the phase but never enriched, so the caller needs
    /// them to decide whether this run destroyed anything (#194 review).
    failed_workspaces: Vec<PathBuf>,
}

/// Run rust-analyzer over ingested Rust files to produce rich symbols and edges.
///
/// Groups files by crate root, spawns a `RustAnalyzerSession` per crate,
/// extracts qualified symbols with call hierarchy and impl-trait info, then
/// stores the enriched symbol documents and edges to ArangoDB.
///
/// This phase is additive: it overwrites top-level symbol documents (same
/// keys as syn) and adds new method-level symbols and cross-file edges
/// that syn cannot produce.
async fn run_rust_analyzer_phase(
    db: &ArangoPool,
    base: &Path,
    rust_files: &[PathBuf],
    command: Option<&str>,
) -> Result<SemanticLspStats> {
    let revisions = enrichment_revisions(db, base, rust_files).await?;
    let groups = group_files_by_crate(rust_files);
    if groups.is_empty() {
        return Ok(SemanticLspStats::default());
    }

    info!(
        crate_count = groups.len(),
        file_count = rust_files.len(),
        "starting rust-analyzer enrichment"
    );

    let mut all_extractions = HashMap::new();
    let mut crates_analyzed = 0;

    for (crate_root, crate_files) in &groups {
        info!(
            crate_root = %crate_root.display(),
            file_count = crate_files.len(),
            "analyzing crate with rust-analyzer"
        );

        let session = match RustAnalyzerSession::start_with_options(
            crate_root,
            command,
            hades_core::code::lsp::DEFAULT_INDEX_TIMEOUT_SECS,
        )
        .await
        {
            Ok(s) => s,
            Err(e) => {
                // If rust-analyzer isn't installed or fails to start,
                // skip this crate but try others.
                warn!(
                    crate_root = %crate_root.display(),
                    error = %e,
                    "failed to start rust-analyzer session, skipping crate"
                );
                continue;
            }
        };

        let extractor = RustSymbolExtractor::new(&session, true).with_path_root(base);
        let file_refs: Vec<&Path> = crate_files.iter().map(|p| p.as_path()).collect();
        let extractions = extractor.extract_crate(&file_refs).await;

        // Convert absolute path keys to relative paths (matching file_key convention).
        for (abs_path_str, extraction) in extractions {
            let abs = Path::new(&abs_path_str);
            let rel = abs
                .strip_prefix(base)
                .unwrap_or(abs)
                .to_string_lossy()
                .to_string();
            all_extractions.insert(rel, extraction);
        }

        crates_analyzed += 1;

        // Graceful shutdown — non-fatal if it fails.
        if let Err(e) = session.shutdown().await {
            debug!(error = %e, "rust-analyzer shutdown warning (non-fatal)");
        }
    }

    store_lsp_extractions(
        db,
        all_extractions,
        revisions,
        crates_analyzed,
        groups.len(),
        "rust-analyzer",
        "ra",
        base.to_str().context("ingest root must be valid UTF-8")?,
    )
    .await
}

/// Run gopls over each discovered Go module. Tree-sitter artifacts remain in
/// place when gopls is absent or a module fails, satisfying the #152 fallback
/// contract without a database-wide language mode.
async fn run_gopls_phase(
    db: &ArangoPool,
    base: &Path,
    go_files: &[PathBuf],
    command: Option<&str>,
) -> Result<SemanticLspStats> {
    let revisions = enrichment_revisions(db, base, go_files).await?;
    let groups = group_files_by_go_module(go_files);
    if groups.is_empty() {
        return Ok(SemanticLspStats::default());
    }
    info!(
        module_count = groups.len(),
        file_count = go_files.len(),
        "starting gopls semantic enrichment"
    );
    let mut all_extractions = HashMap::new();
    let mut modules_analyzed = 0;
    let mut failed_modules: Vec<PathBuf> = Vec::new();
    for (module_root, module_files) in &groups {
        let session = match GoplsSession::start_with_options(
            module_root,
            command,
            hades_core::code::lsp::DEFAULT_INDEX_TIMEOUT_SECS,
        )
        .await
        {
            Ok(session) => session,
            Err(error) => {
                warn!(
                    module_root = %module_root.display(),
                    %error,
                    "gopls unavailable for module; Tree-sitter data retained"
                );
                failed_modules.push(module_root.clone());
                continue;
            }
        };
        let extractor = GoSymbolExtractor::new(&session, true).with_path_root(base);
        let file_refs: Vec<&Path> = module_files.iter().map(PathBuf::as_path).collect();
        for (absolute, extraction) in extractor.extract_module(&file_refs).await {
            let absolute = Path::new(&absolute);
            let relative = absolute
                .strip_prefix(base)
                .unwrap_or(absolute)
                .to_string_lossy()
                .into_owned();
            all_extractions.insert(relative, extraction);
        }
        modules_analyzed += 1;
        if let Err(error) = session.shutdown().await {
            debug!(%error, "gopls shutdown warning (non-fatal)");
        }
    }
    let mut stats = store_lsp_extractions(
        db,
        all_extractions,
        revisions,
        modules_analyzed,
        groups.len(),
        "gopls",
        "gopls",
        base.to_str().context("ingest root must be valid UTF-8")?,
    )
    .await?;
    stats.failed_workspaces = failed_modules;
    Ok(stats)
}

async fn enrichment_revisions(
    db: &ArangoPool,
    base: &Path,
    files: &[PathBuf],
) -> Result<HashMap<String, Option<String>>> {
    let namespace = base.to_str().context("ingest root must be valid UTF-8")?;
    let mut revisions = HashMap::new();
    for path in files {
        let relative = path
            .strip_prefix(base)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let key = keys::scoped_file_key(namespace, &relative);
        let revision = super::codebase_persist::revision(db.writer(), &key).await?;
        revisions.insert(relative, revision);
    }
    Ok(revisions)
}

/// Enrichment is a separate atomic stage: failed stores retain the committed
/// structural graph and do not publish partial symbols, edges or success flags.
#[allow(clippy::too_many_arguments)]
async fn store_lsp_extractions(
    db: &ArangoPool,
    all_extractions: HashMap<String, FileExtraction>,
    revisions: HashMap<String, Option<String>>,
    workspaces: usize,
    workspaces_attempted: usize,
    analyzer: &'static str,
    metadata_prefix: &'static str,
    namespace: &str,
) -> Result<SemanticLspStats> {
    if all_extractions.is_empty() {
        return Ok(SemanticLspStats {
            workspaces,
            workspaces_attempted,
            ..Default::default()
        });
    }
    let file_patches: Vec<_> = all_extractions
        .iter()
        .map(|(path, extraction)| {
            (
                path.clone(),
                keys::scoped_file_key(namespace, path),
                extraction.symbols.len(),
                extraction.analyzed_at.clone(),
            )
        })
        .collect();
    let resolver = LspEdgeResolver::new_scoped(all_extractions, analyzer, namespace);
    let symbols = resolver
        .build_symbol_documents()
        .into_iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let semantic_edges = resolver.build_edges();
    let edge_docs: Vec<(EdgeKind, Value)> = semantic_edges
        .iter()
        .map(|e| {
            let from_suffix = e.from.rsplit('/').next().unwrap_or(&e.from);
            let to_suffix = e.to.rsplit('/').next().unwrap_or(&e.to);
            let edge_key = keys::edge_key(from_suffix, e.kind.as_str(), to_suffix);
            let mut doc = json!({
                "_key": edge_key,
                "_from": e.from,
                "_to": e.to,
                "analysis_tier": "semantic",
                "analyzer": analyzer,
                "resolution": "semantic",
            });
            // Merge edge metadata.
            if let Value::Object(meta) = &e.metadata
                && let Value::Object(ref mut obj) = doc
            {
                for (k, v) in meta {
                    obj.insert(k.clone(), v.clone());
                }
            }
            (e.kind, doc)
        })
        .collect();

    let symbol_count = symbols.len();
    let edge_count = edge_docs.len();
    let collections = CODEBASE
        .all_collections()
        .into_iter()
        .map(|(name, _)| name.to_owned())
        .collect();
    let stored = hades_core::db::transaction::run(db, collections, move |client| async move {
        for (path, key, _, _) in &file_patches {
            let expected = revisions.get(path).ok_or_else(|| hades_core::db::ArangoError::Request("missing enrichment preparation revision".into()))?;
            if expected.is_none() || &super::codebase_persist::revision(&client, key).await? != expected {
                return Err(hades_core::db::ArangoError::Request("file changed during enrichment; retry ingestion".into()));
            }
        }
        let mut batches = vec![(CODEBASE.symbols, symbols)];
        for kind in [EdgeKind::Defines, EdgeKind::Calls, EdgeKind::Implements] {
            batches.push((kind.collection(), edge_docs.iter().filter(|(k, _)| *k == kind).map(|(_, d)| d.clone()).collect()));
        }
        for (collection, docs) in batches {
            if docs.is_empty() { continue; }
            let response = client.post(&format!("document/{collection}?overwriteMode=replace"), &json!(docs)).await?;
            let rows = response.as_array().ok_or_else(|| hades_core::db::ArangoError::Request("invalid enrichment batch response".into()))?;
            if rows.len() != docs.len() || rows.iter().any(|row| row["error"] == true) {
                return Err(hades_core::db::ArangoError::Request(format!("failed enrichment batch in {collection}")));
            }
        }
        for (_, key, count, analyzed_at) in &file_patches {
            let patch = json!({format!("{metadata_prefix}_analyzed"):true,
                format!("{metadata_prefix}_symbol_count"):count,
                format!("{metadata_prefix}_analyzed_at"):analyzed_at});
            client.patch(&format!("document/{}/{key}", CODEBASE.files), &patch).await?;
        }
        super::codebase_persist::query(&client,
            "FOR fk IN @fkeys LET c = LENGTH(FOR s IN @@sym FILTER s.file_key == fk RETURN 1) UPDATE fk WITH { symbol_count: c } IN @@files",
            json!({"fkeys":file_patches.iter().map(|(_,key,_,_)| key).collect::<Vec<_>>(),
                "@sym":CODEBASE.symbols,"@files":CODEBASE.files})).await?;
        Ok(())
    }).await;
    if let Err(error) = stored {
        warn!(%error, analyzer, "atomic enrichment store failed; committed graph retained");
        return Ok(SemanticLspStats {
            workspaces,
            workspaces_attempted,
            store_failed: true,
            ..Default::default()
        });
    }
    Ok(SemanticLspStats {
        symbols: symbol_count,
        edges: edge_count,
        workspaces,
        workspaces_attempted,
        ..Default::default()
    })
}

// ── Python import graph resolution ──────────────────────────────────────

/// Resolve Python import statements to file→file edges.
///
/// Only creates edges for imports that resolve to files within the
/// ingested set. External package imports are silently skipped.
/// Build a symbol index for Python files: bare name → vec of (rel_path, symbol_key).
#[cfg(test)]
fn build_python_symbol_index(
    file_symbols: &HashMap<String, Vec<Symbol>>,
) -> HashMap<String, Vec<(String, String)>> {
    build_python_symbol_index_scoped(file_symbols, "")
}

fn build_python_symbol_index_scoped(
    file_symbols: &HashMap<String, Vec<Symbol>>,
    namespace: &str,
) -> HashMap<String, Vec<(String, String)>> {
    let mut index: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (rel_path, symbols) in file_symbols {
        let fkey = keys::scoped_file_key(namespace, rel_path);
        for sym in symbols {
            // Only index definitions, not imports.
            if sym.kind == SymbolKind::Import {
                continue;
            }
            // Index is keyed by the bare name (call sites use bare names), but
            // the value must be the qualified-name-derived key so edges target
            // the actual stored vertex (#113).
            let skey = keys::symbol_key(&fkey, &sym.qualified_name(), sym.start_line);
            index
                .entry(sym.name.clone())
                .or_default()
                .push((rel_path.clone(), skey));
        }
    }
    index
}

/// Build a mapping from Python module name → relative file path.
fn build_python_module_map(all_files: &HashMap<String, Vec<Symbol>>) -> HashMap<String, String> {
    let mut module_to_file: HashMap<String, String> = HashMap::new();
    for rel_path in all_files.keys() {
        let p = Path::new(rel_path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let mut parts: Vec<&str> = p
            .parent()
            .map(|parent| {
                parent
                    .components()
                    .filter_map(|c| c.as_os_str().to_str())
                    .collect()
            })
            .unwrap_or_default();
        parts.push(stem);
        let module = parts.join(".");
        let module = module
            .strip_suffix(".__init__")
            .unwrap_or(&module)
            .to_string();
        module_to_file.insert(module, rel_path.clone());
    }
    module_to_file
}

/// Resolve a Python module name to a file path, trying exact then prefix match.
fn resolve_module_to_file<'a>(
    module: &str,
    module_to_file: &'a HashMap<String, String>,
) -> Option<&'a String> {
    module_to_file.get(module).or_else(|| {
        let mut parts: Vec<&str> = module.split('.').collect();
        while parts.len() > 1 {
            parts.pop();
            let prefix = parts.join(".");
            if let Some(path) = module_to_file.get(&prefix) {
                return Some(path);
            }
        }
        None
    })
}

#[cfg(test)]
fn resolve_python_imports(
    python_imports: &HashMap<String, Vec<Symbol>>,
    python_file_symbols: &HashMap<String, Vec<Symbol>>,
    symbol_index: &HashMap<String, Vec<(String, String)>>,
) -> Vec<Value> {
    resolve_python_imports_scoped(python_imports, python_file_symbols, symbol_index, "")
}

fn resolve_python_imports_scoped(
    python_imports: &HashMap<String, Vec<Symbol>>,
    python_file_symbols: &HashMap<String, Vec<Symbol>>,
    symbol_index: &HashMap<String, Vec<(String, String)>>,
    namespace: &str,
) -> Vec<Value> {
    // Build module→file mapping from all known Python files.
    let module_to_file = build_python_module_map(python_file_symbols);

    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (source_path, import_syms) in python_imports {
        let source_fkey = keys::scoped_file_key(namespace, source_path);

        for sym in import_syms {
            let import_type = sym
                .metadata
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let module = sym
                .metadata
                .get("module")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            if module.is_empty() {
                continue;
            }

            match import_type {
                "from_import" => {
                    // `from module import Name` — try to resolve Name to a specific symbol.
                    let original_name = sym
                        .metadata
                        .get("original_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&sym.name);

                    // First, find the target file.
                    let target_file = resolve_module_to_file(module, &module_to_file);

                    // Try symbol-level resolution: look up the imported name in the symbol index.
                    let mut resolved = false;
                    if let Some(targets) = symbol_index.get(original_name) {
                        // If we know the target file, prefer symbols from that file.
                        let target = if let Some(tf) = target_file {
                            targets.iter().find(|(path, _)| path == tf)
                        } else {
                            None
                        }
                        .or_else(|| targets.first());

                        if let Some((target_path, target_skey)) = target
                            && target_path != source_path
                        {
                            let edge_key = keys::edge_key(&source_fkey, "imports", target_skey);
                            if seen.insert(edge_key.clone()) {
                                edges.push(json!({
                                    "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                    "_to": format!("{}/{}", CODEBASE.symbols, target_skey),
                                    "_key": edge_key,
                                    "resolved": true,
                                    "style": "from_import",
                                    "source_path": source_path,
                                    "target_path": target_path,
                                    "symbol_name": original_name,
                                    "module_path": module,
                                }));
                                resolved = true;
                            }
                        }
                    }

                    // Fall back to file→file if symbol not found (external package or
                    // symbol not in our index).
                    if !resolved
                        && let Some(target_path) = target_file
                        && target_path != source_path
                    {
                        let target_fkey = keys::scoped_file_key(namespace, target_path);
                        let edge_key = keys::edge_key(&source_fkey, "imports", &target_fkey);
                        if seen.insert(edge_key.clone()) {
                            edges.push(json!({
                                "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                "_key": edge_key,
                                "resolved": false,
                                "style": "from_import",
                                "source_path": source_path,
                                "target_path": target_path,
                                "module_path": module,
                            }));
                        }
                    }
                }

                "import" => {
                    // `import module` — file-level edge (no specific symbol target).
                    if let Some(target_path) = resolve_module_to_file(module, &module_to_file)
                        && target_path != source_path
                    {
                        let target_fkey = keys::scoped_file_key(namespace, target_path);
                        let edge_key = keys::edge_key(&source_fkey, "imports", &target_fkey);
                        if seen.insert(edge_key.clone()) {
                            edges.push(json!({
                                "_from": format!("{}/{}", CODEBASE.files, source_fkey),
                                "_to": format!("{}/{}", CODEBASE.files, target_fkey),
                                "_key": edge_key,
                                "resolved": false,
                                "style": "import",
                                "source_path": source_path,
                                "target_path": target_path,
                                "module_path": module,
                            }));
                        }
                    }
                }

                _ => {}
            }
        }
    }

    edges
}

/// Resolve an analyzer command (config/env override wins over PATH) and probe
/// it from the workspace. Returns the command to use, or None when the
/// operator explicitly accepted the downgrade.
///
/// The probe runs from `workspace` because the rustup shim resolves
/// per-directory (#164): the same `rust-analyzer` can work in a shell and die
/// inside a repo whose rust-toolchain.toml pins a toolchain missing the
/// component. Probing anywhere else validates the wrong toolchain.
fn preflight_or_bail(
    name: &str,
    configured: Option<&str>,
    workspace: &Path,
    allow_analysis_downgrade: bool,
) -> Result<Option<String>> {
    let probe = hades_core::code::lsp::resolve_and_probe(name, configured, workspace);
    match probe.outcome {
        Ok(version) => {
            info!(analyzer = name, %version, source = probe.source, "analyzer preflight passed");
            Ok(Some(probe.command))
        }
        Err(e) if allow_analysis_downgrade => {
            warn!(
                analyzer = name,
                error = %e,
                "analyzer preflight FAILED; proceeding without semantic \
                 enrichment because --allow-analysis-downgrade was passed"
            );
            Ok(None)
        }
        Err(e) => anyhow::bail!(
            "{name} preflight failed ({source}): {e}\n\
             Source files needing it were discovered, and `--force` would purge \
             semantic edges this analyzer rebuilds. Fix the analyzer (for the \
             rustup shim: `rustup component add rust-analyzer --toolchain \
             <the workspace's pinned toolchain>`), pin a binary in hades.yaml \
             under `analyzers.{}`, or pass --allow-analysis-downgrade to \
             proceed without semantic enrichment.",
            name.replace('-', "_"),
            source = probe.source,
        ),
    }
}

/// Is `path` a target for semantic enrichment in `want` language?
///
/// The ONE predicate shared by the preflight gate and the per-file tracking
/// loop, so the set the gate protects and the set the phases process cannot
/// diverge. The language override counts (matching the loop's historical
/// behavior), and unparsed-allowlisted files never count.
fn is_semantic_target(
    path: &Path,
    lang_override: Option<Language>,
    unparsed_set: &std::collections::HashSet<String>,
    want: Language,
    want_ext: &str,
) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    let is_unparsed = Language::from_path(&path.to_string_lossy()).is_none()
        && ext.as_deref().is_some_and(|e| unparsed_set.contains(e));
    !is_unparsed && (lang_override == Some(want) || ext.as_deref() == Some(want_ext))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::test_support::{Fixtures, with_temp_db};
    use std::collections::HashSet;
    use std::fs;
    use tempfile::TempDir;

    // ── #159 regression: shrinking re-ingest must not leave orphan chunks ──

    /// Count documents in `col` whose `file_key` matches.
    async fn count_by_file_key(pool: &ArangoPool, col: &str, fkey: &str) -> u64 {
        let aql = "FOR d IN @@col FILTER d.file_key == @fk COLLECT WITH COUNT INTO n RETURN n";
        let bind = json!({ "@col": col, "fk": fkey });
        hades_core::db::query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    /// Remove every trace of a test fixture file from the codebase graph.
    /// Sweep tag for the #159 live test's fixtures, across all PIDs.
    /// How many of this file's stored chunks contain `needle`.
    ///
    /// The question #7 turns on: not how many chunks exist, but whether any of
    /// them still carries text the source no longer has.
    async fn count_chunks_containing(pool: &ArangoPool, fkey: &str, needle: &str) -> u64 {
        let aql = "FOR c IN @@chunks FILTER c.file_key == @fk AND CONTAINS(c.text, @needle) \
                   COLLECT WITH COUNT INTO n RETURN n";
        let bind = json!({ "@chunks": CODEBASE.chunks, "fk": fkey, "needle": needle });
        hades_core::db::query::query_single(pool, aql, Some(&bind), ExecutionTarget::Reader)
            .await
            .ok()
            .flatten()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    }

    async fn cleanup_fixture(pool: &ArangoPool, fkey: &str) {
        purge_file_symbols_and_edges(pool, fkey).await;
        delete_file_chunks(pool, fkey).await;
        delete_file_embeddings(pool, fkey).await;
        let aql = "FOR d IN @@files FILTER d._key == @fk REMOVE d IN @@files";
        let bind = json!({ "@files": CODEBASE.files, "fk": fkey });
        let _ = hades_core::db::query::query(
            pool,
            aql,
            Some(&bind),
            None,
            false,
            ExecutionTarget::Writer,
        )
        .await;
    }

    /// #193 end to end: a `.go` node stamped `semantic` by the old gopls patch
    /// must be re-ingestable, and must still be protected when nothing will
    /// re-enrich it.
    ///
    /// The regression this pins is a property of a real write-read-write cycle,
    /// not of `should_preserve_tier`'s truth table: the guard returned early at
    /// `preserve_higher_fidelity`, ahead of the `!force` check, so every `.go`
    /// file reported `skipped` on every run after the first with `--force`
    /// included. A unit test on a pure function cannot observe that.
    ///
    /// Both directions are asserted, because the fix is only correct if it
    /// releases the guard *and* leaves it in force where nothing restores the
    /// purged data.
    ///
    /// A comment-only edit re-chunks, and the retired text is *gone* (#7).
    ///
    /// The end-to-end half of the gate test below. That one asserts the decision;
    /// this one drives `ingest_file` twice and reads the stored chunks, because
    /// the harm reported in #7 was not a wrong decision in the abstract -- it was
    /// `db_query` returning a sentence the source had retired, as if current.
    ///
    /// No embedder: chunks are stored either way, and the text is what is under
    /// test. Runs in its own database, so the counts are exact.
    #[tokio::test]
    async fn a_comment_only_edit_removes_the_retired_chunk_text() {
        with_temp_db("chunktext", Fixtures::Codebase, |pool| async move {
            const RETIRED: &str = "the lock file is the subject of this resolution";
            const REPLACEMENT: &str = "the flags gate nothing and the outer run resolved first";

            let dir = TempDir::new().unwrap();
            let path = dir.path().join("manifest.rs");
            let rel_path = format!("__hades_test7_{}/manifest.rs", std::process::id());
            let fkey = keys::file_key(&rel_path);
            let config = HadesConfig::default();

            // Padding so the file chunks into more than one piece, with the
            // sentence under test in a comment above a later symbol -- the exact
            // shape of the edit that used to be skipped.
            let body: String = (0..30)
                .map(|i| {
                    format!(
                        "/// Padded documentation for generated function {i}, long enough \
                         that the chunker emits several chunks for this file.\n\
                         pub fn generated_{i}(input: u64) -> u64 {{ input + {i} }}\n\n"
                    )
                })
                .collect();

            let ingest = |src: String| {
                let pool = pool.clone();
                let path = path.clone();
                let rel_path = rel_path.clone();
                let config = config.clone();
                async move {
                    fs::write(&path, &src).unwrap();
                    let mut imports = ImportContext::default();
                    ingest_file(
                        &pool,
                        None,
                        &config,
                        &path,
                        &rel_path,
                        None,
                        &mut imports,
                        None,
                        false,
                        false,
                        false,
                        FALLBACK_WINDOW_CHARS,
                        "",
                    )
                    .await
                    .expect("ingest")
                }
            };

            let first = ingest(format!("/// {RETIRED}\n{body}")).await;
            assert!(
                first.skipped.is_none_or(|s| !s),
                "the first ingest must not skip"
            );
            assert_eq!(
                count_chunks_containing(&pool, &fkey, RETIRED).await,
                1,
                "setup: the retired sentence must be stored before it is retired"
            );

            // Comment only. Every symbol name is identical, so the old
            // `symbol_hash` gate skipped this and left the sentence behind.
            let second = ingest(format!("/// {REPLACEMENT}\n{body}")).await;
            assert!(
                second.skipped.is_none_or(|s| !s),
                "a comment-only edit must be re-processed, not skipped (#7)"
            );
            assert_eq!(
                count_chunks_containing(&pool, &fkey, RETIRED).await,
                0,
                "the retired sentence is still in a stored chunk, which is what \
                 #7 reported: semantic search serves it as current"
            );
            assert_eq!(
                count_chunks_containing(&pool, &fkey, REPLACEMENT).await,
                1,
                "the replacement text must be stored"
            );
        })
        .await
    }

    // --- symbol key remapping (#9) ------------------------------------------

    /// Re-pointing collapses onto the canonical key instead of duplicating.
    ///
    /// The rust-analyzer and gopls phases re-resolve cross-file `calls` and
    /// `implements` edges for every file in the run, skipped ones included, and
    /// write them under `edge_key(from, kind, to)` for the *new* target. An
    /// in-place `UPDATE` of `_to` left the old document beside that one, keyed
    /// for a target it no longer names: two documents for one relation, which
    /// traversals and neighbour counts double. Found by review of #11.
    #[tokio::test]
    async fn re_pointing_collapses_onto_the_canonical_edge() {
        with_temp_db("repoint", Fixtures::Codebase, |pool| async move {
            let caller = "f_consumer_rs__calls_it__aaa";
            let (old, new) = ("f_provider_rs__target__old", "f_provider_rs__target__new");

            // Only the new symbol exists: the file was rewritten.
            let sym = json!({ "_key": new, "file_key": "f_provider_rs",
                              "qualified_name": "target", "start_line": 20 });
            crud::insert_documents(&pool, CODEBASE.symbols, &[sym], true)
                .await
                .expect("symbol");

            // What the analyzer phase already wrote, at the canonical key.
            let canonical = keys::edge_key(caller, "calls", new);
            let rebuilt = json!({ "_key": canonical,
                "_from": format!("{}/{}", CODEBASE.symbols, caller),
                "_to": format!("{}/{}", CODEBASE.symbols, new) });
            // And the stale one the skipped dependent still owns.
            let stale_key = keys::edge_key(caller, "calls", old);
            let stale = json!({ "_key": stale_key,
                "_from": format!("{}/{}", CODEBASE.symbols, caller),
                "_to": format!("{}/{}", CODEBASE.symbols, old) });
            crud::insert_documents(&pool, CODEBASE.calls_edges, &[rebuilt, stale], true)
                .await
                .expect("edges");
            assert_eq!(count_all(&pool, CODEBASE.calls_edges).await, 2, "setup");

            repoint_inbound_edges(&pool, &[(old.to_string(), new.to_string())]).await;

            assert_eq!(
                count_all(&pool, CODEBASE.calls_edges).await,
                1,
                "one relation must leave one edge; an in-place update leaves two, \
                 the analyzer's and a stale-keyed copy"
            );
            let keys_left = edge_keys(&pool, CODEBASE.calls_edges).await;
            assert_eq!(
                keys_left,
                vec![canonical],
                "the surviving edge must carry the canonical key for its target"
            );
            assert_eq!(
                dangling_in(&pool, CODEBASE.calls_edges).await,
                0,
                "the surviving edge must resolve"
            );
        })
        .await
    }

    /// A chained remap sends each edge to its own target, not the last one.
    ///
    /// Two symbols sharing a qualified name can move so that one's new key is
    /// another's old key (`A -> B` and `B -> C`). Resolving against a collection
    /// being written in the same query would drag the edge for the first symbol
    /// onto the third position, attaching a dependency to a definition nobody
    /// wrote -- the very outcome the count-changed guard exists to prevent.
    /// Reading every edge before writing any is what makes this hold.
    #[tokio::test]
    async fn a_chained_remap_does_not_drag_an_edge_past_its_target() {
        with_temp_db("chain", Fixtures::Codebase, |pool| async move {
            let caller = "f_c_rs__calls_it__aaa";
            let (a, b, c) = ("f_p_rs__dup__a", "f_p_rs__dup__b", "f_p_rs__dup__c");

            let syms: Vec<Value> = [b, c]
                .iter()
                .map(|k| {
                    json!({ "_key": k, "file_key": "f_p_rs",
                                 "qualified_name": "dup", "start_line": 1 })
                })
                .collect();
            crud::insert_documents(&pool, CODEBASE.symbols, &syms, true)
                .await
                .expect("symbols");

            let edges: Vec<Value> = [a, b]
                .iter()
                .map(|t| {
                    json!({ "_key": keys::edge_key(caller, "calls", t),
                            "_from": format!("{}/{}", CODEBASE.symbols, caller),
                            "_to": format!("{}/{}", CODEBASE.symbols, t) })
                })
                .collect();
            crud::insert_documents(&pool, CODEBASE.calls_edges, &edges, true)
                .await
                .expect("edges");

            // A moved to where B was; B moved on to C.
            repoint_inbound_edges(
                &pool,
                &[
                    (a.to_string(), b.to_string()),
                    (b.to_string(), c.to_string()),
                ],
            )
            .await;

            let mut targets = edge_targets(&pool, CODEBASE.calls_edges).await;
            targets.sort();
            assert_eq!(
                targets,
                vec![
                    format!("{}/{}", CODEBASE.symbols, b),
                    format!("{}/{}", CODEBASE.symbols, c),
                ],
                "each edge follows its own symbol; both landing on the last key \
                 would mean an edge was dragged through the chain"
            );
        })
        .await
    }

    /// A chain split across the read chunk boundary still resolves correctly.
    ///
    /// The chunking that keeps a repository-wide remap out of one bind parameter
    /// must not reintroduce the hazard it sits beside: if a write landed between
    /// two reads, the chunk holding `B -> C` would find the edge the chunk
    /// holding `A -> B` had just moved to B, and drag it on to C. Padding puts
    /// the two halves of one chain in different read chunks.
    #[tokio::test]
    async fn a_chain_across_a_chunk_boundary_is_not_dragged() {
        with_temp_db("chunkchain", Fixtures::Codebase, |pool| async move {
            const CHUNK: usize = 2_000;
            let caller = "f_c_rs__calls_it__aaa";
            let (a, b, c) = ("f_p_rs__dup__a", "f_p_rs__dup__b", "f_p_rs__dup__c");

            let syms: Vec<Value> = [b, c]
                .iter()
                .map(|k| {
                    json!({ "_key": k, "file_key": "f_p_rs",
                            "qualified_name": "dup", "start_line": 1 })
                })
                .collect();
            crud::insert_documents(&pool, CODEBASE.symbols, &syms, true)
                .await
                .expect("symbols");

            let edges: Vec<Value> = [a, b]
                .iter()
                .map(|t| {
                    json!({ "_key": keys::edge_key(caller, "calls", t),
                            "_from": format!("{}/{}", CODEBASE.symbols, caller),
                            "_to": format!("{}/{}", CODEBASE.symbols, t) })
                })
                .collect();
            crud::insert_documents(&pool, CODEBASE.calls_edges, &edges, true)
                .await
                .expect("edges");

            // `A -> B` first, then enough inert pairs to fill the chunk, so
            // `B -> C` lands in the next read.
            let mut remap = vec![(a.to_string(), b.to_string())];
            remap.extend(
                (0..CHUNK).map(|i| (format!("f_pad__{i}__old"), format!("f_pad__{i}__new"))),
            );
            remap.push((b.to_string(), c.to_string()));
            assert!(remap.len() > CHUNK, "the chain must span two read chunks");

            repoint_inbound_edges(&pool, &remap).await;

            let mut targets = edge_targets(&pool, CODEBASE.calls_edges).await;
            targets.sort();
            assert_eq!(
                targets,
                vec![
                    format!("{}/{}", CODEBASE.symbols, b),
                    format!("{}/{}", CODEBASE.symbols, c),
                ],
                "a chain split across chunks must still send each edge to its own \
                 symbol; both on the last key means a write landed between reads"
            );
        })
        .await
    }

    async fn count_all(pool: &ArangoPool, col: &str) -> u64 {
        let aql = "FOR d IN @@col COLLECT WITH COUNT INTO n RETURN n";
        hades_core::db::query::query_single(
            pool,
            aql,
            Some(&json!({ "@col": col })),
            ExecutionTarget::Reader,
        )
        .await
        .ok()
        .flatten()
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
    }

    async fn edge_keys(pool: &ArangoPool, col: &str) -> Vec<String> {
        rows_of(pool, "FOR e IN @@col SORT e._key RETURN e._key", col).await
    }

    async fn edge_targets(pool: &ArangoPool, col: &str) -> Vec<String> {
        rows_of(pool, "FOR e IN @@col RETURN e._to", col).await
    }

    async fn rows_of(pool: &ArangoPool, aql: &str, col: &str) -> Vec<String> {
        hades_core::db::query::query(
            pool,
            aql,
            Some(&json!({ "@col": col })),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .map(|r| {
            r.results
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
    }

    async fn dangling_in(pool: &ArangoPool, col: &str) -> u64 {
        let aql = "FOR e IN @@col FILTER DOCUMENT(e._to) == null \
                   COLLECT WITH COUNT INTO n RETURN n";
        hades_core::db::query::query_single(
            pool,
            aql,
            Some(&json!({ "@col": col })),
            ExecutionTarget::Reader,
        )
        .await
        .ok()
        .flatten()
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
    }

    fn sym(name: &str, line: u64, key: &str) -> (String, u64, String) {
        (name.to_string(), line, key.to_string())
    }

    fn written(name: &str, line: u64, key: &str) -> Value {
        json!({ "qualified_name": name, "start_line": line, "_key": key })
    }

    /// A symbol that moved without being renamed is paired with its replacement.
    ///
    /// The #9 case: a comment grows above a symbol, its definition line moves,
    /// `symbol_key` hashes the line, and a dependent that was skipped still
    /// names the old key.
    #[test]
    fn a_moved_symbol_is_paired_with_its_new_key() {
        let before = vec![
            sym("first", 2, "f__first__aaa"),
            sym("target", 8, "f__target__bbb"),
        ];
        let after = vec![
            written("first", 2, "f__first__aaa"),
            written("target", 14, "f__target__ccc"),
        ];
        let pairs = pair_moved_symbol_keys(&before, &after);
        assert_eq!(
            pairs,
            vec![("f__target__bbb".to_string(), "f__target__ccc".to_string())],
            "only the symbol whose key changed is paired"
        );
    }

    /// A renamed symbol is NOT paired, so its dependents dangle honestly.
    ///
    /// Re-pointing here would silently attach a dependency to a definition the
    /// author did not write, which is worse than the dangling edge it replaces.
    #[test]
    fn a_renamed_symbol_is_not_paired() {
        let before = vec![sym("old_name", 4, "f__old_name__aaa")];
        let after = vec![written("new_name", 4, "f__new_name__bbb")];
        assert!(
            pair_moved_symbol_keys(&before, &after).is_empty(),
            "a rename must not be re-pointed: the name is gone, not moved"
        );
    }

    /// When a name's count changes, position no longer identifies a symbol.
    ///
    /// Three `Config::new` become two and the survivors shift up. Pairing by
    /// position would attach a dependent to a different definition that happens
    /// to sit where the old one did. 618 groups in one real corpus share a
    /// qualified name, so this is the common shape, not a corner.
    #[test]
    fn a_name_whose_count_changed_is_not_paired() {
        let before = vec![
            sym("Config::new", 10, "f__Config__new__a"),
            sym("Config::new", 20, "f__Config__new__b"),
            sym("Config::new", 30, "f__Config__new__c"),
        ];
        let after = vec![
            written("Config::new", 10, "f__Config__new__a"),
            written("Config::new", 25, "f__Config__new__d"),
        ];
        assert!(
            pair_moved_symbol_keys(&before, &after).is_empty(),
            "an ambiguous set must be left alone, not guessed at"
        );
    }

    /// Same count, all moved: each is paired in line order.
    #[test]
    fn same_named_symbols_pair_in_line_order() {
        let before = vec![
            sym("Wire::id", 10, "f__Wire__id__a"),
            sym("Wire::id", 20, "f__Wire__id__b"),
        ];
        let after = vec![
            written("Wire::id", 16, "f__Wire__id__x"),
            written("Wire::id", 26, "f__Wire__id__y"),
        ];
        let mut pairs = pair_moved_symbol_keys(&before, &after);
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("f__Wire__id__a".to_string(), "f__Wire__id__x".to_string()),
                ("f__Wire__id__b".to_string(), "f__Wire__id__y".to_string()),
            ],
            "the first of the pair maps to the first, by line"
        );
    }

    /// A file whose symbols did not move produces no work.
    #[test]
    fn an_unmoved_file_pairs_nothing() {
        let before = vec![sym("a", 1, "f__a__k"), sym("b", 9, "f__b__k")];
        let after = vec![written("a", 1, "f__a__k"), written("b", 9, "f__b__k")];
        assert!(pair_moved_symbol_keys(&before, &after).is_empty());
    }

    /// A first ingest has nothing to pair against.
    #[test]
    fn a_new_file_pairs_nothing() {
        assert!(pair_moved_symbol_keys(&[], &[written("a", 1, "f__a__k")]).is_empty());
    }

    /// The incremental gate reads `content_hash`, not `symbol_hash` (#7).
    ///
    /// The bug this guards: a comment-only edit leaves every symbol *name*
    /// identical, so the name-only `symbol_hash` matched and the file was
    /// skipped while its stored chunks kept text the source had retired. A
    /// corpus served that retired text through semantic search as current.
    ///
    /// Asserted at the gate rather than through a full ingest, because that is
    /// where the decision is made and a full ingest needs an embedder.
    #[tokio::test]
    async fn pending_relationships_cannot_be_skipped_or_silently_downgraded() {
        with_temp_db("pending_gate", Fixtures::Codebase, |pool| async move {
            pool.writer()
                .post(
                    "document/codebase_files",
                    &json!({
                        "_key":"pending", "content_hash":"same", "analysis_tier":"semantic",
                        "relationships_pending":true, "chunk_count":1, "embedding_count":1
                    }),
                )
                .await
                .unwrap();
            assert_eq!(
                check_unchanged(&pool, "pending", "same", true)
                    .await
                    .unwrap(),
                Some(false)
            );
            let error =
                preserve_higher_fidelity(&pool, "pending", AnalysisTier::Structural, false, false)
                    .await
                    .unwrap_err();
            assert!(error.to_string().contains("relationship recovery requires"));
            assert!(
                !preserve_higher_fidelity(&pool, "pending", AnalysisTier::Structural, true, false)
                    .await
                    .unwrap()
            );
        })
        .await;
    }

    #[tokio::test]
    async fn the_gate_reads_content_hash_not_symbol_hash() {
        with_temp_db("gate", Fixtures::Codebase, |pool| async move {
            let fkey = "issue7__gate__lib_rs";
            // A file as stored by a previous run.
            let stored = json!({
                "_key": fkey,
                "path": "src/lib.rs",
                "kind": "file",
                "symbol_hash": "names-did-not-move",
                "content_hash": "the-old-bytes",
                "chunk_count": 2,
                "embedding_count": 2,
            });
            crud::insert_documents(&pool, CODEBASE.files, &[stored], true)
                .await
                .expect("store fixture file node");

            // The comment-only edit: same symbol names, different bytes. The
            // gate must refuse to skip.
            assert_eq!(
                check_unchanged(&pool, fkey, "the-new-bytes", true)
                    .await
                    .expect("gate query"),
                Some(false),
                "a file whose content changed must be re-processed, even though \
                 its symbol names are identical -- this is #7"
            );

            // And an untouched file must still skip, or incrementality is gone.
            assert_eq!(
                check_unchanged(&pool, fkey, "the-old-bytes", true)
                    .await
                    .expect("gate query"),
                Some(true),
                "an unchanged file must still skip"
            );
        })
        .await
    }

    /// Uses a disposable database, including cleanup after assertion failures.
    #[tokio::test]
    async fn test_stamped_go_node_is_reingestable_but_still_guarded() {
        with_temp_db("stamped", Fixtures::Codebase, |pool| async move {
            let config = HadesConfig::default();
            let pid = std::process::id();
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("stamped.go");
            let rel_path = format!("__hades_test193_{pid}/stamped.go");
            let fkey = keys::file_key(&rel_path);
            cleanup_fixture(&pool, &fkey).await;

            // A real Go module: `reenrichment_hatch` requires one, because
            // `group_files_by_go_module` drops module-less files and gopls would
            // never be handed this file otherwise.
            fs::write(dir.path().join("go.mod"), "module example.com/fixture\n").unwrap();
            fs::write(
                &path,
                "package fixture\n\nfunc Stamped(input uint64) uint64 { return input + 1 }\n",
            )
            .unwrap();

            // First ingest: Go has no per-file semantic analyzer, so this lands at
            // `structural` with a tree-sitter serialized digest.
            let mut imports = ImportContext::default();
            let first = ingest_file(
                &pool,
                None,
                &config,
                &path,
                &rel_path,
                None,
                &mut imports,
                None,
                false,
                false,
                false,
                FALLBACK_WINDOW_CHARS,
                "",
            )
            .await
            .expect("first ingest failed");
            assert!(
                first.skipped.is_none_or(|s| !s),
                "first ingest must not skip (error: {:?})",
                first.error
            );

            super::super::codebase_persist::store_relationships(
                &pool,
                std::mem::take(&mut imports.committed_revisions),
                Vec::new(),
            )
            .await
            .expect("complete fixture relationship stage");

            // Reproduce the pre-fix state: the old `store_lsp_extractions` stamped
            // the file node `semantic` while leaving `symbol_hash` tree-sitter's.
            upsert_merge_file_node(
                &pool,
                &fkey,
                json!({ "analysis_tier": "semantic", "analyzer": "gopls" }),
            )
            .await
            .expect("failed to stamp fixture node");

            // The bug: with gopls scheduled, this returned skipped regardless of
            // `--force`, because the guard runs ahead of the force check.
            let mut imports = ImportContext::default();
            let released = ingest_file(
                &pool,
                None,
                &config,
                &path,
                &rel_path,
                None,
                &mut imports,
                None,
                true,
                false,
                true,
                FALLBACK_WINDOW_CHARS,
                "",
            )
            .await
            .expect("re-ingest with gopls scheduled failed");
            assert!(
                released.skipped.is_none_or(|s| !s),
                "a stamped .go node must be re-ingestable when the gopls phase will \
             re-enrich it (error: {:?})",
                released.error
            );

            super::super::codebase_persist::store_relationships(
                &pool,
                std::mem::take(&mut imports.committed_revisions),
                Vec::new(),
            )
            .await
            .expect("complete fixture relationship stage");

            // And the guard must still hold when nothing will restore the purge —
            // re-stamp, then re-ingest with no gopls phase scheduled.
            upsert_merge_file_node(
                &pool,
                &fkey,
                json!({ "analysis_tier": "semantic", "analyzer": "gopls" }),
            )
            .await
            .expect("failed to re-stamp fixture node");
            let mut imports = ImportContext::default();
            let preserved = ingest_file(
                &pool,
                None,
                &config,
                &path,
                &rel_path,
                None,
                &mut imports,
                None,
                true,
                false,
                false,
                FALLBACK_WINDOW_CHARS,
                "",
            )
            .await
            .expect("re-ingest without gopls failed");
            assert_eq!(
                preserved.skipped,
                Some(true),
                "with no re-enrichment scheduled the fidelity guard must still \
             preserve (error: {:?})",
                preserved.error
            );

            cleanup_fixture(&pool, &fkey).await;
        })
        .await;
    }

    /// A re-ingest that produces *fewer* chunks than the previous run must not
    /// leave the old high-index chunk docs behind (#159).
    ///
    /// Chunk inserts are overwrite-by-key, so without an explicit delete the
    /// parsed path kept chunks `N+1..M` from the longer previous version
    /// forever — inflating `chunk_count` and stranding chunks that reference
    /// symbols the pre-write purge had already removed. `--force` could not
    /// clear them, so the graph never converged.
    ///
    /// Runs for every parsed language that owns the parsed path, so a future
    /// language added to `Language` inherits the coverage.
    ///
    /// Requires ArangoDB (skips if the socket is absent, per the integration
    /// test convention). Uses a PID-suffixed fixture path so its keys never
    /// collide with real data, and removes every document it wrote.
    #[tokio::test]
    async fn test_reingest_shrinking_file_leaves_no_orphan_chunks() {
        with_temp_db("shrink", Fixtures::Codebase, |pool| async move {
            let config = HadesConfig::default();
            let pid = std::process::id();

            // (extension, long source generator, short source) per parsed language.
            let cases: Vec<(&str, String, &str)> = vec![
                (
                    "rs",
                    (0..40)
                        .map(|i| {
                            format!(
                                "/// Padded documentation for generated function {i}, long enough \
                             that the chunker emits several chunks for this file.\n\
                             pub fn generated_{i}(input: u64) -> u64 {{\n\
                             \x20   let mut acc = input;\n\
                             \x20   for step in 0..{i}u64 {{ acc = acc.wrapping_add(step); }}\n\
                             \x20   acc\n\
                             }}\n\n"
                            )
                        })
                        .collect(),
                    "pub fn only() -> u64 { 1 }\n",
                ),
                (
                    "go",
                    std::iter::once("package fixture\n\n".to_string())
                        .chain((0..40).map(|i| {
                            format!(
                                "// Padded documentation for generated function {i}, long enough \
                             that the chunker emits several chunks for this file.\n\
                             func Generated{i}(input uint64) uint64 {{\n\
                             \x20   acc := input\n\
                             \x20   for step := 0; step < {i}; step++ {{ acc += uint64(step) }}\n\
                             \x20   return acc\n\
                             }}\n\n"
                            )
                        }))
                        .collect(),
                    "package fixture\n\nfunc Only() uint64 { return 1 }\n",
                ),
            ];

            for (ext, long_src, short_src) in cases {
                let dir = TempDir::new().unwrap();
                let path = dir.path().join(format!("shrink.{ext}"));
                let rel_path = format!("__hades_test159_{pid}/shrink.{ext}");
                let fkey = keys::file_key(&rel_path);

                // Start clean in case this PID's fixture somehow survived.
                cleanup_fixture(&pool, &fkey).await;

                fs::write(&path, &long_src).unwrap();
                let mut imports = ImportContext::default();
                let first = ingest_file(
                    &pool,
                    None,
                    &config,
                    &path,
                    &rel_path,
                    None,
                    &mut imports,
                    None,
                    false,
                    false,
                    false,
                    FALLBACK_WINDOW_CHARS,
                    "",
                )
                .await
                .unwrap_or_else(|e| panic!("first ingest failed for .{ext}: {e}"));
                let after_first = count_by_file_key(&pool, CODEBASE.chunks, &fkey).await;
                assert!(
                    after_first > 1,
                    ".{ext} fixture must produce multiple chunks to exercise the shrink case, \
                 got {after_first}"
                );
                assert_eq!(
                    first.num_chunks.unwrap_or(0) as u64,
                    after_first,
                    ".{ext} first ingest: reported chunk count must match stored docs"
                );

                fs::write(&path, short_src).unwrap();
                let mut imports = ImportContext::default();
                let second = ingest_file(
                    &pool,
                    None,
                    &config,
                    &path,
                    &rel_path,
                    None,
                    &mut imports,
                    None,
                    true,
                    false,
                    false,
                    FALLBACK_WINDOW_CHARS,
                    "",
                )
                .await
                .unwrap_or_else(|e| panic!("second ingest failed for .{ext}: {e}"));
                let after_second = count_by_file_key(&pool, CODEBASE.chunks, &fkey).await;

                // The regression: without the delete, after_second would still equal
                // after_first, because overwrite-by-key only rewrote 0..N.
                assert!(
                    after_second < after_first,
                    ".{ext} shrinking re-ingest left orphan chunks: {after_first} before, \
                 {after_second} after (#159)"
                );
                assert_eq!(
                    second.num_chunks.unwrap_or(0) as u64,
                    after_second,
                    ".{ext} second ingest: reported chunk count must match stored docs (#159)"
                );

                // No chunk may reference a symbol the purge removed (validate #7).
                let dangling = {
                    let aql = "FOR c IN @@chunks FILTER c.file_key == @fk \
                           FOR s IN (c.symbols || []) \
                           FILTER DOCUMENT(CONCAT(@syms_name, '/', s)) == null \
                           COLLECT WITH COUNT INTO n RETURN n";
                    let bind = json!({
                        "@chunks": CODEBASE.chunks,
                        "syms_name": CODEBASE.symbols,
                        "fk": fkey,
                    });
                    hades_core::db::query::query_single(
                        &pool,
                        aql,
                        Some(&bind),
                        ExecutionTarget::Reader,
                    )
                    .await
                    .ok()
                    .flatten()
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
                };
                assert_eq!(
                    dangling, 0,
                    ".{ext} chunks reference removed symbols (#159)"
                );

                cleanup_fixture(&pool, &fkey).await;
            }
        })
        .await;
    }

    #[test]
    fn test_unparsed_language_label() {
        assert_eq!(unparsed_language_label("core/kernels/adamw.cu"), "cuda");
        assert_eq!(unparsed_language_label("k.cuh"), "cuda");
        assert_eq!(unparsed_language_label("src/foo.cpp"), "cpp");
        assert_eq!(unparsed_language_label("a/b.h"), "c");
        assert_eq!(unparsed_language_label("notes.txt"), "other");
        assert_eq!(unparsed_language_label("Makefile"), "other");
    }

    #[test]
    fn test_analysis_fidelity_is_monotonic_without_explicit_downgrade() {
        assert!(should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false,
            false
        ));
        assert!(should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Text,
            false,
            false
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            true,
            false
        ));
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Structural),
            AnalysisTier::Semantic,
            false,
            false
        ));
    }

    /// #193: a language whose semantic artifacts come from a post-loop LSP phase
    /// must not be pinned against re-ingest by its own enrichment.
    ///
    /// Go can only ever offer `Structural` from `ingest_file`, so once a node was
    /// stamped `Semantic` the guard fired on every subsequent run and skipped the
    /// file permanently — `--force` included, since the guard runs ahead of it.
    /// When the gopls phase will re-enrich in this same run, the purge it was
    /// protecting is undone immediately, so preserving buys nothing.
    #[test]
    fn test_pending_lsp_reenrichment_releases_the_fidelity_guard() {
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false,
            true
        ));

        // Without a scheduled phase the guard still holds: a C++ node whose
        // libclang analysis is gone has nothing to restore it this run.
        assert!(should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Structural,
            false,
            false
        ));
    }

    /// The hatch must never release a raw-text write, whatever the caller says.
    ///
    /// `Text` is only reached when the semantic analyzer *and* tree-sitter both
    /// failed, and a file that defeated tree-sitter defeats the LSP phase too —
    /// so the premise the hatch rests on ("re-enrichment undoes the purge") is
    /// false exactly there. Left to the caller, this cost a silent, permanent
    /// loss of structure for a partially-written `.go` file.
    #[test]
    fn test_hatch_never_releases_a_raw_text_downgrade() {
        for stored in [AnalysisTier::Semantic, AnalysisTier::Structural] {
            assert!(
                should_preserve_tier(Some(stored), AnalysisTier::Text, false, true),
                "incoming Text must be preserved against stored {stored:?} even with \
                 re-enrichment scheduled"
            );
        }

        // An explicit downgrade is still the operator's call to make.
        assert!(!should_preserve_tier(
            Some(AnalysisTier::Semantic),
            AnalysisTier::Text,
            true,
            true
        ));
    }

    /// The hatch is Go-only, keyed on the file's own extension.
    ///
    /// Rust must never get it: `rust_ast` produces `Semantic` per file and
    /// rust-analyzer only augments, so releasing the guard would let a source
    /// file `syn` cannot parse fall to tree-sitter and overwrite good semantic
    /// artifacts nothing would restore.
    ///
    /// Calls the production predicate rather than restating it, so changing the
    /// rule breaks this test instead of leaving it asserting a private copy.
    #[test]
    fn test_reenrichment_hatch_is_go_only() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/x\n").unwrap();
        let go = dir.path().join("handler.go");
        fs::write(&go, "package x\n").unwrap();

        assert!(reenrichment_hatch(&go, "internal/server/handler.go", true));
        assert!(
            !reenrichment_hatch(&go, "internal/server/handler.go", false),
            "no gopls phase scheduled means nothing will re-supply the purge"
        );

        for rel in ["src/lib.rs", "pkg/mod.py", "src/engine.cpp", "src/util.h"] {
            let other = dir.path().join(rel.rsplit('/').next().unwrap());
            fs::write(&other, "x\n").unwrap();
            assert!(
                !reenrichment_hatch(&other, rel, true),
                "{rel} has a per-file semantic analyzer and must keep the guard"
            );
        }
    }

    /// A `.go` file in no Go module must NOT get the hatch (#194 review).
    ///
    /// `group_files_by_go_module` silently drops paths with no `go.mod`/`go.work`
    /// ancestor, so gopls is never handed them. They land in neither
    /// `workspaces` nor `workspaces_attempted`, which means the caller's
    /// partial-failure check cannot see them either — releasing the guard would
    /// purge their stored enrichment with nothing to restore it and nothing to
    /// report it.
    #[test]
    fn test_hatch_requires_a_resolvable_go_module() {
        let dir = TempDir::new().unwrap();
        let orphan = dir.path().join("stray.go");
        fs::write(&orphan, "package x\n").unwrap();
        assert!(
            !reenrichment_hatch(&orphan, "stray.go", true),
            "a .go file gopls will never be handed must keep the guard"
        );

        // Same file, once the module it belongs to exists.
        fs::write(dir.path().join("go.mod"), "module example.com/x\n").unwrap();
        assert!(
            reenrichment_hatch(&orphan, "stray.go", true),
            "with a module root present the phase will attempt it"
        );
    }

    /// `--language go` must not hand the hatch to the whole tree.
    ///
    /// The flag forces how a file is *parsed*; it cannot make gopls able to
    /// re-enrich a `.rs` or `.py` file. Keying the hatch on `ingest_file`'s
    /// resolved language got this backwards, because that value IS the override
    /// -- every file resolved to Go and every file lost the guard, with only the
    /// Go grammar's recovery heuristic standing between that and a silent
    /// overwrite of semantic artifacts gopls cannot rebuild.
    #[test]
    fn test_language_override_cannot_widen_the_hatch() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("go.mod"), "module example.com/x\n").unwrap();

        // The predicate never sees the override, so there is nothing to widen.
        for rel in ["lib.rs", "mod.py", "engine.cpp"] {
            let p = dir.path().join(rel);
            fs::write(&p, "x\n").unwrap();
            assert!(
                !reenrichment_hatch(&p, rel, true),
                "{rel} must keep the guard even on a `--language go` run"
            );
        }
        let go = dir.path().join("main.go");
        fs::write(&go, "package main\n").unwrap();
        assert!(
            reenrichment_hatch(&go, "cmd/main.go", true),
            "a real Go file still gets the hatch"
        );
    }

    #[test]
    fn test_only_semantic_languages_use_dedicated_edge_resolvers() {
        for language in [Language::Rust, Language::Python, Language::Cpp] {
            assert!(uses_semantic_relationship_resolver(
                language,
                AnalysisTier::Semantic
            ));
            assert!(!uses_semantic_relationship_resolver(
                language,
                AnalysisTier::Structural
            ));
        }
        assert!(!uses_semantic_relationship_resolver(
            Language::Go,
            AnalysisTier::Semantic
        ));
    }

    #[test]
    fn test_discover_files_unparsed_ext() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("shader.wgsl"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("shader.vert"), "void main() {}\n").unwrap();
        fs::write(dir.path().join("app.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap();

        // Without the allowlist: only the .py is picked up.
        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);

        // Explicitly allowlisted extensions are ingested as raw text.
        let allow: HashSet<String> = ["wgsl", "vert"].iter().map(|s| s.to_string()).collect();
        let files = discover_files(dir.path(), None, &allow).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_discover_files_unparsed_single_file() {
        let dir = TempDir::new().unwrap();
        let shader = dir.path().join("backward.wgsl");
        fs::write(&shader, "fn main() {}\n").unwrap();

        // Single unparsed file is rejected without the allowlist...
        assert!(discover_files(&shader, None, &HashSet::new()).is_err());
        // ...and accepted with it.
        let allow: HashSet<String> = ["wgsl"].iter().map(|s| s.to_string()).collect();
        let files = discover_files(&shader, None, &allow).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_single() {
        let dir = TempDir::new().unwrap();
        let py_file = dir.path().join("test.py");
        fs::write(&py_file, "x = 1\n").unwrap();

        let files = discover_files(&py_file, None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("b.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("c.go"), "package demo\n").unwrap();
        fs::write(dir.path().join("d.cpp"), "void run() {}\n").unwrap();
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap();

        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 4); // all registered languages, not .md
    }

    #[test]
    fn test_discover_files_skips_dirs() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        let git_dir = dir.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("config.py"), "x = 1\n").unwrap();
        let pycache = dir.path().join("__pycache__");
        fs::create_dir(&pycache).unwrap();
        fs::write(pycache.join("mod.py"), "x = 1\n").unwrap();

        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1); // only a.py
    }

    #[test]
    fn test_discover_files_includes_extensionless_shebang_scripts() {
        let dir = tempfile::tempdir().unwrap();
        // Extensionless shell script: `--unparsed-ext` is extension-keyed and so
        // can never name this file (#183).
        fs::write(dir.path().join("deploy-thing"), "#!/bin/bash\necho hi\n").unwrap();
        // Extensionless Python script: should be recognized as Python.
        fs::write(dir.path().join("runner"), "#!/usr/bin/env python3\nx = 1\n").unwrap();
        // Extensionless non-script: no shebang, stays out.
        fs::write(dir.path().join("NOTES"), "just prose\n").unwrap();

        let d = discover_files_detailed(dir.path(), None, &HashSet::new()).unwrap();
        let names: Vec<String> = d
            .files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"deploy-thing".to_string()), "got {names:?}");
        assert!(names.contains(&"runner".to_string()), "got {names:?}");
        assert!(!names.contains(&"NOTES".to_string()), "got {names:?}");

        // And the one that stayed out is REPORTED, not silently dropped.
        let unhandled: Vec<&str> = d.unhandled.iter().map(|u| u.reason).collect();
        assert_eq!(d.unhandled.len(), 1, "{:?}", d.unhandled);
        assert_eq!(unhandled[0], "no extension and no shebang");
    }

    #[test]
    fn test_shebang_sniffing_is_extensionless_only() {
        let dir = tempfile::tempdir().unwrap();
        // An extension-bearing script must NOT be admitted by the shebang path:
        // the ingest loop only sniffs extensionless files, so admitting it here
        // would pass discovery and then fail with "cannot detect language"
        // instead of the actionable unsupported-file-type bail.
        let with_ext = dir.path().join("deploy.sh");
        fs::write(&with_ext, "#!/bin/bash\necho hi\n").unwrap();
        assert!(shebang_of(&with_ext).is_none());

        let without_ext = dir.path().join("deploy-thing");
        fs::write(&without_ext, "#!/bin/bash\necho hi\n").unwrap();
        assert!(shebang_of(&without_ext).is_some());
    }

    #[test]
    fn test_first_line_probe_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        // A newline-free blob must not be read whole during a directory walk.
        let blob = dir.path().join("bigblob");
        fs::write(&blob, "x".repeat(2 * 1024 * 1024)).unwrap();
        let line = first_line(&blob).unwrap();
        assert!(line.len() <= 256, "probe read {} bytes", line.len());
        // And it is correctly not treated as a script.
        assert!(shebang_of(&blob).is_none());
    }

    #[test]
    fn test_discover_files_reports_unhandled_extensions() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.py"), "x = 1\n").unwrap();
        fs::write(dir.path().join("data.json"), "{}\n").unwrap();
        fs::write(dir.path().join("Config.toml"), "k = 1\n").unwrap();

        let d = discover_files_detailed(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(d.files.len(), 1, "only the .py is handled");
        // The two unhandled files are counted with a reason rather than falling
        // outside drift's notion of source entirely — that silence was the bug.
        assert_eq!(d.unhandled.len(), 2, "{:?}", d.unhandled);
        assert!(
            d.unhandled
                .iter()
                .all(|u| u.reason == "no handler for extension")
        );
    }

    #[test]
    fn test_discover_files_language_override() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension

        // Without override: no files found.
        let files = discover_files(dir.path(), None, &HashSet::new()).unwrap();
        assert_eq!(files.len(), 0);

        // With override: extensionless file is included.
        let files = discover_files(dir.path(), Some(Language::Python), &HashSet::new()).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_override_excludes_non_source() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("script"), "x = 1\n").unwrap(); // no extension — included
        fs::write(dir.path().join("readme.md"), "# hi\n").unwrap(); // has extension — excluded
        fs::write(dir.path().join("data.json"), "{}").unwrap(); // has extension — excluded
        fs::write(dir.path().join("real.py"), "x = 1\n").unwrap(); // recognized — included

        let files = discover_files(dir.path(), Some(Language::Python), &HashSet::new()).unwrap();
        assert_eq!(files.len(), 2); // script + real.py, not readme.md or data.json
    }

    /// Helper to create a Python import symbol for tests.
    fn make_import_sym(name: &str, import_type: &str, module: &str) -> Symbol {
        let mut metadata = json!({ "type": import_type, "module": module });
        if import_type == "from_import" {
            metadata["original_name"] = json!(name);
        }
        Symbol {
            name: name.to_string(),
            kind: SymbolKind::Import,
            start_line: 1,
            end_line: 1,
            metadata,
        }
    }

    /// Helper to create a definition symbol for tests.
    fn make_def_sym(name: &str, kind: SymbolKind) -> Symbol {
        Symbol {
            name: name.to_string(),
            kind,
            start_line: 1,
            end_line: 10,
            metadata: json!({}),
        }
    }

    #[test]
    fn test_resolve_python_imports_basic() {
        // core/models.py does `from core.utils import helper`
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec![make_import_sym("helper", "from_import", "core.utils")],
        );
        imports.insert("core/utils.py".to_string(), vec![]);

        // utils.py defines a function called `helper`
        let mut file_symbols = HashMap::new();
        file_symbols.insert(
            "core/models.py".to_string(),
            vec![make_def_sym("Model", SymbolKind::Class)],
        );
        file_symbols.insert(
            "core/utils.py".to_string(),
            vec![make_def_sym("helper", SymbolKind::Function)],
        );

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["resolved"], true);
        assert_eq!(edges[0]["style"], "from_import");
        assert_eq!(edges[0]["source_path"], "core/models.py");
        assert_eq!(edges[0]["symbol_name"], "helper");
        // Should be file→symbol edge
        assert!(
            edges[0]["_to"]
                .as_str()
                .unwrap()
                .contains("codebase_symbols")
        );
    }

    #[test]
    fn test_resolve_python_imports_no_self_edge() {
        let mut imports = HashMap::new();
        imports.insert(
            "core/models.py".to_string(),
            vec![make_import_sym("core.models", "import", "core.models")],
        );

        let file_symbols: HashMap<String, Vec<Symbol>> = HashMap::new();
        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_resolve_python_imports_init_package() {
        let mut imports = HashMap::new();
        imports.insert("core/__init__.py".to_string(), vec![]);
        imports.insert(
            "app.py".to_string(),
            vec![make_import_sym("core", "import", "core")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("core/__init__.py".to_string(), vec![]);
        file_symbols.insert("app.py".to_string(), vec![]);

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["target_path"], "core/__init__.py");
    }

    #[test]
    fn test_resolve_python_imports_dedup() {
        let mut imports = HashMap::new();
        imports.insert(
            "a.py".to_string(),
            vec![
                make_import_sym("b", "import", "b"),
                make_import_sym("b", "import", "b"), // duplicate
            ],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("a.py".to_string(), vec![]);
        file_symbols.insert("b.py".to_string(), vec![]);

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_resolve_python_imports_from_import_symbol_level() {
        // server.py does `from config import EmbeddingConfig`
        let mut imports = HashMap::new();
        imports.insert(
            "server.py".to_string(),
            vec![make_import_sym("EmbeddingConfig", "from_import", "config")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("server.py".to_string(), vec![]);
        file_symbols.insert(
            "config.py".to_string(),
            vec![make_def_sym("EmbeddingConfig", SymbolKind::Class)],
        );

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        // Should target the symbol, not the file
        let to = edges[0]["_to"].as_str().unwrap();
        assert!(
            to.starts_with("codebase_symbols/"),
            "expected symbol edge, got: {to}"
        );
        assert!(to.contains("EmbeddingConfig"));
    }

    #[test]
    fn test_resolve_python_imports_fallback_to_file() {
        // server.py does `from config import SomethingUnknown`
        let mut imports = HashMap::new();
        imports.insert(
            "server.py".to_string(),
            vec![make_import_sym("SomethingUnknown", "from_import", "config")],
        );

        let mut file_symbols = HashMap::new();
        file_symbols.insert("server.py".to_string(), vec![]);
        file_symbols.insert("config.py".to_string(), vec![]); // no symbols defined

        let index = build_python_symbol_index(&file_symbols);
        let edges = resolve_python_imports(&imports, &file_symbols, &index);

        assert_eq!(edges.len(), 1);
        // Should fall back to file→file
        let to = edges[0]["_to"].as_str().unwrap();
        assert!(
            to.starts_with("codebase_files/"),
            "expected file edge fallback, got: {to}"
        );
    }

    // ── Route discovery ─────────────────────────────────────────────────

    /// One walk has to place every file, and say so for the ones it cannot.
    #[test]
    fn discovery_routes_code_documents_and_names_the_rest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).expect("mkdir");
        std::fs::create_dir_all(root.join("docs")).expect("mkdir");
        std::fs::write(root.join("src/lib.rs"), "pub fn a() {}\n").expect("write");
        std::fs::write(root.join("src/app.py"), "x = 1\n").expect("write");
        std::fs::write(root.join("docs/spec.md"), "# spec\n").expect("write");
        std::fs::write(root.join("README.md"), "# readme\n").expect("write");
        std::fs::write(root.join("Makefile"), "all:\n").expect("write");
        std::fs::write(root.join("config.toml"), "[a]\n").expect("write");

        let found = discover_by_route(root, &std::collections::HashSet::new()).expect("discovery");

        let mut code: Vec<String> = found
            .code
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        code.sort();
        assert_eq!(code, vec!["app.py", "lib.rs"]);

        let mut docs: Vec<String> = found
            .documents
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        docs.sort();
        assert_eq!(docs, vec!["README.md", "spec.md"]);

        // Neither pipeline claims these, and both are reported with a reason
        // rather than dropped, which is the hole this discovery closes.
        let mut unrouted: Vec<(String, &str)> = found
            .unrouted
            .iter()
            .map(|u| (u.path.rsplit('/').next().unwrap().to_string(), u.reason))
            .collect();
        unrouted.sort();
        assert_eq!(
            unrouted,
            vec![
                ("Makefile".to_string(), "no extension and no shebang"),
                ("config.toml".to_string(), "no handler for extension"),
            ]
        );
    }

    /// An extension the operator asked to embed without a parser is code, since
    /// the code phase is what ingests it. Reporting it as unrouted while that
    /// phase stored it would be wrong in the direction operators trust.
    #[test]
    fn an_unparsed_extension_counts_as_code_not_unrouted() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\n").expect("write");
        std::fs::write(root.join("notes.sql"), "SELECT 1;\n").expect("write");

        let allow = normalize_unparsed_ext(&["toml".to_string()]);
        let found = discover_by_route(root, &allow).expect("discovery");

        assert_eq!(
            found.code.len(),
            1,
            "the manifest is claimed by the code phase"
        );
        assert!(found.code[0].ends_with("Cargo.toml"));
        assert_eq!(found.unrouted.len(), 1, "the .sql is still declined");
        assert!(found.unrouted[0].path.ends_with("notes.sql"));
    }

    // ── Ingest root resolution ──────────────────────────────────────────

    /// One tree named two ways must produce one set of keys.
    ///
    /// Before the root was canonicalized it produced two. The base was
    /// canonical and the discovered paths were not, so `rel_path_for`'s
    /// `strip_prefix` missed and fell back to the whole path as typed: three
    /// files in `crates/hades-proto` keyed as `build_rs` from an absolute
    /// argument and `crates_hades-proto_build_rs` from a relative one, six
    /// nodes for three files, both halves stamped with the same ingest root.
    ///
    /// A `..` detour rather than a relative path, deliberately: a relative path
    /// resolves against the process cwd, and mutating that races every other
    /// test in the binary.
    #[test]
    fn one_tree_named_two_ways_keys_identically() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let tree = tmp.path().join("tree");
        std::fs::create_dir_all(tree.join("src")).expect("mkdir");
        std::fs::write(tree.join("src/lib.rs"), "pub fn a() {}\n").expect("write lib.rs");
        std::fs::write(tree.join("build.rs"), "fn main() {}\n").expect("write build.rs");

        let unparsed = std::collections::HashSet::new();
        let keys_for = |root: &Path| -> Vec<String> {
            let root = resolve_ingest_root(root).expect("root resolves");
            let base = ingest_base_path(&root);
            let mut keys: Vec<String> = discover_files(&root, None, &unparsed)
                .expect("discovery succeeds")
                .iter()
                .map(|f| file_key_for(&base, f))
                .collect();
            keys.sort();
            keys
        };

        let canonical = keys_for(&tree);
        let detoured = keys_for(&tree.join("..").join("tree"));

        assert_eq!(canonical, detoured, "the same tree keyed two ways");
        assert_eq!(
            canonical,
            vec![
                keys::scoped_file_key(tree.to_str().unwrap(), "build.rs"),
                keys::scoped_file_key(tree.to_str().unwrap(), "src/lib.rs"),
            ],
            "keys must be relative to the ingest root, not to how it was typed"
        );
    }

    #[test]
    fn resolve_ingest_root_reports_a_missing_path_by_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("not-here");
        let err = resolve_ingest_root(&missing).expect_err("must not resolve");
        assert!(err.to_string().contains("not-here"), "{err}");
    }

    // ── Embed window construction ───────────────────────────────────────
    //
    // These cover the arithmetic that silently corrupted a corpus: chunk
    // offsets are bytes, the embedding server reads them as characters, and
    // nothing downstream can tell the difference because pooling is correct
    // over whatever range it is handed.

    #[test]
    fn byte_offsets_equal_char_offsets_for_ascii() {
        let text = "fn main() { println!(\"hi\"); }";
        let got = byte_offsets_to_chars(text, &[(0, 9), (10, text.len())]);
        assert_eq!(got, vec![(0, 9), (10, text.chars().count())]);
    }

    #[test]
    fn byte_offsets_shift_after_a_multibyte_character() {
        // The em-dash is three bytes and one character, so every offset past
        // it differs by two. Passing bytes through unconverted is exactly the
        // drift that pointed chunks at the wrong code.
        let text = "let a = 1; // \u{2014} note\nlet b = 2;";
        let dash_bytes = text.find('\u{2014}').unwrap();
        let after_bytes = dash_bytes + '\u{2014}'.len_utf8();
        let got = byte_offsets_to_chars(text, &[(0, dash_bytes), (after_bytes, text.len())]);

        let dash_chars = text.chars().take_while(|c| *c != '\u{2014}').count();
        assert_eq!(got[0], (0, dash_chars));
        assert_eq!(got[1], (dash_chars + 1, text.chars().count()));
        assert_ne!(
            got[1].0, after_bytes,
            "byte and character offsets must not be treated as interchangeable"
        );
    }

    #[test]
    fn byte_offset_inside_a_multibyte_sequence_rounds_down() {
        let text = "a\u{2014}b";
        // Byte 2 is the middle of the em-dash, which no character starts at.
        // Byte 2 rounds down to the em-dash at character 1, and byte 4 is the
        // start of 'b' at character 2.
        let got = byte_offsets_to_chars(text, &[(2, 4)]);
        assert_eq!(got, vec![(1, 2)]);
    }

    #[test]
    fn window_with_unaligned_offsets_is_skipped_not_panicked() {
        let source = "a\u{2014}b".to_string();
        let mut windows = Vec::new();
        // Byte 2 is inside the em-dash. Slicing here would panic before this
        // guard existed, aborting the whole ingest run.
        push_embed_window(
            &mut windows,
            &source,
            2,
            source.len(),
            vec![(0, 1)],
            vec![0],
            "t.rs",
        );
        assert!(windows.is_empty(), "unaligned window must be dropped");
    }

    /// A window's boundaries are not necessarily consecutive in file order, so
    /// the mapping back has to be a lookup rather than a first index plus the
    /// position within the window.
    ///
    /// The packing loop skips a chunk too large for any window. Before the
    /// indices were carried, `first_chunk_index + position` named the wrong chunk
    /// for every boundary after such a gap: each vector was stored under a later
    /// chunk's key, the counts still matched, and nothing showed it.
    #[test]
    fn a_window_records_the_file_order_index_of_every_boundary() {
        let source = "alpha bravo charlie delta".to_string();
        let mut windows = Vec::new();
        // Chunks 3 and 5 survived; 4 was oversized and skipped between them.
        push_embed_window(
            &mut windows,
            &source,
            0,
            source.len(),
            vec![(0, 5), (6, 11)],
            vec![3, 5],
            "t.rs",
        );
        let w = windows.first().expect("window kept");
        assert_eq!(w.chunk_indices, vec![3, 5]);
        assert_eq!(
            w.chunk_indices.len(),
            w.boundaries.len(),
            "one index per boundary, positionally parallel"
        );
        // What the flatten does: position 1 is chunk 5, not chunk 4.
        assert_eq!(w.chunk_indices.get(1).copied(), Some(5));
        assert_eq!(
            w.chunk_indices.get(2),
            None,
            "a position never sent maps nowhere"
        );
    }

    #[test]
    fn window_with_reversed_offsets_is_skipped() {
        let source = "abcdef".to_string();
        let mut windows = Vec::new();
        push_embed_window(&mut windows, &source, 4, 2, vec![(0, 1)], vec![0], "t.rs");
        assert!(windows.is_empty());
    }

    #[test]
    fn window_boundaries_are_relative_to_the_window() {
        let source = "aaaabbbbcccc".to_string();
        let mut windows = Vec::new();
        push_embed_window(
            &mut windows,
            &source,
            4,
            12,
            vec![(0, 4), (4, 8)],
            vec![3, 4],
            "t.rs",
        );
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].text, "bbbbcccc");
        assert_eq!(windows[0].boundaries, vec![(0, 4), (4, 8)]);
        assert_eq!(windows[0].chunk_indices, vec![3, 4]);
    }

    #[tokio::test]
    async fn enrichment_rejects_partial_metadata_and_stale_preparation() {
        use hades_core::code::lsp::symbols::ExtractedSymbol;
        with_temp_db("enrichment_atomic", Fixtures::Codebase, |pool| async move {
            let namespace = "/isolated";
            let path = "fixture.rs";
            let key = keys::scoped_file_key(namespace, path);
            pool.writer()
                .post(
                    "document/codebase_files",
                    &json!({"_key":key,"path":path,"symbol_count":0,"content_hash":"original"}),
                )
                .await
                .unwrap();
            let before = pool
                .writer()
                .get(&format!("document/codebase_files/{key}"))
                .await
                .unwrap();
            let revisions = HashMap::from([(
                path.to_owned(),
                Some(before["_rev"].as_str().unwrap().to_owned()),
            )]);
            let mut extraction = FileExtraction::empty();
            extraction.symbols.push(ExtractedSymbol {
                name: "target".into(),
                qualified_name: "target".into(),
                kind: "function".into(),
                visibility: "public".into(),
                signature: "fn target()".into(),
                start_line: 0,
                end_line: 1,
                parent_symbol: None,
                impl_trait: None,
                is_pyo3: false,
                is_ffi: false,
                is_unsafe: false,
                derives: Vec::new(),
                python_name: None,
                calls: Vec::new(),
            });
            let extractions = HashMap::from([(path.to_owned(), extraction)]);
            pool.writer().put("collection/codebase_files/properties", &json!({
                "schema":{"level":"strict","rule":{"type":"object","required":["fault_marker"]}}
            })).await.unwrap();
            let failed = store_lsp_extractions(
                &pool,
                extractions.clone(),
                revisions.clone(),
                1,
                1,
                "fixture",
                "fixture",
                namespace,
            )
            .await
            .unwrap();
            assert!(failed.store_failed);
            assert_eq!(failed.symbols, 0);
            assert_eq!(
                pool.writer()
                    .get(&format!("document/codebase_files/{key}"))
                    .await
                    .unwrap(),
                before
            );
            assert_eq!(
                crud::count_collection(&pool, CODEBASE.symbols)
                    .await
                    .unwrap(),
                0
            );
            assert_eq!(
                crud::count_collection(&pool, CODEBASE.defines_edges)
                    .await
                    .unwrap(),
                0
            );
            pool.writer()
                .put(
                    "collection/codebase_files/properties",
                    &json!({"schema":null}),
                )
                .await
                .unwrap();
            let success = store_lsp_extractions(
                &pool,
                extractions.clone(),
                revisions.clone(),
                1,
                1,
                "fixture",
                "fixture",
                namespace,
            )
            .await
            .unwrap();
            assert!(!success.store_failed);
            assert_eq!(success.symbols, 1);
            let committed = pool
                .writer()
                .get(&format!("document/codebase_files/{key}"))
                .await
                .unwrap();
            assert_eq!(committed["fixture_analyzed"], true);
            assert_eq!(committed["symbol_count"], 1);
            let stale = store_lsp_extractions(
                &pool,
                extractions,
                revisions,
                1,
                1,
                "fixture",
                "fixture",
                namespace,
            )
            .await
            .unwrap();
            assert!(stale.store_failed);
            assert_eq!(
                pool.writer()
                    .get(&format!("document/codebase_files/{key}"))
                    .await
                    .unwrap(),
                committed
            );
        })
        .await;
    }
}
