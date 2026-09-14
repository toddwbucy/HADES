//! Native Rust implementation of the `hades ingest` command.
//!
//! Generic file ingestion: extract → chunk → embed → store. Supports:
//! - Local file ingest (PDF, LaTeX, text, code via the extractor service)
//! - Batch mode with per-document error isolation
//! - Resumable checkpointing with progress reporting
//! - Bounded concurrency and rate limiting
//! - Custom metadata merging
//! - Collection profile selection
//! - Force re-processing (surgical delete + re-insert)
//!
//! Domain-specific orchestration (custom source APIs, application-specific
//! schemas) belongs in user scripts that compose this command + `hades db insert` etc.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tracing::{info, warn};

use hades_core::HadesConfig;
use hades_core::batch::{BatchProcessor, BatchProcessorConfig, RateLimiter};
use hades_core::chunking::{ChunkingStrategy, TokenChunking};
use hades_core::db::ArangoPool;
use hades_core::db::collections::CollectionProfile;
use hades_core::db::keys;
use hades_core::persephone::embedding::EmbeddingClient;
use hades_core::persephone::extraction::ExtractionClient;
use hades_core::pipeline::{Pipeline, PipelineConfig};

use super::codebase_ingest::{self, PhaseOutcome};
use super::output::{self, OutputFormat};

/// Code-file extensions for auto-detecting the `code` embedding task.
const CODE_EXTENSIONS: &[&str] = &[
    "py", "rs", "cu", "cuh", "cpp", "c", "h", "hpp", "js", "ts", "go", "java", "rb", "swift", "kt",
];

/// Keys that user-provided metadata cannot override.
const PROTECTED_KEYS: &[&str] = &["_key", "status", "source"];

/// Ingest command failed with partial results.
///
/// Returned instead of calling `process::exit` so callers control the exit code.
#[derive(Debug, thiserror::Error)]
#[error("{failed} of {total} documents failed to ingest")]
pub struct IngestFailure {
    pub total: usize,
    pub failed: usize,
}

/// Create the document profile's three collections when they are missing.
///
/// `codebase ingest` has always created its own nine collections and its named
/// graph on the fly; the document path did not create its three, so the first
/// document into a fresh database failed with "collection or view not found:
/// documents" *after* extraction and embedding had already run. The work was
/// done and then discarded at the store step, which is the most expensive place
/// to discover a missing collection.
async fn ensure_document_collections(db: &ArangoPool, profile: &CollectionProfile) -> Result<()> {
    let existing = hades_core::db::crud::list_collections(db, false)
        .await
        .context("failed to list collections")?;
    let names: Vec<&str> = existing.iter().map(|c| c.name.as_str()).collect();

    for name in [profile.metadata, profile.chunks, profile.embeddings] {
        if !names.contains(&name) {
            info!(collection = name, "creating collection");
            hades_core::db::crud::create_collection(db, name, Some(2))
                .await
                .with_context(|| format!("failed to create collection: {name}"))?;
        }
    }
    Ok(())
}

/// Ingest a tree: one command, one root, one graph, extension decides.
///
/// The tree used to require two commands and the operator had to know which
/// files belonged to which. That is how mixed trees lost half of themselves:
/// `codebase ingest` reported markdown as "no handler for extension" and
/// `hades ingest` handed a `.py` file to docling. Here the routing table in
/// `hades_core::ingest_routing` decides per file, both phases run against the
/// same root, and one envelope reports what happened to everything including
/// the files nothing claimed.
///
/// Document keys come out root-relative for free, because the root is the path
/// given, which is what stops eleven `README.md` files from sharing one key.
#[allow(clippy::too_many_arguments)]
pub async fn run_unified(
    config: &HadesConfig,
    root: PathBuf,
    force: bool,
    metadata_json: Option<&str>,
    concurrency: Option<usize>,
    unparsed_ext: &[String],
    collection: Option<&str>,
    task: Option<&str>,
) -> Result<()> {
    let cmd_start = Instant::now();
    let root = codebase_ingest::resolve_ingest_root(&root)?;
    let unparsed_set = codebase_ingest::normalize_unparsed_ext(unparsed_ext);
    let found = codebase_ingest::discover_by_route(&root, &unparsed_set)?;

    info!(
        root = %root.display(),
        code = found.code.len(),
        documents = found.documents.len(),
        unrouted = found.unrouted.len(),
        "routed tree by extension"
    );

    // Code first. The two phases write disjoint collections, so the order is
    // only about which failure an operator sees first, and losing the code
    // graph's edges is the more serious of the two.
    let code = if found.code.is_empty() {
        None
    } else {
        Some(
            codebase_ingest::run_phase(
                config,
                root.clone(),
                None,
                false,
                unparsed_ext,
                None,
                force,
                false,
            )
            .await?,
        )
    };

    // A setup failure in the document phase must not take the code phase's
    // summary with it. `?` here discarded minutes of completed analyzer work,
    // symbols, edges and embeddings — all of it durably stored — because the
    // extraction service was down, and printed nothing. Carrying the error into
    // the envelope is the whole reason `PhaseOutcome` exists.
    let mut document_setup_error: Option<String> = None;
    let documents = if found.documents.is_empty() {
        None
    } else {
        match run_phase(
            config,
            found.documents.clone(),
            false,
            metadata_json,
            &[],
            collection,
            force,
            task,
            None,
            false,
            false,
            concurrency,
            Some(root.clone()),
        )
        .await
        {
            Ok(outcome) => Some(outcome),
            Err(e) => {
                warn!(error = %e, "document phase could not start, reporting the code phase");
                document_setup_error = Some(e.to_string());
                None
            }
        }
    };

    let code_failure = code.as_ref().and_then(|o| o.failure.as_ref());
    let doc_failure = documents.as_ref().and_then(|o| o.failure.as_ref());
    let success = code_failure.is_none() && doc_failure.is_none() && document_setup_error.is_none();

    let result_data = json!({
        "root": root.display().to_string(),
        "routed": {
            "code": found.code.len(),
            "documents": found.documents.len(),
            "unrouted": found.unrouted.len(),
        },
        "code": code.as_ref().map(|o| o.data.clone()),
        "documents": documents.as_ref().map(|o| o.data.clone()),
        // Listed, not counted only: a file nothing claims is the hole this
        // command exists to close, so it has to be nameable from stdout.
        "unrouted": found.unrouted,
        "document_phase_error": document_setup_error,
        "duration_ms": cmd_start.elapsed().as_millis(),
    });

    output::print_output_with_success("ingest", result_data, &OutputFormat::Json, success);

    // Both failures are reported in the envelope above; the process exit needs
    // one, and the code phase's is the one that means the graph lost a layer.
    if code_failure.is_some() {
        return Err(code
            .and_then(|o| o.failure)
            .unwrap_or_else(|| anyhow::anyhow!("code phase failed")));
    }
    if doc_failure.is_some() {
        return Err(documents
            .and_then(|o| o.failure)
            .unwrap_or_else(|| anyhow::anyhow!("document phase failed")));
    }
    if let Some(message) = document_setup_error {
        return Err(anyhow::anyhow!(message));
    }
    Ok(())
}

/// Run the document ingest command, printing its own envelope.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    config: &HadesConfig,
    inputs: Vec<PathBuf>,
    batch: bool,
    metadata_json: Option<&str>,
    claims: &[String],
    collection: Option<&str>,
    force: bool,
    task: Option<&str>,
    id: Option<&str>,
    resume: bool,
    reset: bool,
    concurrency: Option<usize>,
    root: Option<PathBuf>,
) -> Result<()> {
    let outcome = run_phase(
        config,
        inputs,
        batch,
        metadata_json,
        claims,
        collection,
        force,
        task,
        id,
        resume,
        reset,
        concurrency,
        root,
    )
    .await?;
    let success = outcome.failure.is_none();
    output::print_output_with_success("ingest", outcome.data, &OutputFormat::Json, success);
    match outcome.failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The document phase itself.
///
/// This is the entry point called from `main.rs` when the user runs
/// `hades ingest ...`.
#[allow(clippy::too_many_arguments)]
pub async fn run_phase(
    config: &HadesConfig,
    inputs: Vec<PathBuf>,
    batch: bool,
    metadata_json: Option<&str>,
    _claims: &[String],
    collection: Option<&str>,
    force: bool,
    task: Option<&str>,
    id: Option<&str>,
    resume: bool,
    reset: bool,
    concurrency: Option<usize>,
    root: Option<PathBuf>,
) -> Result<PhaseOutcome> {
    let cmd_start = Instant::now();

    // -- Validate inputs -------------------------------------------------------
    if inputs.is_empty() && !resume {
        bail!("no inputs provided. Supply file paths to ingest.");
    }

    if id.is_some() && inputs.len() > 1 {
        bail!("--id can only be used with a single input");
    }

    // Parse custom metadata if provided.
    let extra_metadata: Option<Value> = match metadata_json {
        Some(s) => {
            let val: Value = serde_json::from_str(s).context("--metadata must be valid JSON")?;
            if !val.is_object() {
                bail!("--metadata must be a JSON object, got: {}", val);
            }
            Some(val)
        }
        None => None,
    };

    // Resolve collection profile.
    let profile = match collection {
        Some(name) => CollectionProfile::get(name)
            .ok_or_else(|| anyhow::anyhow!("unknown collection profile: {name}"))?,
        None => CollectionProfile::default_profile(),
    };

    // Inputs stay as the caller wrote them at this level: item ids and batch
    // resume keys derive from these strings, and per-document error isolation
    // means a bad path must become a per-item failure, not a batch abort.
    // Canonicalization happens inside ingest_file, per item (#166).
    let file_paths: Vec<PathBuf> = inputs.to_vec();

    // -- Connect to services ---------------------------------------------------
    let db = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    ensure_document_collections(&db, profile).await?;

    let extractor = ExtractionClient::connect_default()
        .await
        .context("failed to connect to extraction service")?;

    let embedder = EmbeddingClient::connect_at(&config.embedding.service.socket)
        .await
        .context("failed to connect to embedding service")?;

    // -- Build pipeline --------------------------------------------------------
    let embed_task = determine_embed_task(task, profile);
    let pipeline_config = PipelineConfig {
        profile,
        embed_task,
        embed_batch_size: Some(config.embedding.batch.size),
        extract_options: Default::default(),
        overwrite: force,
    };

    let pipeline = Arc::new(Pipeline::new(
        extractor,
        embedder,
        db.clone(),
        pipeline_config,
    ));
    let chunker = Arc::new(TokenChunking {
        chunk_size: config.embedding.chunking.size_tokens as usize,
        overlap: config.embedding.chunking.overlap_tokens as usize,
    });

    // -- Configure batch processor ---------------------------------------------
    let batch_concurrency = concurrency
        .unwrap_or(config.batch_processing.concurrency)
        .max(1);

    let rate_limiter = if config.batch_processing.rate_limit_rps > 0.0 {
        Some(Arc::new(RateLimiter::new(
            config.batch_processing.rate_limit_rps,
            config.batch_processing.rate_limit_retries,
        )))
    } else {
        None
    };

    let batch_config = BatchProcessorConfig {
        concurrency: batch_concurrency,
        state_file: if batch || resume || file_paths.len() > 1 {
            Some(PathBuf::from(".hades-batch-state.json"))
        } else {
            None
        },
        resume,
        reset,
        progress_interval: Duration::from_secs_f64(config.batch_processing.progress_interval_secs),
        rate_limiter,
    };

    let processor = BatchProcessor::new(batch_config);

    // -- Build items for batch processor ---------------------------------------
    let custom_id: Option<Arc<str>> = id.map(Arc::from);
    let extra_metadata = extra_metadata.map(Arc::new);

    let items: Vec<(String, PathBuf)> = file_paths
        .into_iter()
        .map(|path| {
            let item_id = path.display().to_string();
            (item_id, path)
        })
        .collect();

    // -- Process batch ---------------------------------------------------------
    let summary = processor
        .process(items, move |_item_id, path| {
            let pipeline = pipeline.clone();
            let chunker = chunker.clone();
            let db = db.clone();
            let custom_id = custom_id.clone();
            let extra_metadata = extra_metadata.clone();
            let root = root.clone();

            async move {
                ingest_file(
                    &pipeline,
                    chunker.as_ref(),
                    &db,
                    profile,
                    &path,
                    force,
                    extra_metadata.as_deref(),
                    custom_id.as_deref(),
                    root.as_deref(),
                )
                .await
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("batch processing error: {e}"))?;

    // -- Output summary --------------------------------------------------------
    let duration_ms = cmd_start.elapsed().as_millis() as u64;

    let result_values: Vec<Value> = summary
        .results
        .iter()
        .map(|r| {
            let mut val = json!({
                "input": r.item_id,
                "success": r.success,
                "duration_ms": r.duration_ms,
            });
            if r.skipped == Some(true) {
                val["skipped"] = json!(true);
            }
            // Merge domain-specific fields from the process_fn result.
            if let Some(ref data) = r.data
                && let Some(obj) = data.as_object()
            {
                for (k, v) in obj {
                    val[k] = v.clone();
                }
            }
            if let Some(ref err) = r.error {
                val["error"] = json!(err.message);
            }
            val
        })
        .collect();

    // Count skipped from both checkpoint resume and database-exists checks.
    let skipped = summary.skipped
        + summary
            .results
            .iter()
            .filter(|r| {
                r.skipped != Some(true)
                    && r.data
                        .as_ref()
                        .and_then(|d| d.get("skipped"))
                        .and_then(|v| v.as_bool())
                        == Some(true)
            })
            .count();

    let result_data = json!({
        "total": summary.total,
        "completed": summary.completed,
        "failed": summary.failed,
        "skipped": skipped,
        "results": result_values,
        "duration_ms": duration_ms,
    });

    // The envelope's success reflects item outcomes, not merely "the batch
    // ran" — an all-items-failed run must not print success: true (#166).
    let failure = (summary.failed > 0).then(|| {
        anyhow::Error::from(IngestFailure {
            total: summary.total,
            failed: summary.failed,
        })
    });

    Ok(PhaseOutcome {
        data: result_data,
        failure,
    })
}

// ── Local file ingest ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn ingest_file(
    pipeline: &Pipeline,
    chunker: &(dyn ChunkingStrategy + Send + Sync),
    db: &ArangoPool,
    profile: &CollectionProfile,
    path: &Path,
    force: bool,
    extra_metadata: Option<&Value>,
    custom_id: Option<&str>,
    root: Option<&Path>,
) -> Result<Value> {
    // Identity FIRST, from the path as the caller wrote it: the key comes from
    // the caller's own path, so a symlinked input keeps the name the caller used
    // rather than silently adopting the target's.
    let doc_key = derive_doc_key(path, custom_id, root);

    // THEN canonicalize for everything that crosses a process boundary. The
    // extraction service runs as its own user in its own working directory, so
    // a relative path that resolves here fails there with a misleading
    // service-side "File not found" (#166). A missing input fails HERE, per
    // item — the batch continues and the failure reaches the summary and the
    // envelope like any other item error.
    let path: PathBuf = std::fs::canonicalize(path)
        .with_context(|| format!("input path not found or unreadable: {}", path.display()))?;
    let path = path.as_path();

    // Who already holds this key, and is it this same file?
    //
    // The skip below is right for re-ingesting one document and wrong for two
    // different documents that happen to share a key, and the key alone cannot
    // tell them apart. Comparing the stored `source_path` can. Without this, the
    // second file is reported as skipped inside a successful run, which is how
    // 3 of this repository's 17 markdown files vanished on their first ingest.
    //
    // Checked even under `--force`, because there the collision is worse: force
    // overwrites, so a different document would be replaced rather than skipped.
    let canonical_input = path.display().to_string();
    // Identity is the path relative to the root when there is one, because that
    // is what the key is derived from. Comparing absolute paths made a relocated
    // or bind-mounted corpus a hard error on every file, with no override, since
    // this check deliberately runs under `--force` as well.
    let incoming_rel = root.and_then(|r| {
        std::fs::canonicalize(r)
            .ok()
            .and_then(|r| path.strip_prefix(&r).ok().map(|p| p.display().to_string()))
    });
    if let Some((stored_rel, stored_path)) =
        existing_source_identity(db, profile.metadata, &doc_key).await?
    {
        // A collision needs two identities of the same kind to compare. When the
        // stored document predates `source_rel` and this run has one, the
        // absolute paths are not evidence of anything, so it is treated as the
        // same document and re-ingested rather than refused forever.
        let collision = match (&stored_rel, &incoming_rel) {
            (Some(stored), Some(incoming)) => stored != incoming,
            (None, None) => stored_path
                .as_ref()
                .is_some_and(|stored| stored != &canonical_input),
            _ => false,
        };
        if collision {
            let stored_shown = stored_rel
                .clone()
                .or_else(|| stored_path.clone())
                .unwrap_or_else(|| "<unknown>".into());
            let incoming_shown = incoming_rel
                .clone()
                .unwrap_or_else(|| canonical_input.clone());
            bail!(
                "document key '{doc_key}' is already held by a different file.\n  \
                 stored: {stored_shown}\n  incoming: {incoming_shown}\n\
                 Two files reduce to the same key. With --root that means two paths \
                 that normalize alike; without it, two files sharing a stem in \
                 different directories. Pass --id to name one of them explicitly."
            );
        }
        if !force {
            info!(
                doc_key,
                "already ingested, skipping (use --force to re-process)"
            );
            return Ok(json!({"skipped": true}));
        }
    }

    // Process through pipeline.
    let result = pipeline.process_document(path, &doc_key, chunker).await;

    if !result.success {
        bail!(
            "{}",
            result
                .error
                .unwrap_or_else(|| "unknown pipeline error".into())
        );
    }

    // Update metadata with source info + extra metadata.
    let mut file_meta = json!({
        "source": "local",
        "source_path": path.display().to_string(),
        // The identity the collision check compares, stable across a move of the
        // tree. Absent when the caller named files rather than a root.
        "source_rel": incoming_rel,
        "status": "PROCESSED",
    });

    // Detect code files and tag them.
    if is_code_file(path) {
        file_meta["pipeline"] = json!("code");
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            file_meta["file_type"] = json!(format!("{ext}_source"));
        }
    }

    // Merge user-provided extra metadata, skipping protected keys.
    merge_extra_metadata(&mut file_meta, extra_metadata);
    file_meta["_key"] = json!(doc_key);

    if let Err(e) =
        hades_core::db::crud::update_document(db, profile.metadata, &doc_key, &file_meta).await
    {
        warn!(doc_key, error = %e, "failed to update file metadata");
    }

    Ok(json!({
        "num_chunks": result.chunk_count,
    }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Merge user-provided extra metadata into a document, skipping protected keys.
fn merge_extra_metadata(doc: &mut Value, extra: Option<&Value>) {
    if let Some(extra) = extra
        && let Some(obj) = extra.as_object()
    {
        for (k, v) in obj {
            if PROTECTED_KEYS.contains(&k.as_str()) {
                warn!(key = k, "ignoring protected key in user metadata");
                continue;
            }
            doc[k] = v.clone();
        }
    }
}

/// Check if a document key already exists in a collection.
/// The `source_path` of the document already stored under `doc_key`, if any.
///
/// Returns `Ok(None)` when nothing holds the key. `Ok(Some(None))` means a
/// document is there but predates `source_path` being recorded, which cannot be
/// compared and so is treated as the same document.
async fn existing_source_identity(
    db: &ArangoPool,
    collection: &str,
    doc_key: &str,
) -> Result<Option<(Option<String>, Option<String>)>> {
    match hades_core::db::crud::get_document(db, collection, doc_key).await {
        Ok(doc) => {
            let field = |name: &str| doc.get(name).and_then(|v| v.as_str()).map(str::to_string);
            Ok(Some((field("source_rel"), field("source_path"))))
        }
        Err(e) if e.is_not_found() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// The document key for an input, and where it came from.
///
/// Two derivations, because the command serves two shapes of input. A single
/// paper is identified by its own name, and a tree of files is identified by
/// position within the tree.
///
/// **The stem alone is not unique in a tree**, which cost 3 of 17 documents on
/// this repository's own first ingest and would have cost 10 of 101 on
/// WeaverTools: 11 files named `README.md` all key as `README`, the first one
/// wins, and the rest hit the already-ingested branch and are reported as
/// skipped inside a run whose envelope says success. `--root` keys by the path
/// relative to that directory instead, the way `codebase ingest` has always
/// keyed files, so every file in a tree gets its own identity.
fn derive_doc_key(path: &Path, custom_id: Option<&str>, root: Option<&Path>) -> String {
    if let Some(id) = custom_id {
        return keys::normalize_document_key(id);
    }
    if let Some(root) = root {
        // Canonicalize both sides or the prefix will not strip: the root comes
        // from the command line and the input may be relative to somewhere else.
        let canonical_root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let canonical_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if let Ok(rel) = canonical_path.strip_prefix(&canonical_root) {
            // Drop the extension, keep the directories: docs/a/spec.md becomes
            // docs_a_spec. Two files differing only by extension in one
            // directory would still collide, which is a narrower case than the
            // one this fixes and is visible as a collision rather than silent.
            let rel = rel.with_extension("");
            return keys::normalize_document_key(&rel.to_string_lossy());
        }
        warn!(
            path = %path.display(),
            root = %root.display(),
            "input is not under --root, keying by file stem instead"
        );
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    keys::normalize_document_key(stem)
}

/// The embedding task for this ingest: the explicit `--task`, else the profile's.
///
/// It used to be guessed from file extensions, which crossed vector spaces: a
/// `.py` file ingested as a document was embedded with the `code` adapter into
/// the `default` profile, whose queries use `retrieval.query`. Measured on this
/// repository's own graph, that mismatch costs about a third of the separation
/// between the top hit and the median. The profile owns both sides now, so they
/// cannot drift.
///
/// `--task` still overrides, because a caller who names an adapter has said what
/// they want. Nothing records which adapter a stored corpus used, so overriding
/// it is the caller's problem to keep track of.
fn determine_embed_task(task: Option<&str>, profile: &CollectionProfile) -> String {
    task.unwrap_or(profile.passage_task).to_string()
}

/// Check if a file path looks like a code file by extension.
fn is_code_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| CODE_EXTENSIONS.contains(&ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this exists for: eleven `README.md` files, one key.
    #[test]
    fn root_relative_keys_separate_same_named_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        for dir in ["docs", "deploy/systemd", "seeds"] {
            std::fs::create_dir_all(root.join(dir)).expect("mkdir");
            std::fs::write(root.join(dir).join("README.md"), "# readme\n").expect("write");
        }

        let keys: Vec<String> = ["docs", "deploy/systemd", "seeds"]
            .iter()
            .map(|dir| derive_doc_key(&root.join(dir).join("README.md"), None, Some(root)))
            .collect();

        assert_eq!(
            keys,
            vec![
                "docs_README".to_string(),
                "deploy_systemd_README".to_string(),
                "seeds_README".to_string(),
            ]
        );

        // Without a root they all collapse, which is the behaviour that dropped
        // 3 of 17 documents. Pinned so the difference stays visible.
        let stems: Vec<String> = ["docs", "deploy/systemd", "seeds"]
            .iter()
            .map(|dir| derive_doc_key(&root.join(dir).join("README.md"), None, None))
            .collect();
        assert_eq!(stems, vec!["README".to_string(); 3]);
    }

    #[test]
    fn custom_id_wins_over_both_derivations() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("paper.md"), "x").expect("write");
        assert_eq!(
            derive_doc_key(
                &tmp.path().join("paper.md"),
                Some("2501.12345v2"),
                Some(tmp.path())
            ),
            "2501_12345"
        );
    }

    #[test]
    fn input_outside_the_root_falls_back_to_the_stem() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let other = tempfile::tempdir().expect("tempdir");
        std::fs::write(other.path().join("elsewhere.md"), "x").expect("write");
        assert_eq!(
            derive_doc_key(&other.path().join("elsewhere.md"), None, Some(tmp.path())),
            "elsewhere"
        );
    }
}
