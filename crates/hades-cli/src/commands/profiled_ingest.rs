//! Foreground CLI wiring for the shared, profile-bound file pipeline.
use super::*;
use hades_core::{
    ingest_routing::{Route, route_for},
    pipeline::profiled_documents::{ProfiledDocuments, files::FileInput},
};

/// Reject unpinned options before provider/extractor startup and avoid legacy checkpoints.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    config: &HadesConfig,
    db: ArangoPool,
    inputs: Vec<PathBuf>,
    profile: &CollectionProfile,
    metadata: Option<Value>,
    claims: &[String],
    force: bool,
    task: Option<&str>,
    id: Option<&str>,
    batch: bool,
    resume: bool,
    reset: bool,
    concurrency: Option<usize>,
    root: Option<PathBuf>,
    provenance: Arc<hades_core::source_git::Batch>,
) -> Result<PhaseOutcome> {
    let start = Instant::now();
    if batch || resume || reset {
        bail!(
            "model-bound ingestion cannot use --batch, --resume or --reset until checkpoints carry profile pins; pass explicit files without these flags"
        );
    }
    if task.is_some() || !claims.is_empty() {
        bail!(
            "model-bound document ingestion does not accept --task or --claims; its embedding task comes from the saved profile"
        );
    }
    if (profile.metadata, profile.chunks, profile.embeddings)
        != ("documents", "chunks", "embeddings")
    {
        bail!("model-bound ingestion currently supports the documents collection profile only");
    }
    for path in &inputs {
        if route_for(path) != Route::Document {
            bail!(
                "{} is disabled_by_profile for document ingestion; code and unrouted files need their own supported workload",
                path.display()
            );
        }
    }
    let concurrency = concurrency.unwrap_or(config.batch_processing.concurrency);
    if !(1..=4).contains(&concurrency) {
        bail!("model-bound document ingestion requires concurrency 1..=4");
    }
    let metadata = metadata.unwrap_or_else(|| json!({}));
    if serde_json::to_vec(&metadata)?.len() > 64 * 1024 {
        bail!("--metadata must fit within 64 KiB");
    }
    let routes = config.registered_providers.as_ref().context(
        "model-bound or initializing workspace requires registered_providers for document ingestion",
    )?;
    let catalog = routes.catalog()?;
    let documents = Arc::new(match &config.ingest_binding {
        Some(hades_core::ingest_jobs::Binding::Profiled(pin)) => {
            ProfiledDocuments::connect_pinned(db, &catalog, pin).await?
        }
        Some(hades_core::ingest_jobs::Binding::Legacy) => {
            bail!("legacy ingestion job cannot adopt a model-bound workspace; start a new job")
        }
        None => ProfiledDocuments::connect(db, &catalog).await?,
    });
    // No extraction connection until database policy and the actual loaded model agree.
    let extractor = ExtractionClient::connect_at(&config.extraction.service.socket).await?;
    provenance.prepare(&inputs).await;
    let envelope_git = if let Some(root) = root.as_deref() {
        provenance.resolve(root).await?
    } else {
        None
    };
    // Retrying an uncertain commit is an operator decision. No automatic retry or
    // legacy path-only checkpoint may turn a failed write into silent replacement.
    let processor = BatchProcessor::new(BatchProcessorConfig {
        concurrency,
        state_file: None,
        resume: false,
        reset: false,
        progress_interval: Duration::from_secs(1),
        rate_limiter: None,
    });
    let id = id.map(str::to_owned);
    let summary = processor
        .process(
            inputs
                .into_iter()
                .map(|path| (path.display().to_string(), path))
                .collect(),
            move |_item, path| {
                let documents = documents.clone();
                let extractor = extractor.clone();
                let provenance = provenance.clone();
                let root = root.clone();
                let id = id.clone();
                let metadata = metadata.clone();
                async move {
                    let source_git = provenance.resolve(&path).await?;
                    let receipt = documents
                        .ingest_file(
                            FileInput {
                                path,
                                root,
                                id,
                                source_git: source_git.clone(),
                                metadata,
                            },
                            &extractor,
                            force,
                        )
                        .await?;
                    Ok(json!({
                        "document_key":receipt.document_key, "num_chunks":receipt.chunks,
                        "embedding_binding":receipt.binding, "source_git":source_git,
                        "embedding_profile_state":"ready"
                    }))
                }
            },
        )
        .await?;
    let mut outcome = batch_outcome(summary, envelope_git, start)?;
    outcome.data["embedding_profile_state"] = json!("ready");
    if let Some(hades_core::ingest_jobs::Binding::Profiled(pin)) = &config.ingest_binding {
        outcome.data["profile_pin"] = json!(pin);
    }
    Ok(outcome)
}
