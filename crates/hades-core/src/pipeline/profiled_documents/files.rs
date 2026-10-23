//! Bounded foreground file ingestion under one saved database/profile observation.
use super::*;
use crate::{
    ingest_routing::{Route, route_for},
    persephone::extraction::{ExtractOptions, ExtractionClient},
    source_git::SourceGit,
};
use hades_proto::extraction::SourceType;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Component, Path, PathBuf},
    sync::{Arc, LazyLock},
};
use tokio::sync::Semaphore;

/// Keep source snapshots below the extractor's 16 MiB protobuf message ceiling.
pub const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const PROVENANCE_FORMAT: &str = "hades-file-ingestion-v1";
static ADMISSION: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(4)));

/// Local source intent. Model, task, chunking and storage are never caller overrides.
pub struct FileInput {
    pub path: PathBuf,
    pub root: Option<PathBuf>,
    pub id: Option<String>,
    /// An observation at batch admission, not an atomic Git/content snapshot.
    pub source_git: Option<SourceGit>,
    /// User metadata is stored in its own namespace, never merged into binding fields.
    pub metadata: Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileIdentity {
    source_path: String,
    source_rel: Option<String>,
    custom_id: Option<String>,
}

impl FileIdentity {
    /// Preserve exact names and extensions; root-relative identities survive relocation.
    fn source_id(&self) -> String {
        if let Some(id) = &self.custom_id {
            format!("file-id:{id}")
        } else if let Some(relative) = &self.source_rel {
            format!("relative-file:{relative}")
        } else {
            format!("local-file:{}", self.source_path)
        }
    }

    /// Explicit IDs cannot overwrite another file, even under force.
    pub(super) fn same_source(&self, incoming: &Self) -> bool {
        self.custom_id == incoming.custom_id
            && match (&self.source_rel, &incoming.source_rel) {
                (Some(saved), Some(new)) => saved == new,
                (None, None) => self.source_path == incoming.source_path,
                _ => false,
            }
    }

    /// Refuse ambiguous paths, lossy encodings and invalid source identifiers.
    fn validate(&self, source_id: &str) -> Result<(), ArangoError> {
        let path = Path::new(&self.source_path);
        ensure(
            path.is_absolute()
                && path
                    .components()
                    .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
                && !self.source_path.chars().any(char::is_control)
                && self.source_path.len() <= 4096,
            "invalid canonical source path",
        )?;
        if let Some(relative) = &self.source_rel {
            ensure(
                !relative.is_empty()
                    && Path::new(relative)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)))
                    && !relative.chars().any(char::is_control),
                "invalid root-relative source path",
            )?;
        }
        if let Some(id) = &self.custom_id {
            ensure(!id.trim().is_empty(), "--id must not be empty")?;
        }
        ensure(
            source_id == self.source_id()
                && source_id.len() <= 4096
                && !source_id.chars().any(char::is_control),
            "invalid file source identifier",
        )
    }
}

/// This provenance and user metadata commit with the complete document, never afterward.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FileProvenance {
    format: String,
    pub identity: FileIdentity,
    content_sha256: String,
    content_bytes: usize,
    source_git: Option<SourceGit>,
    extractor_metadata: BTreeMap<String, String>,
    source_type: String,
    metadata: Value,
}

impl FileProvenance {
    /// Validate both newly extracted and persisted provenance without trusting arbitrary fields.
    pub(super) fn validate(&self, source_id: &str) -> Result<(), ArangoError> {
        self.identity.validate(source_id)?;
        ensure(
            self.format == PROVENANCE_FORMAT
                && (1..=MAX_FILE_BYTES).contains(&self.content_bytes)
                && self.content_sha256.len() == 64
                && self
                    .content_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && self.metadata.is_object()
                && SourceType::from_str_name(&self.source_type)
                    .is_some_and(|t| t != SourceType::Code)
                && self
                    .extractor_metadata
                    .get("extractor")
                    .is_none_or(|v| v != "pymupdf_fallback"),
            "invalid file-ingestion provenance",
        )?;
        if let Some(commit) = self.source_git.as_ref().and_then(|git| git.commit.as_ref()) {
            ensure(
                [40, 64].contains(&commit.len()) && commit.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid source Git commit",
            )?;
        }
        ensure(
            serde_json::to_vec(self)?.len() <= MAX_METADATA_BYTES,
            "file-ingestion metadata exceeds 64 KiB",
        )
    }
}

impl ProfiledDocuments {
    /// Extract one bounded file snapshot, then plan/embed and atomically publish it.
    /// Existing sources require explicit replacement. The source revision is pinned
    /// before extraction; failures never fall back to a different model or writer.
    pub async fn ingest_file(
        &self,
        input: FileInput,
        extractor: &ExtractionClient,
        replace: bool,
    ) -> Result<DocumentReceipt, PipelineError> {
        ensure(
            input.metadata.is_object()
                && serde_json::to_vec(&input.metadata)
                    .map_err(ArangoError::from)?
                    .len()
                    <= MAX_METADATA_BYTES,
            "--metadata must be an object within 64 KiB",
        )?;
        ensure(
            route_for(&input.path) == Route::Document,
            "only routed document files are supported by profile-aware ingestion",
        )?;
        let retention = Arc::new(ADMISSION.clone().try_acquire_owned().map_err(|_| {
            invalid("file ingestion is busy; at most four files may be active per process")
        })?);
        self.current_retaining(retention.clone()).await?;
        let reader_retention = retention.clone();
        let (bytes, mut provenance, name) = tokio::task::spawn_blocking(move || {
            let _retention = reader_retention;
            read_input(input)
        })
        .await
        .map_err(|_| invalid("file snapshot worker stopped"))??;
        let source_id = provenance.identity.source_id();
        provenance.validate(&source_id)?;
        let key = digest_key("bd", &json!([source_id]));
        let admission = storage::admit_source(
            self.pool.writer(),
            &key,
            replace,
            self.passage.binding(),
            &self.snapshot,
            Some(&provenance.identity),
        )
        .await?;
        let mut metadata = admission.metadata;
        metadata
            .as_object_mut()
            .expect("validated source metadata")
            .extend(
                provenance
                    .metadata
                    .as_object()
                    .expect("validated input metadata")
                    .clone(),
            );
        provenance.metadata = metadata;
        provenance.validate(&source_id)?;
        self.current_retaining(retention.clone()).await?;
        // The extractor receives exactly the bytes hashed above. It cannot reopen
        // a changed local path or depend on the CLI's working directory.
        let extraction = extractor
            .extract_bytes(&name, bytes, ExtractOptions::default())
            .await?;
        ensure(
            extraction.source_type != SourceType::Code,
            "extractor returned code for a document-only workload",
        )?;
        provenance.extractor_metadata = extraction.metadata.into_iter().collect();
        provenance.source_type = extraction.source_type.as_str_name().into();
        let prepared = self
            .prepare_admitted(
                DocumentInput {
                    source_id,
                    text: extraction.full_text,
                },
                admission.revision,
                admission.owners,
                Some(provenance),
                retention.clone(),
            )
            .await?;
        self.commit_retaining(prepared, retention).await
    }
}

/// Read a regular file through a fixed descriptor with an explicit byte ceiling.
fn read_input(input: FileInput) -> Result<(Vec<u8>, FileProvenance, String), PipelineError> {
    let read = || -> std::io::Result<_> {
        let path = std::fs::canonicalize(&input.path)?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)?;
        Ok((path, file))
    };
    let (path, file) =
        read().map_err(|e| PipelineError::Other(format!("cannot open source file: {e}")))?;
    ensure(
        route_for(&path) == Route::Document,
        "canonical input is not a supported document file",
    )?;
    let info = file
        .metadata()
        .map_err(|e| PipelineError::Other(format!("cannot inspect source file: {e}")))?;
    ensure(
        info.is_file() && info.len() <= MAX_FILE_BYTES as u64,
        "source must be a regular file no larger than 8 MiB; split larger inputs explicitly",
    )?;
    let source_path = path
        .to_str()
        .ok_or_else(|| invalid("source path must be valid UTF-8"))?
        .to_owned();
    let source_rel = input
        .root
        .map(|root| {
            let root = std::fs::canonicalize(root)
                .map_err(|e| invalid(&format!("cannot resolve --root: {e}")))?;
            let relative = path
                .strip_prefix(&root)
                .map_err(|_| invalid("source file is outside --root"))?;
            ensure(root.is_dir(), "--root must be a directory")?;
            Ok::<_, ArangoError>(
                relative
                    .to_str()
                    .ok_or_else(|| invalid("relative path must be valid UTF-8"))?
                    .to_owned(),
            )
        })
        .transpose()?;
    let mut bytes = Vec::new();
    file.take((MAX_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| PipelineError::Other(format!("cannot read source file: {e}")))?;
    ensure(
        !bytes.is_empty() && bytes.len() <= MAX_FILE_BYTES,
        "source snapshot must contain 1..=8388608 bytes",
    )?;
    let provenance = FileProvenance {
        format: PROVENANCE_FORMAT.into(),
        identity: FileIdentity {
            source_path,
            source_rel,
            custom_id: input.id,
        },
        content_sha256: hash(&bytes),
        content_bytes: bytes.len(),
        source_git: input.source_git,
        extractor_metadata: BTreeMap::new(),
        source_type: SourceType::Unknown.as_str_name().into(),
        metadata: input.metadata,
    };
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid("invalid source filename"))?
        .to_owned();
    Ok((bytes, provenance, name))
}
