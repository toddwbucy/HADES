//! Read-only bridge to the protected local registration format written by Python.
//! Descriptors anchor child opens; a shared nonblocking flock interoperates with
//! ModelRegistry without holding its lock while a provider loads large weights.

use std::{
    ffi::{CString, OsStr},
    fs::{File, OpenOptions},
    io::Read,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Component, Path},
};

use serde::{Deserialize, Serialize};

use super::super::embedding::EmbeddingError;
use crate::embedding_profile::{ArtifactManifest, MAX_MANIFEST_BYTES, Md5Digest, label, require};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Registration {
    format: RegistrationFormat,
    pub provider_id: String,
    pub adapter: String,
    pub model_id: String,
    pub revision: String,
    pub manifest: ArtifactManifest,
    pub manifest_md5: Md5Digest,
    snapshot: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
enum RegistrationFormat {
    #[serde(rename = "hades-model-registration-v1")]
    V1,
}

/// Operator-approved identity only; inspection does not load or verify a provider.
#[derive(Debug, Clone, Serialize)]
pub struct RegistrationSummary {
    pub provider_id: String,
    pub adapter: String,
    pub model_id: String,
    pub revision: String,
    pub manifest_md5: Md5Digest,
    pub files: usize,
    pub loaded_verified: bool,
}

/// Detect replacement or a change-and-restore while observing the remote runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Registered {
    pub value: Registration,
    directories: Vec<(u64, u64)>,
    record: (u64, u64, u64, i64, i64, i64, i64),
}

impl Registered {
    /// Return portable registration metadata without installation paths or a serving receipt.
    pub fn summary(&self) -> RegistrationSummary {
        let value = &self.value;
        RegistrationSummary {
            provider_id: value.provider_id.clone(),
            adapter: value.adapter.clone(),
            model_id: value.model_id.clone(),
            revision: value.revision.clone(),
            manifest_md5: value.manifest_md5.clone(),
            files: value.manifest.files.len(),
            loaded_verified: false,
        }
    }
}

/// Match the registrar's bounded, path-free provider keys.
pub(super) fn provider_id(value: &str) -> Result<(), EmbeddingError> {
    require(
        !value.is_empty()
            && value.len() <= 64
            && value.as_bytes()[0].is_ascii_lowercase()
            && value
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'_' | b'-')),
        "provider ID must match [a-z][a-z0-9_-]{0,63}",
    )?;
    Ok(())
}

/// Check this service account's ownership rather than trusting path spelling.
fn owned(info: &std::fs::Metadata) -> bool {
    // SAFETY: geteuid has no arguments or memory ownership requirements.
    info.uid() == unsafe { libc::geteuid() }
}

/// Validate a private directory after opening it without following its final link.
fn directory(file: &File, frozen: bool) -> Result<std::fs::Metadata, EmbeddingError> {
    let info = file.metadata()?;
    require(
        info.is_dir()
            && owned(&info)
            && if frozen {
                info.mode() & 0o7277 == 0
            } else {
                info.mode() & 0o7777 == 0o700
            },
        "registry directories must be private and owned by this account; artifacts must be read-only",
    )?;
    Ok(info)
}

/// Open a fixed child relative to an already checked descriptor, refusing links and FIFOs.
fn child(
    parent: &File,
    name: impl AsRef<OsStr>,
    is_directory: bool,
) -> Result<File, EmbeddingError> {
    let name = CString::new(name.as_ref().as_encoded_bytes())
        .map_err(|_| EmbeddingError::InvalidResponse("invalid registry path".into()))?;
    let flags = libc::O_RDONLY
        | libc::O_CLOEXEC
        | libc::O_NOFOLLOW
        | libc::O_NONBLOCK
        | if is_directory { libc::O_DIRECTORY } else { 0 };
    // SAFETY: parent owns a live descriptor; name is NUL-terminated; no creation flag is used.
    let fd = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful openat returned a new exclusively owned descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// Require private, single-link regular metadata before reading or locking it.
fn metadata_file(file: &File) -> Result<std::fs::Metadata, EmbeddingError> {
    let info = file.metadata()?;
    require(
        info.is_file() && owned(&info) && info.mode() & 0o7077 == 0 && info.nlink() == 1,
        "registration metadata must be private regular files owned by this account",
    )?;
    Ok(info)
}

/// Read one complete protected record; never initialize directories, adopt digests, or load weights.
pub(super) fn read(root: &Path, id: &str) -> Result<Registered, EmbeddingError> {
    provider_id(id)?;
    let mut root_file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")?;
    for component in root.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => root_file = child(&root_file, name, true)?,
            _ => {
                return Err(EmbeddingError::InvalidResponse(
                    "registry path must be absolute without traversal".into(),
                ));
            }
        }
    }
    let root = root_file;
    let root_info = directory(&root, false)?;
    let records = child(&root, "records", true)?;
    let artifacts = child(&root, "artifacts", true)?;
    let sessions = child(&root, "sessions", true)?;
    let infos = [&records, &artifacts, &sessions].map(|f| directory(f, false));
    let mut directories = vec![(root_info.dev(), root_info.ino())];
    for info in infos {
        let info = info?;
        directories.push((info.dev(), info.ino()));
    }
    let lock = child(&root, ".lock", false)?;
    metadata_file(&lock)?;
    // SAFETY: lock owns a live descriptor. Drop releases the flock on every return path.
    require(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } == 0,
        "model registry is busy or cannot be locked; retry explicitly",
    )?;
    let record = child(&records, format!("{id}.json"), false)?;
    let before = metadata_file(&record)?;
    require(
        before.len() <= (2 * MAX_MANIFEST_BYTES) as u64,
        "registration exceeds 2 MiB",
    )?;
    let mut encoded = Vec::new();
    (&record)
        .take((2 * MAX_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut encoded)?;
    require(
        encoded.len() <= 2 * MAX_MANIFEST_BYTES,
        "registration exceeds 2 MiB",
    )?;
    let after = metadata_file(&record)?;
    let stamp = |m: &std::fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    require(
        stamp(&before) == stamp(&after),
        "registration changed while reading",
    )?;
    let value: Registration = serde_json::from_slice(&encoded)
        .map_err(|e| EmbeddingError::InvalidResponse(format!("invalid model registration: {e}")))?;
    require(
        value.provider_id == id,
        "registration names a different provider",
    )?;
    for item in [&value.adapter, &value.model_id, &value.revision] {
        label(item)?;
    }
    require(
        value.snapshot.len() == 38
            && value.snapshot.starts_with("model_")
            && value.snapshot[6..]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid registered artifact snapshot",
    )?;
    value.manifest.check_digest(&value.manifest_md5)?;
    let snapshot = child(&artifacts, &value.snapshot, true)?;
    let info = directory(&snapshot, true)?;
    directories.push((info.dev(), info.ino()));
    Ok(Registered {
        value,
        directories,
        record: stamp(&after),
    })
}
