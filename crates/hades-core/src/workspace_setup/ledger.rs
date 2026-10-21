//! Private, descriptor-relative durable receipts. Atomic rename plus file/directory
//! fsync precede database effects. One nonblocking process lock serializes setup.
use super::{SetupError, SetupReceipt, SetupRequest, SetupStatus, SetupStep, ensure, token};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Component, Path, PathBuf},
    sync::Arc,
};

const LIMIT: usize = 2 * 1024 * 1024;

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Scope {
    pub format: String,
    pub installation: String,
    pub endpoint_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    format: String,
    pub owner: String,
    pub request: SetupRequest,
    pub receipt: SetupReceipt,
    pub database_id: Option<String>,
}

/// A small durable name reservation; never inferred by scanning accessible databases.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    format: String,
    owner: String,
    operation_id: String,
    request_sha256: String,
    database: String,
}

struct Guard {
    root: File,
    lock: File,
}
impl Drop for Guard {
    fn drop(&mut self) {
        // A child may inherit the open file description before CLOEXEC takes effect.
        // Release the lease when its final owner/worker finishes, even if that child
        // has not exec'd yet; merely closing our descriptor could leave it locked.
        // SAFETY: the final Arc owns this live descriptor and no ledger work remains.
        unsafe { libc::flock(self.lock.as_raw_fd(), libc::LOCK_UN) };
    }
}
pub(super) struct Lease {
    guard: Arc<Guard>,
}

impl Lease {
    /// Acquire a process-wide lease without blocking an async worker on filesystem sync.
    pub async fn open(root: PathBuf, scope: Scope) -> Result<Self, SetupError> {
        blocking(move || {
            absolute(&root)?;
            let mut directory = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
                .open("/")?;
            for component in root.components() {
                if let Component::Normal(name) = component {
                    let name = CString::new(name.as_encoded_bytes())
                        .map_err(|_| SetupError::Invalid("invalid ledger path".into()))?;
                    directory = open(&directory, &name, libc::O_RDONLY | libc::O_DIRECTORY)?;
                }
            }
            let info = directory.metadata()?;
            ensure(
                info.is_dir() && owned(&info) && info.mode() & 0o7777 == 0o700,
                "ledger root must be an existing private 0700 directory owned by this account",
            )?;
            let lock = child(&directory, ".lock", libc::O_RDWR | libc::O_CREAT)?;
            regular(&lock)?;
            // SAFETY: lock owns the live descriptor; Guard releases the acquired lease.
            if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return Err(SetupError::Busy);
            }
            let guard = Arc::new(Guard {
                root: directory,
                lock,
            });
            if let Some(saved) = read::<Scope>(&guard, "scope.json")? {
                ensure(
                    saved == scope,
                    "ledger belongs to a different installation or endpoint",
                )?;
            } else {
                write(&guard, "scope.json", &scope)?;
            }
            Ok(Self { guard })
        })
        .await
    }

    /// Read bounded, strictly typed data and cross-check the receipt against its confirmed request.
    pub async fn read(&self, operation: &str) -> Result<Option<Record>, SetupError> {
        token(operation)?;
        let name = format!("op_{operation}.json");
        let operation = operation.to_owned();
        let guard = self.guard.clone();
        blocking(move || {
            let saved: Option<Record> = read(&guard, &name)?;
            if let Some(record) = &saved {
                record.request.validate()?;
                token(&record.owner)?;
                ensure(
                    record.format == "hades-setup-record-v1"
                        && record.request.operation_id == operation
                        && record.receipt.operation_id == operation
                        && record.receipt.database == record.request.database
                        && record.receipt.request_sha256 == record.request.digest()?,
                    "invalid setup record identity",
                )?;
                ensure(
                    record.database_id.as_ref().is_none_or(|id| {
                        !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_digit())
                    }),
                    "invalid recorded database identity",
                )?;
                let before_ack =
                    matches!(record.receipt.step, SetupStep::Validate | SetupStep::Create);
                ensure(
                    before_ack == record.database_id.is_none(),
                    "setup checkpoint and database acknowledgement disagree",
                )?;
                ensure(
                    (record.receipt.status == SetupStatus::Ready)
                        == (record.receipt.step == SetupStep::Complete),
                    "setup readiness checkpoint is inconsistent",
                )?;
                let legal_status = match record.receipt.status {
                    SetupStatus::Validating => record.receipt.step == SetupStep::Validate,
                    SetupStatus::Creating => record.receipt.step == SetupStep::Create,
                    SetupStatus::Initializing => {
                        !before_ack && record.receipt.step != SetupStep::Complete
                    }
                    SetupStatus::Failed => {
                        !matches!(record.receipt.step, SetupStep::Create | SetupStep::Complete)
                    }
                    SetupStatus::Ready | SetupStatus::NeedsReconciliation => true,
                };
                ensure(legal_status, "setup status and checkpoint disagree")?;
                ensure(
                    match record.receipt.status {
                        SetupStatus::Failed | SetupStatus::NeedsReconciliation => {
                            record.receipt.diagnostic.as_deref().is_some_and(|code| {
                                [
                                    "provider_verification_failed",
                                    "database_operation_failed",
                                    "ownership_or_metadata_uncertain",
                                    "setup_invariant_or_ledger_failed",
                                ]
                                .contains(&code)
                            })
                        }
                        _ => record.receipt.diagnostic.is_none(),
                    },
                    "invalid setup diagnostic",
                )?;
            }
            Ok(saved)
        })
        .await
    }

    /// Reserve operation and database before creation, refusing changed requests or another owner.
    pub async fn claim(&self, request: SetupRequest, owner: &str) -> Result<Record, SetupError> {
        let digest = request.digest()?;
        let record = if let Some(record) = self.read(&request.operation_id).await? {
            ensure(
                record.owner == owner && record.receipt.request_sha256 == digest,
                "operation belongs to another owner or different request",
            )?;
            record
        } else {
            Record {
                format: "hades-setup-record-v1".into(),
                owner: owner.into(),
                receipt: SetupReceipt {
                    operation_id: request.operation_id.clone(),
                    database: request.database.clone(),
                    request_sha256: digest,
                    status: SetupStatus::Validating,
                    step: SetupStep::Validate,
                    diagnostic: None,
                },
                request,
                database_id: None,
            }
        };
        let guard = self.guard.clone();
        let claim = Claim {
            format: "hades-setup-claim-v1".into(),
            owner: record.owner.clone(),
            operation_id: record.request.operation_id.clone(),
            request_sha256: record.receipt.request_sha256.clone(),
            database: record.request.database.clone(),
        };
        let saved_record = record.clone();
        blocking(move || {
            let name = format!("db_{}.json", claim.database);
            let saved = read::<Claim>(&guard, &name)?;
            if let Some(saved) = &saved {
                ensure(
                    saved == &claim,
                    "database name is reserved by another setup operation",
                )?;
            }
            let operation = format!("op_{}.json", saved_record.request.operation_id);
            if read::<Record>(&guard, &operation)?.is_none() {
                write(&guard, &operation, &saved_record)?;
            }
            if saved.is_none() {
                write(&guard, &name, &claim)?;
            }
            Ok(())
        })
        .await?;
        Ok(record)
    }

    /// Retain the lease inside blocking work even if the awaiting caller is cancelled.
    pub async fn save(&self, record: &Record) -> Result<(), SetupError> {
        let guard = self.guard.clone();
        let record = record.clone();
        blocking(move || {
            write(
                &guard,
                &format!("op_{}.json", record.request.operation_id),
                &record,
            )
        })
        .await
    }
}

/// Require explicit absolute paths and reject traversal before walking descriptors.
pub(super) fn absolute(path: &Path) -> Result<(), SetupError> {
    ensure(
        path.is_absolute()
            && path
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
        "ledger path must be absolute without traversal",
    )
}

/// Compare effective ownership without consulting process environment variables.
fn owned(info: &std::fs::Metadata) -> bool {
    // SAFETY: geteuid takes no arguments and has no memory ownership requirements.
    info.uid() == unsafe { libc::geteuid() }
}

/// Refuse links, device nodes, FIFOs, shared metadata and extra hard links.
fn regular(file: &File) -> Result<(), SetupError> {
    let info = file.metadata()?;
    ensure(
        info.is_file() && owned(&info) && info.mode() & 0o7777 == 0o600 && info.nlink() == 1,
        "ledger metadata must be private single-link 0600 regular files",
    )
}

/// Open one child anchored in a checked directory, without following its final link.
fn open(root: &File, name: &CString, flags: i32) -> Result<File, SetupError> {
    // SAFETY: the directory descriptor and NUL-terminated name are live; creation uses explicit 0600.
    let fd = unsafe {
        libc::openat(
            root.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0o600,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful openat returned a new exclusively owned descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// Convert fixed ASCII record names into relative descriptor opens.
fn child(root: &File, name: &str, flags: i32) -> Result<File, SetupError> {
    open(
        root,
        &CString::new(name)
            .map_err(|_| SetupError::Invalid("invalid ledger record name".into()))?,
        flags,
    )
}

/// Distinguish absence from corruption or unreadable metadata; duplicates are rejected by serde.
fn read<T: DeserializeOwned>(guard: &Guard, name: &str) -> Result<Option<T>, SetupError> {
    let file = match child(&guard.root, name, libc::O_RDONLY) {
        Err(SetupError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        result => result?,
    };
    regular(&file)?;
    ensure(
        file.metadata()?.len() <= LIMIT as u64,
        "setup record exceeds 2 MiB",
    )?;
    let mut raw = Vec::new();
    file.take((LIMIT + 1) as u64).read_to_end(&mut raw)?;
    ensure(raw.len() <= LIMIT, "setup record exceeds 2 MiB")?;
    Ok(Some(serde_json::from_slice(&raw)?))
}

/// Publish a complete synced record, then sync its directory entry before permitting side effects.
fn write(guard: &Guard, name: &str, value: &impl Serialize) -> Result<(), SetupError> {
    let encoded = serde_json::to_vec(value)?;
    ensure(encoded.len() <= LIMIT, "setup record exceeds 2 MiB")?;
    // Validate any previous target instead of silently replacing hostile/corrupt filesystem objects.
    match child(&guard.root, name, libc::O_RDONLY) {
        Ok(file) => regular(&file)?,
        Err(SetupError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let temporary = CString::new(format!("pending_{:032x}", rand::random::<u128>())).unwrap();
    let mut file = open(
        &guard.root,
        &temporary,
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
    )?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    let target =
        CString::new(name).map_err(|_| SetupError::Invalid("invalid ledger record name".into()))?;
    // SAFETY: both names are NUL-terminated children of the live checked directory descriptor.
    if unsafe {
        libc::renameat(
            guard.root.as_raw_fd(),
            temporary.as_ptr(),
            guard.root.as_raw_fd(),
            target.as_ptr(),
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    guard.root.sync_all()?;
    Ok(())
}

/// Bound synchronous filesystem work stays on the blocking pool; cancellation retains its guards.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, SetupError> + Send + 'static,
) -> Result<T, SetupError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|_| SetupError::Invalid("ledger worker stopped".into()))?
}
