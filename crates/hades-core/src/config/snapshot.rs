//! Bounded, sealed Linux memory-file handoff for an ingestion child's config.
//! Configuration and credential bytes never enter argv or temporary files.
use super::HadesConfig;
use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;

const MAX_BYTES: usize = 64 * 1024;
const SEALS: i32 = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    config: HadesConfig,
    #[serde(default)]
    provenance: crate::source_git::Snapshot,
    // These remain skipped in ordinary configuration YAML.
    password: Option<String>,
    cuda_visible_devices: Option<String>,
}

struct LimitedWriter {
    file: File,
    written: usize,
}
impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_BYTES.saturating_sub(self.written) {
            return Err(std::io::Error::other(
                "ingestion configuration exceeds its byte limit",
            ));
        }
        let count = self.file.write(bytes)?;
        self.written += count;
        Ok(count)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

pub struct Snapshot(Arc<File>);
impl Snapshot {
    pub fn new(config: &HadesConfig) -> Result<Self> {
        Self::with_provenance(config, crate::source_git::Snapshot::default())
    }
    pub fn with_provenance(
        config: &HadesConfig,
        provenance: crate::source_git::Snapshot,
    ) -> Result<Self> {
        // SAFETY: constant NUL-terminated name and documented Linux flags.
        let fd = unsafe {
            libc::memfd_create(
                c"hades-ingest-config".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        ensure!(fd >= 0, "cannot create private ingestion configuration");
        // SAFETY: successful memfd_create returned a new exclusively owned FD.
        let file = unsafe { File::from_raw_fd(fd) };
        let file = if fd < 3 {
            // A daemon may have closed stdio. Keep the handoff away from the
            // descriptors that Command replaces while preparing the child.
            // SAFETY: duplicate our owned FD; the original closes on scope exit.
            let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
            ensure!(
                duplicate >= 3,
                "cannot reserve ingestion configuration descriptor"
            );
            // SAFETY: successful duplication returned a new owned FD.
            unsafe { File::from_raw_fd(duplicate) }
        } else {
            file
        };
        let fd = file.as_raw_fd();
        // SAFETY: owned descriptor; restrict its metadata permissions as well.
        ensure!(
            unsafe { libc::fchmod(fd, 0o600) } == 0,
            "cannot restrict ingestion configuration"
        );
        let envelope = Envelope {
            version: 2,
            provenance,
            config: config.clone(),
            password: config.database.password.clone(),
            cuda_visible_devices: config.gpu.cuda_visible_devices.clone(),
        };
        let mut writer = LimitedWriter { file, written: 0 };
        serde_json::to_writer(&mut writer, &envelope).map_err(|_| {
            anyhow!("cannot serialize ingestion configuration within its byte limit")
        })?;
        writer.file.seek(SeekFrom::Start(0))?;
        // SAFETY: no writable mapping exists and serialization is complete.
        ensure!(
            unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, SEALS) } == 0,
            "cannot seal ingestion configuration"
        );
        Ok(Self(Arc::new(writer.file)))
    }

    /// Only the selected child's pre_exec clears CLOEXEC. Other concurrent
    /// spawns do not inherit this FD across exec. The command retains ownership.
    pub fn inherit(&self, command: &mut tokio::process::Command) -> i32 {
        let file = Arc::clone(&self.0);
        let fd = file.as_raw_fd();
        // SAFETY: only async-signal-safe fcntl is called after fork; the closure
        // uses an already-owned FD and neither allocates nor locks in the child.
        unsafe {
            command.pre_exec(move || {
                if libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        fd
    }
}

/// Borrow the inherited FD, mark it CLOEXEC, and read through an owned duplicate.
/// Never assume ownership of an arbitrary descriptor supplied by the caller.
pub fn load_inherited(fd: i32) -> Result<HadesConfig> {
    load_ingest(fd).map(|(config, _)| config)
}

/// Recover the same per-run observations captured by admission (#186).
pub fn load_ingest(fd: i32) -> Result<(HadesConfig, crate::source_git::Snapshot)> {
    ensure!(fd >= 3, "invalid ingestion configuration descriptor");
    // SAFETY: fcntl validates the numeric descriptor without dereferencing it.
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & SEALS == SEALS,
        "ingestion configuration must be an immutable sealed memory file"
    );
    // SAFETY: prevent later analyzer exec inheritance and obtain a fresh owned FD.
    let duplicate = unsafe {
        if libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) == -1 {
            -1
        } else {
            libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3)
        }
    };
    ensure!(
        duplicate >= 0,
        "cannot read inherited ingestion configuration"
    );
    // SAFETY: F_DUPFD_CLOEXEC created this owned descriptor.
    let file = unsafe { File::from_raw_fd(duplicate) };
    let length = file.metadata()?.len();
    ensure!(
        length <= MAX_BYTES as u64,
        "ingestion configuration exceeds its byte limit"
    );
    // Seals make this length immutable. Positional reads avoid sharing a seek
    // offset with the parent or another reader of the inherited descriptor.
    let mut bytes = vec![0; length as usize];
    file.read_exact_at(&mut bytes, 0)?;
    let envelope: Envelope = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow!("invalid inherited ingestion configuration"))?;
    ensure!(
        matches!(envelope.version, 1 | 2),
        "unsupported ingestion configuration version"
    );
    let mut config = envelope.config;
    config.database.password = envelope.password;
    config.gpu.cuda_visible_devices = envelope.cuda_visible_devices;
    Ok((config, envelope.provenance))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> HadesConfig {
        let mut config = HadesConfig::with_database("private_snapshot");
        config.database.host = "fixture.invalid".into();
        config.database.password = Some("synthetic-only-secret".into());
        config.database.sockets.readwrite = Some("/private/fixture/db.sock".into());
        config.embedding.service.socket = "/private/fixture/embed.sock".into();
        config.extraction.service.socket = "/private/fixture/extract.sock".into();
        config.gpu.device = "cuda:7".into();
        config.gpu.cuda_visible_devices = Some("fixture-mask".into());
        config.batch_processing.concurrency = 3;
        config.analyzers.rust_analyzer = Some("/private/fixture/analyzer".into());
        config
    }
    #[test]
    fn snapshot_is_bounded_sealed_and_preserves_skipped_fields() {
        let config = fixture();
        let snapshot = Snapshot::new(&config).unwrap();
        let restored = load_inherited(snapshot.0.as_raw_fd()).unwrap();
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            serde_json::to_value(&config).unwrap()
        );
        assert_eq!(restored.database.password, config.database.password);
        assert_eq!(
            restored.gpu.cuda_visible_devices,
            config.gpu.cuda_visible_devices
        );
        assert!(
            !serde_json::to_string(&config)
                .unwrap()
                .contains("synthetic-only-secret")
        );
        // SAFETY: attempt one byte on our sealed fixture.
        assert_eq!(
            unsafe { libc::write(snapshot.0.as_raw_fd(), b"x".as_ptr().cast(), 1) },
            -1
        );
        let mut oversized = config;
        oversized.embedding.model.name = "x".repeat(MAX_BYTES);
        assert!(Snapshot::new(&oversized).is_err());
        for fd in [-1, 0, 1, 2, i32::MAX] {
            assert!(load_inherited(fd).is_err());
        }
        assert!(load_inherited(tempfile::tempfile().unwrap().as_raw_fd()).is_err());
    }
    #[test]
    fn inherited_probe() {
        let Ok(fd) = std::env::var("HADES_TEST_SNAPSHOT_FD") else {
            return;
        };
        let fd = fd.parse().unwrap();
        let loaded = load_inherited(fd).unwrap();
        // SAFETY: inspect only the inherited fixture FD's descriptor flags.
        assert_ne!(
            unsafe { libc::fcntl(fd, libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );
        let expected = fixture();
        assert_eq!(
            serde_json::to_value(&loaded).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(loaded.database.password, expected.database.password);
        assert_eq!(
            loaded.gpu.cuda_visible_devices,
            expected.gpu.cuda_visible_devices
        );
        // This probe runs in its own process. Simulate a daemon with stdin
        // closed and ensure the next snapshot never occupies a stdio slot.
        // SAFETY: close only this private probe's standard input.
        unsafe {
            libc::close(0);
        }
        let next = Snapshot::new(&expected).unwrap();
        assert!(next.0.as_raw_fd() >= 3);
        assert_eq!(
            load_inherited(next.0.as_raw_fd())
                .unwrap()
                .database
                .password,
            expected.database.password
        );
    }
    #[tokio::test]
    async fn selected_child_inherits_snapshot_despite_conflicting_environment() {
        let snapshot = Snapshot::new(&fixture()).unwrap();
        let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "config::snapshot::tests::inherited_probe",
                "--nocapture",
            ])
            .env_clear()
            .env("HADES_DATABASE", "wrong")
            .env("ARANGO_PASSWORD", "wrong")
            .env("HADES_CONFIG", "/missing/config")
            .env("CUDA_VISIBLE_DEVICES", "wrong");
        let fd = snapshot.inherit(&mut child);
        child.env("HADES_TEST_SNAPSHOT_FD", fd.to_string());
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            child.kill_on_drop(true).output(),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!String::from_utf8_lossy(&result.stdout).contains("synthetic-only-secret"));
    }
}
