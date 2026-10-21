//! Service-free contracts for durable reservations and refusal of damaged local metadata.
use super::*;
use crate::embedding_profile::ProfileState;
use serde_json::{Value, json};
use std::os::unix::fs::{PermissionsExt, symlink};

/// Build portable synthetic intent; no database socket or provider is needed.
fn request() -> SetupRequest {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/embedding_profile.json")).unwrap();
    let mut request = SetupRequest {
        operation_id: "fixture-op".into(),
        database: "fixture_workspace".into(),
        profile: serde_json::from_value(fixture["profile"].clone()).unwrap(),
        manifest: serde_json::from_value(fixture["manifest"].clone()).unwrap(),
    };
    request.profile.state = ProfileState::Initializing;
    request
}
/// Allocate a private ledger on the local temporary filesystem.
fn directory() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    directory
}
/// Use an explicit stable test scope with no runtime secrets.
fn scope() -> Scope {
    Scope {
        format: "hades-setup-scope-v1".into(),
        installation: "fixture".into(),
        endpoint_sha256: "1".repeat(64),
    }
}

/// A reopened ledger retains the exact request, owner and database reservation without rewriting it.
#[tokio::test]
async fn durable_reservations_are_exact_and_process_locked() {
    let root = directory();
    let lease = Lease::open(root.path().into(), scope()).await.unwrap();
    let record = lease.claim(request(), "owner").await.unwrap();
    assert!(matches!(
        Lease::open(root.path().into(), scope()).await,
        Err(SetupError::Busy)
    ));
    let path = root.path().join("op_fixture-op.json");
    let bytes = std::fs::read(&path).unwrap();
    assert!(lease.claim(request(), "other-owner").await.is_err());
    let mut changed = request();
    changed.profile.generation = "another".into();
    assert!(lease.claim(changed, "owner").await.is_err());
    let mut competing = request();
    competing.operation_id = "competitor".into();
    assert!(lease.claim(competing, "owner").await.is_err());
    assert!(!root.path().join("op_competitor.json").exists());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    drop(lease);
    let reopened = Lease::open(root.path().into(), scope()).await.unwrap();
    assert_eq!(
        reopened.claim(request(), "owner").await.unwrap().receipt,
        record.receipt
    );
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    drop(reopened);
    let mut elsewhere = scope();
    elsewhere.endpoint_sha256 = "2".repeat(64);
    assert!(Lease::open(root.path().into(), elsewhere).await.is_err());
}

/// A descriptor inherited before exec cannot extend the lease after its final worker releases it.
#[tokio::test]
async fn inherited_descriptors_do_not_extend_a_completed_lease() {
    let root = directory();
    let lease = Lease::open(root.path().into(), scope()).await.unwrap();
    let worker = lease.guard.clone();
    // dup and fork share an open file description; no actual child or timing race is needed.
    let inherited = worker.lock.try_clone().unwrap();
    drop(lease);
    assert!(matches!(
        Lease::open(root.path().into(), scope()).await,
        Err(SetupError::Busy)
    ));
    drop(worker);
    let reopened = Lease::open(root.path().into(), scope()).await.unwrap();
    drop(inherited);
    assert!(matches!(
        Lease::open(root.path().into(), scope()).await,
        Err(SetupError::Busy)
    ));
    drop(reopened);
    Lease::open(root.path().into(), scope()).await.unwrap();
}

/// A partial temporary file cannot replace the last fully published operation record.
#[tokio::test]
async fn incomplete_temporary_records_are_never_adopted() {
    let root = directory();
    let lease = Lease::open(root.path().into(), scope()).await.unwrap();
    let record = lease.claim(request(), "owner").await.unwrap();
    drop(lease);
    std::fs::write(root.path().join("pending_interrupted"), b"{partial JSON").unwrap();
    let reopened = Lease::open(root.path().into(), scope()).await.unwrap();
    assert_eq!(
        reopened.read("fixture-op").await.unwrap().unwrap().receipt,
        record.receipt
    );
}

/// Metadata inspection rejects corruption instead of defaulting missing receipt fields or replacing files.
#[tokio::test]
async fn malformed_and_inconsistent_records_fail_closed() {
    for variant in [
        "format",
        "digest",
        "owner",
        "checkpoint",
        "status",
        "diagnostic",
        "ready",
        "unknown",
        "duplicate",
        "oversized",
    ] {
        let root = directory();
        let lease = Lease::open(root.path().into(), scope()).await.unwrap();
        lease.claim(request(), "owner").await.unwrap();
        let path = root.path().join("op_fixture-op.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        match variant {
            "format" => value["format"] = json!("v999"),
            "digest" => value["receipt"]["request_sha256"] = json!("0".repeat(64)),
            "owner" => value["owner"] = json!("../bad"),
            "checkpoint" => value["receipt"]["step"] = json!("marker"),
            "status" => value["receipt"]["status"] = json!("creating"),
            "diagnostic" => value["receipt"]["diagnostic"] = json!("unknown"),
            "ready" => value["receipt"]["status"] = json!("ready"),
            _ => {}
        }
        let encoded = match variant {
            "unknown" => {
                value["extra"] = json!(true);
                value.to_string()
            }
            "duplicate" => {
                value
                    .to_string()
                    .replacen("\"format\":", "\"format\":\"duplicate\",\"format\":", 1)
            }
            "oversized" => " ".repeat(LIMIT + 1),
            _ => value.to_string(),
        };
        std::fs::write(&path, &encoded).unwrap();
        assert!(lease.read("fixture-op").await.is_err(), "{variant}");
        assert!(lease.claim(request(), "owner").await.is_err(), "{variant}");
        assert_eq!(std::fs::read_to_string(path).unwrap(), encoded);
    }
}

/// Unsafe root paths and metadata objects cannot provide setup authority.
#[tokio::test]
async fn private_metadata_rejects_symlinks_hardlinks_and_permissions() {
    for variant in [
        "root_mode",
        "root_link",
        "record_mode",
        "record_link",
        "hardlink",
        "fifo",
        "lock_link",
    ] {
        let root = directory();
        let lease = Lease::open(root.path().into(), scope()).await.unwrap();
        lease.claim(request(), "owner").await.unwrap();
        drop(lease);
        let path = root.path().join("op_fixture-op.json");
        match variant {
            "root_mode" => {
                std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755))
                    .unwrap()
            }
            "root_link" => {
                let parent = directory();
                let link = parent.path().join("linked");
                symlink(root.path(), &link).unwrap();
                assert!(Lease::open(link, scope()).await.is_err());
                continue;
            }
            "record_mode" => {
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap()
            }
            "record_link" => {
                let target = root.path().join("target");
                std::fs::rename(&path, &target).unwrap();
                symlink(target, &path).unwrap();
            }
            "hardlink" => std::fs::hard_link(&path, root.path().join("alias")).unwrap(),
            "fifo" => {
                std::fs::remove_file(&path).unwrap();
                let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
                // SAFETY: this is a temporary fixture path and a valid NUL-terminated string.
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            }
            _ => {
                let lock = root.path().join(".lock");
                std::fs::remove_file(&lock).unwrap();
                symlink(&path, lock).unwrap();
            }
        }
        if let Ok(lease) = Lease::open(root.path().into(), scope()).await {
            assert!(lease.read("fixture-op").await.is_err(), "{variant}");
        }
    }
}

/// Reject unsupported intent before it can reserve names or be serialized to the durable ledger.
#[test]
fn invalid_requests_are_rejected_before_reservation() {
    for database in ["", "_system", "../other", "a/b", "a?x", "a b", "日本語"] {
        let mut r = request();
        r.database = database.into();
        assert!(r.validate().is_err());
    }
    let mut r = request();
    r.database = "a".repeat(65);
    assert!(r.validate().is_err());
    let mut r = request();
    r.operation_id = "../operation".into();
    assert!(r.validate().is_err());
    let mut r = request();
    r.profile.state = ProfileState::Ready;
    assert!(r.validate().is_err());
    let mut r = request();
    r.profile.workloads.code = r.profile.workloads.documents.take();
    assert!(r.validate().is_err());
    let mut r = request();
    r.profile
        .workloads
        .documents
        .as_mut()
        .unwrap()
        .space
        .representation
        .distance = crate::embedding_profile::Distance::Dot;
    assert!(r.validate().is_err());
}
