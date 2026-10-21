//! Protected registration -> actual HTTP observation -> immutable role-bound client.
//! Synthetic identities and private Unix peers; no GPU, installed registry, or database.
#![cfg(unix)]

use hades_core::{
    embedding_profile::{
        ArtifactManifest, ChunkPolicy, EmbeddingProfile, EmbeddingRole, ProfileState, WorkloadKind,
    },
    persephone::{
        bound_embedding::ProviderConnection,
        embedding::{EmbeddingEndpoint, EmbeddingError},
        provider_registry::ProviderCatalog,
    },
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    fs,
    os::{
        fd::AsRawFd,
        unix::fs::{PermissionsExt, symlink},
    },
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[path = "common/registered_provider.rs"]
mod registration_fixture;

#[path = "provider_discovery/workloads.rs"]
mod workloads;

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    profile: EmbeddingProfile,
    manifest: ArtifactManifest,
    description: Value,
}

impl Fixture {
    /// Declare expected local identity separately from the peer's mutable reply queue.
    fn new() -> Self {
        let wire: Value =
            serde_json::from_str(include_str!("fixtures/embedding_profile.json")).unwrap();
        let profile: EmbeddingProfile = serde_json::from_value(wire["profile"].clone()).unwrap();
        let manifest: ArtifactManifest = serde_json::from_value(wire["manifest"].clone()).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let root = registration_fixture::create(directory.path(), &profile, &manifest);
        let description = json!({"model":profile.model,"capabilities":profile.capabilities,
            "space":profile.workloads.documents.as_ref().unwrap().space,"receipt":wire["receipt"]});
        Self {
            _directory: directory,
            root,
            profile,
            manifest,
            description,
        }
    }

    /// Use only this independently declared provider's private operational connection.
    fn catalog(&self, peer: &Peer) -> ProviderCatalog {
        ProviderCatalog::new(
            self.root.clone(),
            vec![peer.connection(&self.profile.runtime_binding.provider_id)],
        )
        .unwrap()
    }

    /// Locate only the fixture record for controlled corruption tests.
    fn record(&self) -> PathBuf {
        self.root
            .join("records")
            .join(format!("{}.json", self.profile.runtime_binding.provider_id))
    }
}

struct Reply {
    status: u16,
    body: String,
    before: Option<Box<dyn FnOnce() + Send>>,
    delay: Duration,
}
impl Reply {
    /// Construct an ordinary independent JSON observation or embedding response.
    fn json(value: Value) -> Self {
        Self {
            status: 200,
            body: value.to_string(),
            before: None,
            delay: Duration::ZERO,
        }
    }
}

/// HTTP request line and original bytes recorded independently of the client.
type CapturedRequest = (String, Vec<u8>);

struct Peer {
    _directory: tempfile::TempDir,
    path: PathBuf,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    server: tokio::task::JoinHandle<()>,
}
impl Peer {
    /// Serve raw replies and record even unexpected contacts until the owning test drops us.
    fn new(replies: Vec<Reply>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let parent = directory.path().join("sockets");
        fs::create_dir(&parent).unwrap();
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o700)).unwrap();
        let path = parent.join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = requests.clone();
        let mut replies: VecDeque<_> = replies.into();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::time::timeout(Duration::from_secs(5), async {
                    let mut headers = Vec::new();
                    while !headers.ends_with(b"\r\n\r\n") {
                        headers.push(stream.read_u8().await.unwrap());
                        assert!(headers.len() < 8192);
                    }
                    let headers = String::from_utf8(headers).unwrap();
                    let length: usize = headers.lines().find_map(|line| line.to_lowercase()
                        .strip_prefix("content-length:").map(|n| n.trim().parse().unwrap())).unwrap_or(0);
                    assert!(length < 1024 * 1024);
                    let mut body = vec![0; length];
                    stream.read_exact(&mut body).await.unwrap();
                    observed.lock().unwrap().push((headers.lines().next().unwrap().into(), body));
                    let mut reply = replies.pop_front().unwrap_or(Reply { status: 500,
                        body: "unexpected contact".into(), before: None, delay: Duration::ZERO });
                    if let Some(action) = reply.before.take() { action(); }
                    tokio::time::sleep(reply.delay).await;
                    let wire = format!("HTTP/1.1 {} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.status, reply.body.len(), reply.body);
                    let _ = stream.write_all(wire.as_bytes()).await;
                }).await.expect("fixture request must finish");
            }
        });
        Self {
            _directory: directory,
            path,
            requests,
            server,
        }
    }

    /// Provide only an allowlisted ID, private socket, timeout, and response bound.
    fn connection(&self, id: &str) -> ProviderConnection {
        ProviderConnection {
            provider_id: id.into(),
            endpoint: EmbeddingEndpoint::Unix(self.path.clone()),
            timeout: Duration::from_secs(3),
            max_response_bytes: 1024 * 1024,
        }
    }

    /// Inspect actual wire contacts, including empty-body verification requests.
    fn requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Peer {
    /// Reclaim fixture work even when a negative contract assertion fails.
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// Metadata inspection never turns registration into a serving receipt or contacts a model.
#[tokio::test]
async fn inspect_lists_only_allowlisted_ids_without_loading() {
    let fixture = Fixture::new();
    let peer = Peer::new(vec![]);
    let catalog = fixture.catalog(&peer);
    assert_eq!(
        catalog.provider_ids().collect::<Vec<_>>(),
        ["fixture-local-provider"]
    );
    let summary = catalog.inspect("fixture-local-provider").await.unwrap();
    assert!(!summary.loaded_verified);
    assert_eq!(summary.manifest_md5, fixture.manifest.digest().unwrap());
    assert_eq!(summary.model_id, fixture.profile.model.id);
    for id in [
        "unconfigured",
        "../fixture-local-provider",
        "",
        "/tmp/provider",
    ] {
        assert!(catalog.verify(id).await.is_err());
    }
    assert!(peer.requests().is_empty());
}

/// Observation supplies actual model/space data, while the database retains explicit intent.
#[tokio::test]
async fn verified_observation_builds_profiles_and_pins_role_clients() {
    let fixture = Fixture::new();
    let embedding = json!({"model":fixture.profile.model.id,"receipt":fixture.description["receipt"],
        "data":[{"index":0,"embedding":vec![0.5;384]}]});
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(embedding),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify("fixture-local-provider")
        .await
        .unwrap();
    assert_eq!(json!(observed.description()), fixture.description);
    assert_eq!(
        observed.manifest().digest().unwrap(),
        fixture.manifest.digest().unwrap()
    );
    let profile = observed
        .document_profile(
            "new-corpus".into(),
            ChunkPolicy::Conventional {
                passage_tokens: 500,
                overlap_tokens: 100,
            },
        )
        .unwrap();
    assert_eq!(profile.state, ProfileState::Initializing);
    assert_eq!(profile.revision, 1);
    assert!(profile.workloads.code.is_none());
    assert!(
        observed
            .document_profile(
                "new".into(),
                ChunkPolicy::Late {
                    context_cap_tokens: 1000,
                    minimum_window_overlap_tokens: 100,
                    passages: None,
                }
            )
            .is_err()
    );
    assert!(
        observed
            .client(
                &profile,
                observed.manifest(),
                WorkloadKind::Documents,
                EmbeddingRole::Query
            )
            .await
            .is_err()
    );
    let client = observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Query,
        )
        .await
        .unwrap();
    client.embed(&["one".into()], None).await.unwrap();
    let requests = peer.requests();
    assert_eq!(requests[0].0, "POST /v1/verify HTTP/1.1");
    assert!(requests[0].1.is_empty());
    let request: Value = serde_json::from_slice(&requests[1].1).unwrap();
    assert_eq!(request["expected_receipt"], fixture.description["receipt"]);
    assert_eq!(request["model"], fixture.profile.model.id);
    assert_eq!(request["task"], "retrieval.query");
}

/// Reject malformed configuration instead of falling back to ambient endpoints or providers.
#[test]
fn catalog_refuses_ambiguous_or_unbounded_operational_settings() {
    let fixture = Fixture::new();
    let base = ProviderConnection {
        provider_id: "valid".into(),
        endpoint: EmbeddingEndpoint::Unix("/tmp/private/provider.sock".into()),
        timeout: Duration::from_secs(1),
        max_response_bytes: 1024,
    };
    assert!(ProviderCatalog::new(PathBuf::from("relative"), vec![base.clone()]).is_err());
    assert!(ProviderCatalog::new(fixture.root.clone(), vec![base.clone(), base.clone()]).is_err());
    assert!(ProviderCatalog::new(fixture.root.clone(), vec![base.clone(); 129]).is_err());
    for variant in [
        "tcp",
        "relative",
        "traversal",
        "zero_timeout",
        "timeout",
        "zero_bytes",
        "bytes",
        "id",
    ] {
        let mut wrong = base.clone();
        match variant {
            "tcp" => wrong.endpoint = EmbeddingEndpoint::Tcp("http://localhost:8087/v1".into()),
            "relative" => wrong.endpoint = EmbeddingEndpoint::Unix("provider.sock".into()),
            "traversal" => wrong.endpoint = EmbeddingEndpoint::Unix("/tmp/../provider.sock".into()),
            "zero_timeout" => wrong.timeout = Duration::ZERO,
            "timeout" => wrong.timeout = Duration::from_secs(601),
            "zero_bytes" => wrong.max_response_bytes = 0,
            "bytes" => wrong.max_response_bytes = 8 * 1024 * 1024 + 1,
            _ => wrong.provider_id = "not/a/key".into(),
        }
        assert!(
            ProviderCatalog::new(fixture.root.clone(), vec![wrong]).is_err(),
            "{variant}"
        );
    }
}

/// Missing, unsafe, or busy metadata must fail before any network request.
#[tokio::test]
async fn protected_metadata_and_lock_checks_precede_network() {
    for variant in [
        "missing",
        "root_mode",
        "records_mode",
        "file_mode",
        "root_link",
        "parent_link",
        "file_link",
        "fifo",
        "artifact_write",
        "artifact_link",
        "locked",
    ] {
        let mut fixture = Fixture::new();
        let peer = Peer::new(vec![]);
        let mut held = None;
        match variant {
            "missing" => fs::remove_file(fixture.record()).unwrap(),
            "root_mode" => {
                fs::set_permissions(&fixture.root, fs::Permissions::from_mode(0o755)).unwrap()
            }
            "records_mode" => fs::set_permissions(
                fixture.root.join("records"),
                fs::Permissions::from_mode(0o750),
            )
            .unwrap(),
            "file_mode" => {
                fs::set_permissions(fixture.record(), fs::Permissions::from_mode(0o644)).unwrap()
            }
            "root_link" => {
                let link = fixture._directory.path().join("linked");
                symlink(&fixture.root, &link).unwrap();
                fixture.root = link;
            }
            "parent_link" => {
                let link = fixture._directory.path().join("parent-link");
                symlink(fixture._directory.path(), &link).unwrap();
                fixture.root = link.join("registry");
            }
            "file_link" => {
                let original = fixture.record().with_extension("real");
                fs::rename(fixture.record(), &original).unwrap();
                symlink(original, fixture.record()).unwrap();
            }
            "fifo" => {
                fs::remove_file(fixture.record()).unwrap();
                let path = std::ffi::CString::new(fixture.record().as_os_str().as_encoded_bytes())
                    .unwrap();
                // SAFETY: temporary owned path is NUL-terminated; mkfifo only creates that fixture.
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            }
            "artifact_write" => fs::set_permissions(
                fixture
                    .root
                    .join("artifacts")
                    .join(format!("model_{}", "0".repeat(32))),
                fs::Permissions::from_mode(0o700),
            )
            .unwrap(),
            "artifact_link" => {
                let path = fixture
                    .root
                    .join("artifacts")
                    .join(format!("model_{}", "0".repeat(32)));
                let moved = path.with_extension("real");
                fs::rename(&path, &moved).unwrap();
                symlink(moved, path).unwrap();
            }
            _ => {
                let lock = fs::File::open(fixture.root.join(".lock")).unwrap();
                // SAFETY: owned descriptor; Drop releases this fixture lock.
                assert_eq!(
                    unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                    0
                );
                held = Some(lock);
            }
        }
        assert!(
            fixture
                .catalog(&peer)
                .verify("fixture-local-provider")
                .await
                .is_err(),
            "{variant}"
        );
        assert!(peer.requests().is_empty(), "{variant}");
        drop(held);
    }
}

/// Strict parsing and manifest checks reject forged records without querying the selected service.
#[tokio::test]
async fn invalid_registrations_and_saved_model_mismatches_do_not_load() {
    for variant in [
        "unknown",
        "duplicate",
        "manifest_duplicate",
        "digest",
        "id",
        "snapshot",
        "format",
        "oversize",
        "model",
        "revision",
    ] {
        let mut fixture = Fixture::new();
        let peer = Peer::new(vec![]);
        let raw = fs::read_to_string(fixture.record()).unwrap();
        let mut record: Value = serde_json::from_str(&raw).unwrap();
        match variant {
            "unknown" => record["unknown"] = json!(true),
            "duplicate" => {
                fs::write(
                    fixture.record(),
                    raw.replacen(
                        "\"provider_id\":",
                        "\"provider_id\":\"fixture-local-provider\",\"provider_id\":",
                        1,
                    ),
                )
                .unwrap();
            }
            "manifest_duplicate" => {
                fs::write(
                    fixture.record(),
                    raw.replacen("\"files\":", "\"files\":[],\"files\":", 1),
                )
                .unwrap();
            }
            "digest" => record["manifest_md5"] = json!("0".repeat(32)),
            "id" => record["provider_id"] = json!("other"),
            "snapshot" => record["snapshot"] = json!("../../outside"),
            "format" => record["format"] = json!("other"),
            "oversize" => {
                fs::write(fixture.record(), " ".repeat(2 * 1024 * 1024 + 1)).unwrap();
            }
            "model" => fixture.profile.model.id = "unregistered-model".into(),
            "revision" => fixture.profile.model.revision = "unregistered-revision".into(),
            _ => unreachable!(),
        }
        if !matches!(variant, "duplicate" | "manifest_duplicate" | "oversize") {
            fs::write(fixture.record(), record.to_string()).unwrap();
        }
        assert!(
            fixture
                .catalog(&peer)
                .verify_profile(&fixture.profile, &fixture.manifest)
                .await
                .is_err(),
            "{variant}"
        );
        assert!(peer.requests().is_empty(), "{variant}");
    }
}

/// Decode observations directly from bytes so duplicate and unknown identity fields cannot disappear.
#[tokio::test]
async fn strict_wire_and_inconsistent_live_descriptions_are_refused() {
    let fixture = Fixture::new();
    let original = fixture.description.to_string();
    let mut replies = vec![
        original.replacen(
            "\"capabilities\":",
            "\"capabilities\":{},\"capabilities\":",
            1,
        ),
        original.replacen(
            "\"dimension\":384",
            "\"dimension\":384,\"dimension\":384",
            1,
        ),
        original.replacen("\"text\":true", "\"text\":\"true\"", 1),
        original.replacen("\"receipt\":", "\"unexpected\":1,\"receipt\":", 1),
    ];
    for (pointer, value) in [
        ("/model/id", json!("other")),
        ("/model/revision", json!("other")),
        ("/model/manifest_md5", json!("0".repeat(32))),
        ("/model/tokenizer_files", json!(["missing.json"])),
        ("/receipt/artifact_md5", json!("0".repeat(32))),
        ("/receipt/space_id", json!("0".repeat(64))),
        ("/receipt/serving_generation", json!("")),
        ("/capabilities/text", json!(false)),
        ("/capabilities/max_input_tokens", json!(0)),
        ("/capabilities/tasks", json!(["retrieval.query"])),
        (
            "/capabilities/tasks",
            json!(["retrieval.query", "retrieval.query"]),
        ),
        ("/space/representation/dimension", json!(0)),
    ] {
        let mut wrong = fixture.description.clone();
        *wrong.pointer_mut(pointer).unwrap() = value;
        replies.push(wrong.to_string());
    }
    for body in replies {
        let peer = Peer::new(vec![Reply {
            body,
            status: 200,
            before: None,
            delay: Duration::ZERO,
        }]);
        assert!(
            fixture
                .catalog(&peer)
                .verify("fixture-local-provider")
                .await
                .is_err()
        );
        assert_eq!(peer.requests().len(), 1);
    }
}

/// Provider loads must not outlive transport limits or introduce oversized observation metadata.
#[tokio::test]
async fn verification_bounds_time_bytes_and_http_errors() {
    let fixture = Fixture::new();
    for variant in ["timeout", "bytes", "http"] {
        let mut reply = Reply::json(fixture.description.clone());
        match variant {
            "timeout" => reply.delay = Duration::from_millis(250),
            "bytes" => reply.body = "x".repeat(64 * 1024 + 1),
            _ => reply.status = 409,
        }
        let peer = Peer::new(vec![reply]);
        let mut connection = peer.connection("fixture-local-provider");
        if variant == "timeout" {
            connection.timeout = Duration::from_millis(40);
        }
        let catalog = ProviderCatalog::new(fixture.root.clone(), vec![connection]).unwrap();
        let error = catalog
            .verify("fixture-local-provider")
            .await
            .err()
            .expect("must reject");
        match variant {
            "timeout" => assert!(matches!(error, EmbeddingError::Timeout(_))),
            "bytes" => assert!(matches!(error, EmbeddingError::InvalidResponse(_))),
            _ => assert!(matches!(error, EmbeddingError::Http { status: 409, .. })),
        }
        assert_eq!(peer.requests().len(), 1);
    }
}

/// Replacing the protected record with identical bytes still invalidates an in-flight observation.
#[tokio::test]
async fn registration_change_during_load_discards_observation() {
    let fixture = Fixture::new();
    let record = fixture.record();
    let raw = fs::read(&record).unwrap();
    let mut reply = Reply::json(fixture.description.clone());
    reply.before = Some(Box::new(move || {
        let next = record.with_extension("replacement");
        fs::write(&next, &raw).unwrap();
        fs::set_permissions(&next, fs::Permissions::from_mode(0o600)).unwrap();
        fs::rename(next, record).unwrap();
    }));
    let peer = Peer::new(vec![reply, Reply::json(fixture.description.clone())]);
    let catalog = fixture.catalog(&peer);
    let error = catalog
        .verify("fixture-local-provider")
        .await
        .err()
        .expect("changed record must fail");
    assert!(error.to_string().contains("registration changed during"));
    assert!(catalog.verify("fixture-local-provider").await.is_ok());
}

/// A smaller live budget is operational; changed spaces or required capabilities are incompatible.
#[tokio::test]
async fn saved_policy_is_checked_against_actual_capabilities_without_changing_it() {
    let fixture = Fixture::new();
    let mut smaller = fixture.description.clone();
    smaller["capabilities"]["max_input_tokens"] = json!(128);
    let peer = Peer::new(vec![Reply::json(smaller)]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    assert_eq!(
        observed
            .description()
            .workload(WorkloadKind::Documents)
            .unwrap()
            .capabilities
            .max_input_tokens,
        128
    );
    assert_eq!(fixture.profile.capabilities.max_input_tokens, 2048);
    observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Passage,
        )
        .await
        .unwrap();
    for mode in ["late", "specialized", "space"] {
        let mut profile = fixture.profile.clone();
        match mode {
            "late" => {
                profile.capabilities.late_chunking = true;
                profile.capabilities.tokenizer_offsets = true;
                profile.workloads.documents.as_mut().unwrap().chunking = ChunkPolicy::Late {
                    context_cap_tokens: 1000,
                    minimum_window_overlap_tokens: 100,
                    passages: None,
                };
            }
            "specialized" => {
                profile.capabilities.specialized_code = true;
                profile.workloads.code = profile.workloads.documents.take();
                profile.workloads.code.as_mut().unwrap().specialized_code = true;
            }
            _ => {
                profile
                    .workloads
                    .documents
                    .as_mut()
                    .unwrap()
                    .space
                    .representation
                    .pooling = "another-pooling".into()
            }
        }
        let peer = Peer::new(vec![Reply::json(fixture.description.clone())]);
        assert!(
            fixture
                .catalog(&peer)
                .verify_profile(&profile, &fixture.manifest)
                .await
                .is_err(),
            "{mode}"
        );
    }
}

/// Local socket identity includes private ownership; a symlink or public parent is not admitted.
#[tokio::test]
async fn unsafe_socket_paths_are_rejected_before_http() {
    let fixture = Fixture::new();
    for variant in ["parent", "ancestor", "symlink"] {
        let peer = Peer::new(vec![]);
        let mut connection = peer.connection("fixture-local-provider");
        if variant == "parent" {
            fs::set_permissions(
                peer.path.parent().unwrap(),
                fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        } else if variant == "ancestor" {
            fs::set_permissions(peer._directory.path(), fs::Permissions::from_mode(0o777)).unwrap();
        } else {
            let link = peer._directory.path().join("linked.sock");
            symlink(&peer.path, &link).unwrap();
            connection.endpoint = EmbeddingEndpoint::Unix(link);
        }
        let catalog = ProviderCatalog::new(fixture.root.clone(), vec![connection]).unwrap();
        assert!(catalog.verify("fixture-local-provider").await.is_err());
        assert!(peer.requests().is_empty());
    }
}
