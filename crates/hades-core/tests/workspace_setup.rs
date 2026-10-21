//! Recoverable setup against disposable databases, protected ledgers and synthetic providers.
use hades_core::{
    db::{ArangoClient, ArangoPool, crud},
    embedding_profile::{ProfileState, store::ProfileStore},
    pipeline::profiled_documents::{DocumentInput, ProfiledDocuments},
    test_support::{Fixtures, with_temp_db},
    workspace_setup::{SetupError, SetupRequest, SetupStatus, SetupStep, WorkspaceSetup},
};
use serde_json::json;
use std::{fs, os::unix::fs::PermissionsExt, path::Path, sync::atomic::Ordering, time::Duration};
#[path = "common/setup_proxy.rs"]
mod fault;
#[allow(dead_code)]
#[path = "common/profiled_provider.rs"]
mod provider;
use fault::{Proxy, Rule};
use provider::{Fixture, Peer};

#[path = "workspace_setup/recovery.rs"]
mod recovery;

/// An explicit private fixture ledger, never the workstation's installation state.
fn ledger() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
    directory
}
/// Use only the disposable harness's selected socket and test credentials.
fn admin(pool: &ArangoPool) -> ArangoClient {
    ArangoClient::with_socket(
        pool.writer().socket_path().unwrap().into(),
        "_system",
        &std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into()),
        &std::env::var("ARANGO_PASSWORD").unwrap(),
    )
}
/// Let the coordinator create the exact database name the harness owns and will drop.
async fn remove_fixture_database(pool: &ArangoPool) {
    admin(pool)
        .delete(&format!("database/{}", pool.database()))
        .await
        .unwrap();
}
/// Build explicit document intent using independently declared fixture model metadata.
fn request(pool: &ArangoPool, fixture: &Fixture) -> SetupRequest {
    SetupRequest {
        operation_id: "setup-one".into(),
        database: pool.database().into(),
        profile: fixture.profile.clone(),
        manifest: fixture.manifest.clone(),
    }
}
/// Construct trusted operational settings separately from the serializable request.
fn setup(root: &Path, admin: ArangoClient, peer: &Peer) -> WorkspaceSetup {
    WorkspaceSetup::new(
        root.into(),
        "test-installation".into(),
        "operator".into(),
        admin,
        peer.catalog(),
    )
    .unwrap()
}
/// Single-use failures preserve a real database effect when before=false.
fn rule(method: &'static str, path: &'static str, name: Option<&'static str>, nth: usize) -> Rule {
    Rule {
        method,
        path,
        name,
        nth,
        before: false,
        hold: false,
        fail: true,
        callback: None,
    }
}

/// Creation publishes exact policy, supports document use, and repeats without modifying immutable metadata.
#[tokio::test]
async fn creation_and_identical_retries_preserve_profile_and_documents() {
    with_temp_db("setup_roundtrip", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let setup = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        let receipt = setup.run(request.clone()).await.unwrap();
        assert_eq!(receipt.status, SetupStatus::Ready);
        assert_eq!(receipt.step, SetupStep::Complete);
        assert_eq!(
            setup.inspect(&request.operation_id).await.unwrap(),
            Some(receipt.clone())
        );
        assert_eq!(peer.observations(), 2);
        let store = ProfileStore::new(pool.clone()).unwrap();
        let before = store.load().await.unwrap();
        assert_eq!(before.profile().revision, 2);
        let documents = ProfiledDocuments::connect(pool.clone(), &peer.catalog())
            .await
            .unwrap();
        let input = DocumentInput {
            source_id: "notes".into(),
            text: "apple café".into(),
        };
        let prepared = documents.prepare(input, false).await.unwrap();
        documents.commit(prepared).await.unwrap();
        // Background jobs add operational history only after readiness. An
        // identical setup retry must preserve and validate that collection.
        pool.writer()
            .post(
                "collection",
                &json!({"name":"hades_ingest_jobs","type":2,"waitForSync":true}),
            )
            .await
            .unwrap();
        let history = crud::insert_document(
            &pool,
            "hades_ingest_jobs",
            &json!({"_key":"fixture_job","status":"completed"}),
        )
        .await
        .unwrap();
        assert_eq!(setup.run(request.clone()).await.unwrap(), receipt);
        assert_eq!(
            crud::get_document(&pool, "hades_ingest_jobs", "fixture_job")
                .await
                .unwrap()["_rev"],
            history["_rev"]
        );
        pool.writer()
            .put(
                "collection/hades_ingest_jobs/properties",
                &json!({"waitForSync":false}),
            )
            .await
            .unwrap();
        assert!(setup.run(request.clone()).await.is_err());
        pool.writer()
            .put(
                "collection/hades_ingest_jobs/properties",
                &json!({"waitForSync":true}),
            )
            .await
            .unwrap();
        assert_eq!(setup.run(request.clone()).await.unwrap(), receipt);
        assert_eq!(store.load().await.unwrap(), before);
        assert_eq!(
            documents.search("apple", 1, 2).await.unwrap().items.len(),
            1
        );
        assert!(!serde_json::to_string(&receipt).unwrap().contains("socket"));
        let mut changed = request;
        changed.profile.generation = "different".into();
        assert!(setup.run(changed).await.is_err());
        assert_eq!(store.load().await.unwrap(), before);
    })
    .await;
}

/// An existing database is never adopted, emptied, marked, or assigned the current model.
#[tokio::test]
async fn contextual_setup_retains_explicit_policy_and_retries_after_ingestion() {
    with_temp_db("setup_late", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::late();
        let peer = Peer::new(&fixture);
        let root = ledger();
        let setup = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        let receipt = setup.run(request.clone()).await.unwrap();
        assert_eq!(receipt.status, SetupStatus::Ready);
        let store = ProfileStore::new(pool.clone()).unwrap();
        let before = store.load().await.unwrap();
        assert_eq!(before.profile().workloads, fixture.profile.workloads);
        let documents = fixture.connect(&pool, &peer).await;
        let prepared = documents
            .prepare(
                DocumentInput {
                    source_id: "late".into(),
                    text: "apple café\nbanana 東京\napple tail".into(),
                },
                false,
            )
            .await
            .unwrap();
        documents.commit(prepared).await.unwrap();
        assert_eq!(documents.search("banana", 1, 10).await.unwrap().scanned, 3);
        assert_eq!(setup.run(request).await.unwrap(), receipt);
        assert_eq!(store.load().await.unwrap(), before);
    })
    .await;
}

#[tokio::test]
async fn existing_database_is_preserved_and_requires_reconciliation() {
    with_temp_db("setup_existing", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "authored", Some(2))
            .await
            .unwrap();
        crud::insert_document(
            &pool,
            "authored",
            &json!({"_key":"source","text":"preserve me"}),
        )
        .await
        .unwrap();
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let setup = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        assert!(matches!(
            setup.run(request.clone()).await,
            Err(SetupError::Reconciliation(_))
        ));
        assert_eq!(
            setup
                .inspect(&request.operation_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            SetupStatus::NeedsReconciliation
        );
        assert!(setup.run(request).await.is_err());
        assert_eq!(
            crud::get_document(&pool, "authored", "source")
                .await
                .unwrap()["text"],
            "preserve me"
        );
        assert!(
            pool.writer()
                .get("collection/hades_workspace")
                .await
                .is_err()
        );
        assert!(pool.writer().get("collection/hades_setup").await.is_err());
    })
    .await;
}

/// Real acknowledged collection/marker/profile/schema/ready effects can be reconciled without reset.
#[tokio::test]
async fn lost_initialization_responses_resume_only_owned_steps() {
    for (tag, rule, step) in [
        (
            "marker_collection",
            rule("POST", "collection", Some("hades_setup"), 1),
            SetupStep::Marker,
        ),
        (
            "marker",
            rule("POST", "document/hades_setup", None, 1),
            SetupStep::Marker,
        ),
        (
            "metadata_collection",
            rule("POST", "collection", Some("hades_workspace"), 1),
            SetupStep::Profile,
        ),
        (
            "profile_commit",
            rule("PUT", "transaction/", None, 1),
            SetupStep::Profile,
        ),
        (
            "documents",
            rule("POST", "collection", Some("documents"), 1),
            SetupStep::Schema,
        ),
        (
            "chunks",
            rule("POST", "collection", Some("chunks"), 1),
            SetupStep::Schema,
        ),
        (
            "embeddings",
            rule("POST", "collection", Some("embeddings"), 1),
            SetupStep::Schema,
        ),
        (
            "ready_commit",
            rule("PUT", "transaction/", None, 3),
            SetupStep::Publish,
        ),
    ] {
        with_temp_db(tag, Fixtures::Empty, move |pool| async move {
            remove_fixture_database(&pool).await;
            let fixture = Fixture::new(2);
            let peer = Peer::new(&fixture);
            let root = ledger();
            let proxy = Proxy::new(pool.writer(), rule);
            let setup = setup(root.path(), proxy.admin.clone(), &peer);
            let request = request(&pool, &fixture);
            assert!(setup.run(request.clone()).await.is_err(), "{tag}");
            assert_eq!(proxy.hits.load(Ordering::SeqCst), 1, "{tag}");
            let failed = setup.inspect(&request.operation_id).await.unwrap().unwrap();
            assert_eq!(failed.status, SetupStatus::Failed, "{tag}");
            assert_eq!(failed.step, step, "{tag}");
            let diagnostic = setup.diagnose(&request.operation_id).await.unwrap();
            assert_eq!(diagnostic.receipt.as_ref(), Some(&failed));
            assert_eq!(
                setup.inspect(&request.operation_id).await.unwrap(),
                Some(failed)
            );
            assert_eq!(
                diagnostic.action,
                hades_core::workspace_setup::RecoveryAction::RetryIdenticalRequest,
                "{tag}: {diagnostic:?}"
            );
            let store = ProfileStore::new(pool.clone()).unwrap();
            let before = store.load().await.ok();
            assert_eq!(
                setup.run(request).await.unwrap().status,
                SetupStatus::Ready,
                "{tag}"
            );
            let after = store.load().await.unwrap();
            assert_eq!(after.profile().state, ProfileState::Ready);
            if let Some(before) = before
                && before.profile().state == ProfileState::Ready
            {
                assert_eq!(after, before);
            }
            assert_eq!(
                pool.writer()
                    .get("collection/hades_workspace/count")
                    .await
                    .unwrap()["count"],
                5
            );
        })
        .await;
    }
}

/// A created name with an unrecorded acknowledgement is deliberately not recoverable automatically.
#[tokio::test]
async fn lost_create_acknowledgement_never_adopts_the_created_name() {
    with_temp_db("setup_lost_create", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let proxy = Proxy::new(pool.writer(), rule("POST", "database", None, 1));
        let setup = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        assert!(setup.run(request.clone()).await.is_err());
        assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);
        let diagnostic = setup.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(
            diagnostic.action,
            hades_core::workspace_setup::RecoveryAction::InspectRequired
        );
        assert_eq!(
            diagnostic.reason,
            hades_core::workspace_setup::RecoveryReason::CreationUnacknowledged
        );
        assert_eq!(diagnostic.database_exists, Some(true));
        assert_eq!(
            setup
                .inspect(&request.operation_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            SetupStatus::NeedsReconciliation
        );
        assert!(setup.run(request).await.is_err());
        assert!(pool.writer().get("database/current").await.is_ok());
        assert_eq!(
            pool.writer()
                .get("collection?excludeSystem=true")
                .await
                .unwrap()["result"],
            json!([])
        );
    })
    .await;
}

/// Cancellation retains the create checkpoint even after its in-flight HTTP effect reaches the server.
#[tokio::test]
async fn cancelled_creation_stops_at_reconciliation_on_restart() {
    with_temp_db("setup_cancel_create", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let mut fault = rule("POST", "database", None, 1);
        fault.hold = true;
        let proxy = Proxy::new(pool.writer(), fault);
        let setup = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        let task_setup = setup.clone();
        let task_request = request.clone();
        let task = tokio::spawn(async move { task_setup.run(task_request).await });
        tokio::time::timeout(Duration::from_secs(10), proxy.entered.notified())
            .await
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        proxy.release.notify_one();
        assert!(matches!(
            setup.run(request.clone()).await,
            Err(SetupError::Reconciliation(_))
        ));
        assert_eq!(
            setup
                .inspect(&request.operation_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            SetupStatus::NeedsReconciliation
        );
        assert_eq!(
            pool.writer()
                .get("collection?excludeSystem=true")
                .await
                .unwrap()["result"],
            json!([])
        );
    })
    .await;
}

/// Cooperating processes cannot overlap setup; other operations/owners cannot reuse its reservation.
#[tokio::test]
async fn competing_operations_and_authorities_cannot_take_over() {
    with_temp_db("setup_race", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let setup = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        let (entered, release) = peer.hold_next_verification();
        let task_setup = setup.clone();
        let task_request = request.clone();
        let task = tokio::spawn(async move { task_setup.run(task_request).await });
        tokio::time::timeout(Duration::from_secs(10), entered)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            setup.run(request.clone()).await,
            Err(SetupError::Busy)
        ));
        release.send(()).unwrap();
        task.await.unwrap().unwrap();
        let mut other = request.clone();
        other.operation_id = "other-op".into();
        assert!(setup.run(other).await.is_err());
        let stranger = WorkspaceSetup::new(
            root.path().into(),
            "test-installation".into(),
            "other-owner".into(),
            admin(&pool),
            peer.catalog(),
        )
        .unwrap();
        assert!(stranger.run(request.clone()).await.is_err());
        assert!(stranger.inspect(&request.operation_id).await.is_err());
        let elsewhere = WorkspaceSetup::new(
            root.path().into(),
            "other-installation".into(),
            "operator".into(),
            admin(&pool),
            peer.catalog(),
        )
        .unwrap();
        assert!(elsewhere.run(request).await.is_err());
    })
    .await;
}

/// Schema and ownership changes never turn partial setup into a ready/adopted database.
#[tokio::test]
async fn changed_ownership_schema_and_profile_are_preserved_and_refused() {
    for variant in [
        "marker",
        "incarnation",
        "edge",
        "unknown",
        "profile",
        "data",
        "ready_early",
        "extra_metadata",
    ] {
        with_temp_db(variant, Fixtures::Empty, move |pool| async move {
            remove_fixture_database(&pool).await;
            let fixture = Fixture::new(2);
            let peer = Peer::new(&fixture);
            let root = ledger();
            let mut injection = rule("POST", "collection", Some("documents"), 1);
            injection.before = true;
            let proxy = Proxy::new(pool.writer(), injection);
            let setup = setup(root.path(), proxy.admin.clone(), &peer);
            let request = request(&pool, &fixture);
            assert!(setup.run(request.clone()).await.is_err());
            assert_eq!(
                setup
                    .inspect(&request.operation_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .step,
                SetupStep::Schema
            );
            match variant {
                "marker" => {
                    pool.writer()
                        .patch(
                            "document/hades_setup/operation",
                            &json!({"payload":{"owner":"someone-else"}}),
                        )
                        .await
                        .unwrap();
                }
                "incarnation" => {
                    remove_fixture_database(&pool).await;
                    admin(&pool)
                        .post("database", &json!({"name":pool.database()}))
                        .await
                        .unwrap();
                    crud::create_collection(&pool, "authored", Some(2))
                        .await
                        .unwrap();
                    crud::insert_document(
                        &pool,
                        "authored",
                        &json!({"_key":"source","text":"new owner"}),
                    )
                    .await
                    .unwrap();
                }
                "edge" => {
                    crud::create_collection(&pool, "documents", Some(3))
                        .await
                        .unwrap();
                }
                "unknown" => {
                    crud::create_collection(&pool, "authored", Some(2))
                        .await
                        .unwrap();
                }
                "profile" => {
                    pool.writer()
                        .patch(
                            "document/hades_workspace/profile_1",
                            &json!({"payload":{"generation":"other"}}),
                        )
                        .await
                        .unwrap();
                }
                "data" => {
                    pool.writer()
                        .post(
                            "collection",
                            &json!({"name":"documents","type":2,"waitForSync":true}),
                        )
                        .await
                        .unwrap();
                    crud::insert_document(
                        &pool,
                        "documents",
                        &json!({"_key":"source","text":"do not adopt"}),
                    )
                    .await
                    .unwrap();
                }
                "ready_early" => {
                    let store = ProfileStore::new(pool.clone()).unwrap();
                    let snapshot = store.load().await.unwrap();
                    store.mark_ready(&snapshot).await.unwrap();
                }
                _ => {
                    crud::insert_document(
                        &pool,
                        "hades_workspace",
                        &json!({"_key":"extra","note":"preserve"}),
                    )
                    .await
                    .unwrap();
                }
            }
            assert!(setup.run(request.clone()).await.is_err(), "{variant}");
            assert_ne!(
                setup
                    .inspect(&request.operation_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .status,
                SetupStatus::Ready
            );
            match variant {
                "incarnation" => assert_eq!(
                    crud::get_document(&pool, "authored", "source")
                        .await
                        .unwrap()["text"],
                    "new owner"
                ),
                "data" => assert_eq!(
                    crud::get_document(&pool, "documents", "source")
                        .await
                        .unwrap()["text"],
                    "do not adopt"
                ),
                "extra_metadata" => assert_eq!(
                    crud::get_document(&pool, "hades_workspace", "extra")
                        .await
                        .unwrap()["note"],
                    "preserve"
                ),
                "profile" => assert_eq!(
                    crud::get_document(&pool, "hades_workspace", "profile_1")
                        .await
                        .unwrap()["payload"]["generation"],
                    "other"
                ),
                _ => {}
            }
            assert_eq!(
                peer.observations(),
                1,
                "unsafe resume contacted provider: {variant}"
            );
        })
        .await;
    }
}

/// Actual provider verification is repeated before readiness, not cached from the initial selection.
#[tokio::test]
async fn provider_drift_leaves_initializing_profile_until_explicit_compatible_retry() {
    with_temp_db("setup_provider_drift", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let proxy = Proxy::new(
            pool.writer(),
            rule("POST", "collection", Some("embeddings"), 1),
        );
        let setup = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        assert!(setup.run(request.clone()).await.is_err());
        let mut changed = peer.receipt();
        changed.artifact_md5 = serde_json::from_value(json!("0".repeat(32))).unwrap();
        peer.set_receipt(changed);
        assert!(matches!(
            setup.run(request.clone()).await,
            Err(SetupError::Provider(_))
        ));
        let store = ProfileStore::new(pool.clone()).unwrap();
        let before = store.load().await.unwrap();
        assert_eq!(before.profile().state, ProfileState::Initializing);
        let receipt = setup.inspect(&request.operation_id).await.unwrap().unwrap();
        assert_eq!(receipt.status, SetupStatus::Failed);
        assert_eq!(receipt.step, SetupStep::Verify);
        peer.set_receipt(fixture.receipt);
        assert_eq!(setup.run(request).await.unwrap().status, SetupStatus::Ready);
        assert_eq!(store.load().await.unwrap().profile().revision, 2);
    })
    .await;
}

/// Failure to persist the final receipt never undoes readiness or rewrites metadata on recovery.
#[tokio::test]
async fn failed_final_ledger_checkpoint_reconciles_ready_database() {
    with_temp_db("setup_ledger_failure", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let record_path = root.path().join("op_setup-one.json");
        let path = record_path.clone();
        let mut injection = rule("PUT", "transaction/", None, 4);
        injection.fail = false;
        injection.callback = Some(Box::new(move || {
            fs::set_permissions(path, fs::Permissions::from_mode(0o400)).unwrap()
        }));
        let proxy = Proxy::new(pool.writer(), injection);
        let setup = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        assert!(setup.run(request.clone()).await.is_err());
        assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);
        let store = ProfileStore::new(pool.clone()).unwrap();
        let before = store.load().await.unwrap();
        assert_eq!(before.profile().state, ProfileState::Ready);
        fs::set_permissions(record_path, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            setup
                .inspect(&request.operation_id)
                .await
                .unwrap()
                .unwrap()
                .step,
            SetupStep::Publish
        );
        assert_eq!(setup.run(request).await.unwrap().status, SetupStatus::Ready);
        assert_eq!(store.load().await.unwrap(), before);
    })
    .await;
}

/// A profile change during the final provider check cannot publish a stale readiness snapshot.
#[tokio::test]
async fn profile_change_during_verification_requires_new_admission() {
    with_temp_db("setup_profile_race", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let proxy = Proxy::new(
            pool.writer(),
            rule("POST", "collection", Some("embeddings"), 1),
        );
        let setup = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        assert!(setup.run(request.clone()).await.is_err());
        let (entered, release) = peer.hold_next_verification();
        let task_setup = setup.clone();
        let task_request = request.clone();
        let task = tokio::spawn(async move { task_setup.run(task_request).await });
        tokio::time::timeout(Duration::from_secs(10), entered)
            .await
            .unwrap()
            .unwrap();
        let profile = pool
            .writer()
            .get("document/hades_workspace/profile_1")
            .await
            .unwrap();
        pool.writer()
            .patch(
                "document/hades_workspace/profile_1",
                &json!({"payload":profile["payload"]}),
            )
            .await
            .unwrap();
        release.send(()).unwrap();
        assert!(task.await.unwrap().is_err());
        let store = ProfileStore::new(pool.clone()).unwrap();
        assert_eq!(
            store.load().await.unwrap().profile().state,
            ProfileState::Initializing
        );
        assert_eq!(setup.run(request).await.unwrap().status, SetupStatus::Ready);
    })
    .await;
}

/// A broken proxy fails the owning test even if a caller accepts the request error as expected.
#[tokio::test]
async fn proxy_failures_cannot_satisfy_expected_rejections() {
    with_temp_db("setup_proxy_failure", Fixtures::Empty, |pool| async move {
        let root = ledger();
        let missing = ArangoClient::with_socket(
            root.path().join("absent.sock"),
            pool.database(),
            "fixture",
            "fixture",
        );
        let proxy = Proxy::new(&missing, rule("DELETE", "unused", None, 1));
        assert_eq!(
            fs::metadata(proxy.admin.socket_path().unwrap().parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o700
        );
        assert!(matches!(
            proxy.admin.get("version").await,
            Err(hades_core::db::ArangoError::Api { status: 502, .. })
        ));
        assert_eq!(proxy.hits.load(Ordering::SeqCst), 0);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(proxy))).is_err());
    })
    .await;
}
