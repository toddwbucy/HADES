//! Diagnostics and restored ledger copies cannot grant ownership of another database incarnation.
use super::*;
use hades_core::db::query::{ExecutionTarget, query};
use hades_core::workspace_setup::{RecoveryAction, RecoveryReason};
use std::collections::BTreeMap;

fn ledger_bytes(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

/// Capture every metadata payload and revision to detect diagnostic writes.
async fn metadata_rows(pool: &ArangoPool) -> Vec<serde_json::Value> {
    query(
        pool,
        "FOR row IN hades_workspace SORT row._key RETURN row",
        None,
        None,
        false,
        ExecutionTarget::Writer,
    )
    .await
    .unwrap()
    .results
}

/// Capture collection definitions and counts to detect diagnostic repair or creation.
async fn schema_state(pool: &ArangoPool) -> BTreeMap<String, serde_json::Value> {
    let inventory = pool
        .writer()
        .get("collection?excludeSystem=true")
        .await
        .unwrap();
    let mut state = BTreeMap::new();
    for entry in inventory["result"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let properties = pool
            .writer()
            .get(&format!("collection/{name}/properties"))
            .await
            .unwrap();
        let indexes = pool
            .writer()
            .get(&format!("index?collection={name}"))
            .await
            .unwrap();
        let count = crud::count_collection(pool, name).await.unwrap();
        state.insert(
            name.into(),
            json!({"properties":properties,"indexes":indexes,"count":count}),
        );
    }
    state
}

#[tokio::test]
async fn owner_index_creation_resumes_but_ready_schema_is_never_repaired_implicitly() {
    for before in [true, false] {
        with_temp_db(
            "setup_owner_index",
            Fixtures::Empty,
            move |pool| async move {
                remove_fixture_database(&pool).await;
                let fixture = Fixture::new(2);
                let peer = Peer::new(&fixture);
                let root = ledger();
                let mut fault = rule("POST", "index?collection=chunks", None, 1);
                fault.before = before;
                let proxy = Proxy::new(pool.writer(), fault);
                let service = setup(root.path(), proxy.admin.clone(), &peer);
                let request = request(&pool, &fixture);
                assert!(service.run(request.clone()).await.is_err());
                assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);
                let partial = schema_state(&pool).await;
                assert_eq!(
                    partial["chunks"]["indexes"]["indexes"]
                        .as_array()
                        .unwrap()
                        .len(),
                    if before { 1 } else { 2 }
                );
                let report = service.diagnose(&request.operation_id).await.unwrap();
                assert_eq!(report.action, RecoveryAction::RetryIdenticalRequest);
                assert_eq!(
                    schema_state(&pool).await,
                    partial,
                    "diagnosis must not finish the index write"
                );
                assert_eq!(
                    service.run(request.clone()).await.unwrap().status,
                    SetupStatus::Ready
                );
                let ready = schema_state(&pool).await;
                for name in ["chunks", "embeddings"] {
                    let indexes = ready[name]["indexes"]["indexes"].as_array().unwrap();
                    assert_eq!(indexes.len(), 2);
                    assert!(indexes.iter().any(|index| index["type"] == "persistent"
                        && index["fields"] == json!(["parent_key"])
                        && index["unique"] == false
                        && index["sparse"] == false));
                }
                ProfiledDocuments::connect(pool.clone(), &peer.catalog())
                    .await
                    .unwrap();
                let index = ready["embeddings"]["indexes"]["indexes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|index| index["type"] == "persistent")
                    .unwrap();
                hades_core::db::index::drop_index(&pool, index["id"].as_str().unwrap())
                    .await
                    .unwrap();
                let damaged = schema_state(&pool).await;
                assert_eq!(
                    service
                        .diagnose(&request.operation_id)
                        .await
                        .unwrap()
                        .action,
                    RecoveryAction::InspectRequired
                );
                assert!(service.run(request).await.is_err());
                assert!(
                    ProfiledDocuments::connect(pool.clone(), &peer.catalog())
                        .await
                        .is_err()
                );
                assert_eq!(
                    schema_state(&pool).await,
                    damaged,
                    "ready setup must not silently repair missing indexes"
                );
            },
        )
        .await;
    }
}

/// Missing collections can resume, but every existing collection must retain setup's exact schema.
#[tokio::test]
async fn schema_checkpoint_refuses_incompatible_collections_without_repair_or_hold() {
    for name in ["documents", "chunks", "embeddings", "hades_workspace"] {
        for defect in ["nonempty", "durability", "indexes"] {
            // Workspace metadata legitimately has four records; only data collections must be empty.
            if name == "hades_workspace" && defect == "nonempty" {
                continue;
            }
            with_temp_db(
                "setup_partial_schema",
                Fixtures::Empty,
                move |pool| async move {
                    remove_fixture_database(&pool).await;
                    let fixture = Fixture::new(2);
                    let peer = Peer::new(&fixture);
                    let root = ledger();
                    let interrupted = if name == "hades_workspace" {
                        "documents"
                    } else {
                        name
                    };
                    let proxy = Proxy::new(
                        pool.writer(),
                        rule("POST", "collection", Some(interrupted), 1),
                    );
                    let service = setup(root.path(), proxy.admin.clone(), &peer);
                    let request = request(&pool, &fixture);
                    assert!(service.run(request.clone()).await.is_err());
                    assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);
                    let receipt = service
                        .inspect(&request.operation_id)
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(receipt.step, SetupStep::Schema);
                    assert_eq!(receipt.status, SetupStatus::Failed);
                    let bytes = ledger_bytes(root.path());
                    let metadata = metadata_rows(&pool).await;
                    let observations = peer.observations();
                    let partial = schema_state(&pool).await;
                    let retry = service.diagnose(&request.operation_id).await.unwrap();
                    assert_eq!(retry.action, RecoveryAction::RetryIdenticalRequest);
                    assert_eq!(retry.reason, RecoveryReason::InitializationIncomplete);
                    assert_eq!(schema_state(&pool).await, partial);

                    match defect {
                        "nonempty" => {
                            crud::insert_document(
                                &pool,
                                name,
                                &json!({"_key":"preserve","text":"independent source"}),
                            )
                            .await
                            .unwrap();
                        }
                        "durability" => {
                            pool.writer()
                                .put(
                                    &format!("collection/{name}/properties"),
                                    &json!({"waitForSync":false}),
                                )
                                .await
                                .unwrap();
                        }
                        "indexes" => {
                            pool.writer()
                                .post(
                                    &format!("index?collection={name}"),
                                    &json!({"type":"persistent","fields":["text"],"unique":false}),
                                )
                                .await
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                    let authored = if defect == "nonempty" {
                        Some(crud::get_document(&pool, name, "preserve").await.unwrap())
                    } else {
                        None
                    };
                    let damaged = schema_state(&pool).await;
                    let report = service.diagnose(&request.operation_id).await.unwrap();
                    assert_eq!(
                        report.action,
                        RecoveryAction::InspectRequired,
                        "{name}/{defect}"
                    );
                    assert_eq!(
                        report.reason,
                        RecoveryReason::SchemaNotVerified,
                        "{name}/{defect}"
                    );
                    assert_eq!(report.receipt, Some(receipt.clone()));
                    assert_eq!(
                        service.inspect(&request.operation_id).await.unwrap(),
                        Some(receipt)
                    );
                    assert_eq!(ledger_bytes(root.path()), bytes);
                    assert_eq!(metadata_rows(&pool).await, metadata);
                    assert_eq!(schema_state(&pool).await, damaged);
                    assert_eq!(peer.observations(), observations);
                    if let Some(authored) = authored {
                        assert_eq!(
                            crud::get_document(&pool, name, "preserve").await.unwrap(),
                            authored
                        );
                    }
                },
            )
            .await;
        }
    }
}

/// Missing data collections are retryable only while the initialized profile matches durable intent.
#[tokio::test]
async fn schema_checkpoint_requires_intact_matching_metadata_without_changing_it() {
    for defect in [
        "malformed",
        "different_profile",
        "missing_profile",
        "extra_record",
    ] {
        with_temp_db(
            "setup_schema_metadata",
            Fixtures::Empty,
            move |pool| async move {
                remove_fixture_database(&pool).await;
                let fixture = Fixture::new(2);
                let peer = Peer::new(&fixture);
                let root = ledger();
                let mut failure = rule("POST", "collection", Some("documents"), 1);
                failure.before = true;
                let proxy = Proxy::new(pool.writer(), failure);
                let service = setup(root.path(), proxy.admin.clone(), &peer);
                let request = request(&pool, &fixture);
                assert!(service.run(request.clone()).await.is_err());
                assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);
                let receipt = service
                    .inspect(&request.operation_id)
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(receipt.step, SetupStep::Schema);
                let bytes = ledger_bytes(root.path());
                let observations = peer.observations();
                let retry = service.diagnose(&request.operation_id).await.unwrap();
                assert_eq!(retry.action, RecoveryAction::RetryIdenticalRequest);
                assert_eq!(retry.reason, RecoveryReason::InitializationIncomplete);
                let store = ProfileStore::new(pool.clone()).unwrap();
                let saved = store.load().await.unwrap();
                let profile_path = "document/hades_workspace/profile_1";
                let mut profile = pool.writer().get(profile_path).await.unwrap();
                match defect {
                    "malformed" => {
                        profile["payload"] = json!("damaged profile");
                        pool.writer()
                            .patch(profile_path, &json!({"payload":profile["payload"]}))
                            .await
                            .unwrap();
                        assert!(store.load().await.is_err());
                    }
                    "different_profile" => {
                        profile["payload"]["runtime_binding"]["provider_id"] =
                            json!("another-provider");
                        pool.writer()
                            .patch(profile_path, &json!({"payload":profile["payload"]}))
                            .await
                            .unwrap();
                        let changed = store.load().await.unwrap();
                        assert_ne!(changed.profile(), saved.profile());
                    }
                    "missing_profile" => {
                        pool.writer().delete(profile_path).await.unwrap();
                        assert!(store.load().await.is_err());
                    }
                    "extra_record" => {
                        crud::insert_document(
                            &pool,
                            "hades_workspace",
                            &json!({"_key":"unexpected","text":"preserve me"}),
                        )
                        .await
                        .unwrap();
                        assert_eq!(store.load().await.unwrap(), saved);
                    }
                    _ => unreachable!(),
                }
                let contents = metadata_rows(&pool).await;
                let report = service.diagnose(&request.operation_id).await.unwrap();
                assert_eq!(report.action, RecoveryAction::InspectRequired, "{defect}");
                assert_eq!(
                    report.reason,
                    RecoveryReason::MetadataNotVerified,
                    "{defect}"
                );
                assert_eq!(report.receipt, Some(receipt));
                assert_eq!(ledger_bytes(root.path()), bytes);
                assert_eq!(peer.observations(), observations);
                assert_eq!(metadata_rows(&pool).await, contents);
                for name in ["documents", "chunks", "embeddings"] {
                    assert!(
                        pool.writer()
                            .get(&format!("collection/{name}/properties"))
                            .await
                            .unwrap_err()
                            .is_not_found()
                    );
                }
            },
        )
        .await;
    }
}

#[tokio::test]
async fn pending_marker_retry_never_adopts_foreign_contents() {
    with_temp_db("setup_pending_marker", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let proxy = Proxy::new(
            pool.writer(),
            rule("POST", "collection", Some("hades_setup"), 1),
        );
        let service = setup(root.path(), proxy.admin.clone(), &peer);
        let request = request(&pool, &fixture);
        assert!(service.run(request.clone()).await.is_err());
        let bytes = ledger_bytes(root.path());
        let retry = service.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(retry.action, RecoveryAction::RetryIdenticalRequest);
        assert_eq!(retry.database_id_matches, Some(true));
        assert!(
            !retry.ownership_verified,
            "the absent marker is not a verified marker"
        );
        crud::create_collection(&pool, "authored", Some(2))
            .await
            .unwrap();
        let authored = crud::insert_document(
            &pool,
            "authored",
            &json!({"_key":"preserve","text":"independent source"}),
        )
        .await
        .unwrap();
        let report = service.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(report.action, RecoveryAction::InspectRequired);
        assert_eq!(report.reason, RecoveryReason::OwnershipNotVerified);
        assert_eq!(ledger_bytes(root.path()), bytes);
        assert_eq!(
            crud::get_document(&pool, "authored", "preserve")
                .await
                .unwrap()["_rev"],
            authored["_rev"]
        );
        assert_eq!(
            pool.writer()
                .get("collection/hades_setup/count")
                .await
                .unwrap()["count"],
            0
        );
    })
    .await;
}

#[tokio::test]
async fn ready_diagnostics_preserve_records_and_do_not_load_providers() {
    with_temp_db("setup_diagnose", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::late();
        let peer = Peer::new(&fixture);
        let root = ledger();
        let service = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        let receipt = service.run(request.clone()).await.unwrap();
        let store = ProfileStore::new(pool.clone()).unwrap();
        let snapshot = store.load().await.unwrap();
        let bytes = ledger_bytes(root.path());
        let observations = peer.observations();
        let report = service.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(report.action, RecoveryAction::None);
        assert_eq!(report.reason, RecoveryReason::Ready);
        assert_eq!(report.database_id_matches, Some(true));
        assert!(report.ownership_verified);
        assert_eq!(report.profile_state, Some(ProfileState::Ready));
        assert_eq!(report.receipt, Some(receipt));
        assert_eq!(
            service.diagnose("unknown-operation").await.unwrap().reason,
            RecoveryReason::OperationMissing
        );
        assert_eq!(ledger_bytes(root.path()), bytes);
        assert_eq!(store.load().await.unwrap(), snapshot);
        assert_eq!(peer.observations(), observations);
        let serialized = serde_json::to_string(&report).unwrap();
        for secret in [
            "socket",
            "password",
            "registry",
            root.path().to_str().unwrap(),
        ] {
            assert!(!serialized.contains(secret));
        }
        let other = WorkspaceSetup::new(
            root.path().into(),
            "test-installation".into(),
            "another-owner".into(),
            admin(&pool),
            peer.catalog(),
        )
        .unwrap();
        assert!(other.diagnose(&request.operation_id).await.is_err());
        let claim_path = root.path().join(format!("db_{}.json", pool.database()));
        let claim_bytes = fs::read(&claim_path).unwrap();
        fs::remove_file(&claim_path).unwrap();
        let report = service.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(report.reason, RecoveryReason::ReservationNotVerified);
        assert!(
            !claim_path.exists(),
            "diagnosis recreated the missing reservation"
        );
        fs::write(&claim_path, &claim_bytes).unwrap();
        fs::set_permissions(&claim_path, fs::Permissions::from_mode(0o600)).unwrap();
        // Schema corruption is reported while the durable historical receipt remains intact.
        pool.writer()
            .put(
                "collection/chunks/properties",
                &json!({"waitForSync":false}),
            )
            .await
            .unwrap();
        let report = service.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(report.reason, RecoveryReason::SchemaNotVerified);
        assert_eq!(report.action, RecoveryAction::InspectRequired);
        assert_eq!(ledger_bytes(root.path()), bytes);
        assert_eq!(store.load().await.unwrap(), snapshot);
        assert_eq!(
            pool.writer()
                .get("collection/chunks/properties")
                .await
                .unwrap()["waitForSync"],
            false
        );
    })
    .await;
}

#[tokio::test]
async fn a_restored_ledger_preserves_scope_and_cannot_adopt_a_recreated_database() {
    with_temp_db("setup_recovery_copy", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let fixture = Fixture::new(2);
        let peer = Peer::new(&fixture);
        let root = ledger();
        let original = setup(root.path(), admin(&pool), &peer);
        let request = request(&pool, &fixture);
        original.run(request.clone()).await.unwrap();
        // No setup operation is active while the complete private ledger is copied.
        let restored = ledger();
        for entry in fs::read_dir(root.path()).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), restored.path().join(entry.file_name())).unwrap();
        }
        let bytes = ledger_bytes(restored.path());
        let recovery = setup(restored.path(), admin(&pool), &peer);
        assert_eq!(
            recovery
                .diagnose(&request.operation_id)
                .await
                .unwrap()
                .reason,
            RecoveryReason::Ready
        );
        let another_installation = WorkspaceSetup::new(
            restored.path().into(),
            "replacement-installation".into(),
            "operator".into(),
            admin(&pool),
            peer.catalog(),
        )
        .unwrap();
        assert!(
            another_installation
                .diagnose(&request.operation_id)
                .await
                .is_err()
        );
        remove_fixture_database(&pool).await;
        let missing = recovery.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(missing.reason, RecoveryReason::DatabaseMissing);
        admin(&pool)
            .post("database", &json!({"name":pool.database()}))
            .await
            .unwrap();
        crud::create_collection(&pool, "authored", Some(2))
            .await
            .unwrap();
        let authored = crud::insert_document(
            &pool,
            "authored",
            &json!({"_key":"preserve","text":"independent source"}),
        )
        .await
        .unwrap();
        let changed = recovery.diagnose(&request.operation_id).await.unwrap();
        assert_eq!(changed.reason, RecoveryReason::DatabaseIncarnationChanged);
        assert_eq!(changed.action, RecoveryAction::InspectRequired);
        assert_eq!(changed.database_id_matches, Some(false));
        assert_eq!(ledger_bytes(restored.path()), bytes);
        assert_eq!(
            crud::get_document(&pool, "authored", "preserve")
                .await
                .unwrap()["_rev"],
            authored["_rev"]
        );
        assert_eq!(peer.observations(), 2);
    })
    .await;
}
