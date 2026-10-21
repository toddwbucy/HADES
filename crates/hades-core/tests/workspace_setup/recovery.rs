//! Diagnostics and restored ledger copies cannot grant ownership of another database incarnation.
use super::*;
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
