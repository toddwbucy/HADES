//! Database-local profile contracts; every write uses a disposable database.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use hades_core::{
    db::{ArangoError, ArangoPool, crud},
    embedding_profile::{
        ArtifactManifest, EmbeddingProfile, EmbeddingRole, ProfileState, ProviderReceipt,
        WorkloadKind,
        store::{COLLECTION, ProfileSnapshot, ProfileStore},
    },
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

/// Load synthetic artifact/behavior data without a model or provider service.
fn fixture() -> (EmbeddingProfile, ArtifactManifest, ProviderReceipt) {
    let value: Value =
        serde_json::from_str(include_str!("fixtures/embedding_profile.json")).unwrap();
    let mut profile: EmbeddingProfile = serde_json::from_value(value["profile"].clone()).unwrap();
    profile.state = ProfileState::Initializing;
    (
        profile,
        serde_json::from_value(value["manifest"].clone()).unwrap(),
        serde_json::from_value(value["receipt"].clone()).unwrap(),
    )
}

/// Initialize and activate synthetic policy, then create a separate write target.
async fn ready(pool: &ArangoPool) -> (ProfileStore, ProfileSnapshot) {
    let store = ProfileStore::new(pool.clone()).unwrap();
    let (profile, manifest, _) = fixture();
    let initial = store.initialize(profile, manifest).await.unwrap();
    let snapshot = store.mark_ready(&initial).await.unwrap();
    crud::create_collection(pool, "fixture_vectors", Some(2))
        .await
        .unwrap();
    (store, snapshot)
}

/// Count the exact target rows after commit or rollback.
async fn rows(pool: &ArangoPool) -> u64 {
    pool.writer()
        .get("collection/fixture_vectors/count")
        .await
        .unwrap()["count"]
        .as_u64()
        .unwrap()
}

/// Prove refusal happens before callback admission, even if transaction errors are caught.
async fn refused(store: &ProfileStore, snapshot: &ProfileSnapshot) -> ArangoError {
    let entered = Arc::new(AtomicBool::new(false));
    let observed = entered.clone();
    let result = store
        .guarded_write(
            snapshot,
            vec!["fixture_vectors".into()],
            move |_| async move {
                observed.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;
    assert!(
        !entered.load(Ordering::SeqCst),
        "rejected work entered the callback"
    );
    result.unwrap_err()
}

/// Verify persisted identity, readiness, and retained provenance agree with the binding contract.
#[tokio::test]
async fn roundtrip_retains_immutable_revisions_and_resolves_a_binding() {
    with_temp_db("profile_roundtrip", Fixtures::Empty, |pool| async move {
        let (profile, manifest, receipt) = fixture();
        let store = ProfileStore::new(pool.clone()).unwrap();
        assert!(store.load().await.is_err());
        let initial = store
            .initialize(profile.clone(), manifest.clone())
            .await
            .unwrap();
        assert_eq!(initial.profile(), &profile);
        assert_eq!(
            initial.manifest().digest().unwrap(),
            manifest.digest().unwrap()
        );
        assert!(
            initial
                .profile()
                .bind(
                    initial.manifest(),
                    WorkloadKind::Documents,
                    EmbeddingRole::Query,
                    &receipt
                )
                .is_err()
        );
        let snapshot = store.mark_ready(&initial).await.unwrap();
        assert_eq!(snapshot.profile().revision, 2);
        assert_eq!(snapshot.profile().state, ProfileState::Ready);
        assert_eq!(store.load().await.unwrap(), snapshot);
        let binding = snapshot
            .profile()
            .bind(
                snapshot.manifest(),
                WorkloadKind::Documents,
                EmbeddingRole::Query,
                &receipt,
            )
            .unwrap();
        assert_eq!(binding.profile_revision, 2);
        assert_eq!(binding.dimension, 384);
        assert_eq!(binding.expected, receipt);
        assert!(
            snapshot
                .profile()
                .bind(
                    snapshot.manifest(),
                    WorkloadKind::Code,
                    EmbeddingRole::Query,
                    &receipt
                )
                .is_err()
        );
        assert!(store.mark_ready(&initial).await.is_err());
        assert!(store.mark_ready(&snapshot).await.is_err());
        assert_eq!(
            crud::get_document(&pool, COLLECTION, "profile_1")
                .await
                .unwrap()["payload"],
            serde_json::to_value(profile).unwrap()
        );
        assert_eq!(
            pool.writer()
                .get(&format!("collection/{COLLECTION}/count"))
                .await
                .unwrap()["count"],
            5
        );
    })
    .await;
}

/// Reject invalid input before writes and preserve every existing source document.
#[tokio::test]
async fn invalid_or_existing_databases_are_never_adopted() {
    with_temp_db("profile_admission", Fixtures::Empty, |pool| async move {
        let (mut profile, manifest, _) = fixture();
        let store = ProfileStore::new(pool.clone()).unwrap();
        profile.state = ProfileState::Ready;
        assert!(
            store
                .initialize(profile.clone(), manifest.clone())
                .await
                .is_err()
        );
        profile.state = ProfileState::Initializing;
        profile
            .workloads
            .documents
            .as_mut()
            .unwrap()
            .space
            .representation
            .dimension = 0;
        assert!(store.initialize(profile, manifest.clone()).await.is_err());
        assert!(
            pool.writer()
                .get(&format!("collection/{COLLECTION}"))
                .await
                .is_err()
        );
        crud::create_collection(&pool, "existing_sources", Some(2))
            .await
            .unwrap();
        crud::insert_document(
            &pool,
            "existing_sources",
            &json!({"_key":"source", "text":"keep this authored source"}),
        )
        .await
        .unwrap();
        assert!(
            store
                .initialize(fixture().0, manifest)
                .await
                .unwrap_err()
                .to_string()
                .contains("new empty database")
        );
        assert!(
            pool.writer()
                .get(&format!("collection/{COLLECTION}"))
                .await
                .is_err()
        );
        assert_eq!(
            crud::get_document(&pool, "existing_sources", "source")
                .await
                .unwrap()["text"],
            "keep this authored source"
        );
    })
    .await;
}

/// Prove competing setup attempts cannot overwrite another initializer or readiness update.
#[tokio::test]
async fn racing_initializers_and_readiness_updates_have_one_winner() {
    with_temp_db("profile_race", Fixtures::Empty, |pool| async move {
        let (profile, manifest, _) = fixture();
        let store = ProfileStore::new(pool.clone()).unwrap();
        let (a, b) = tokio::join!(
            store.initialize(profile.clone(), manifest.clone()),
            store.initialize(profile, manifest)
        );
        assert_ne!(a.is_ok(), b.is_ok());
        let initial = a.or(b).unwrap();
        let (a, b) = tokio::join!(store.mark_ready(&initial), store.mark_ready(&initial));
        assert_ne!(a.is_ok(), b.is_ok());
        assert_eq!(store.load().await.unwrap(), a.or(b).unwrap());
    })
    .await;
}

/// Keep initial state intact when an immutable revision key already belongs to other data.
#[tokio::test]
async fn readiness_failure_leaves_initial_pointer_and_records_intact() {
    with_temp_db("profile_failed_ready", Fixtures::Empty, |pool| async move {
        let (profile, manifest, _) = fixture();
        let store = ProfileStore::new(pool.clone()).unwrap();
        let initial = store.initialize(profile, manifest).await.unwrap();
        // Simulate leftover/conflicting metadata; never overwrite an immutable revision.
        crud::insert_document(
            &pool,
            COLLECTION,
            &json!({"_key":"profile_2","sentinel":true}),
        )
        .await
        .unwrap();
        assert!(store.mark_ready(&initial).await.is_err());
        assert_eq!(store.load().await.unwrap(), initial);
        assert_eq!(
            crud::get_document(&pool, COLLECTION, "profile_2")
                .await
                .unwrap()["sentinel"],
            true
        );
        crud::create_collection(&pool, "fixture_vectors", Some(2))
            .await
            .unwrap();
        assert!(
            refused(&store, &initial)
                .await
                .to_string()
                .contains("not ready")
        );
        assert_eq!(rows(&pool).await, 0);
    })
    .await;
}

/// Commit prepared data atomically and undo partial writes on callback or policy failure.
#[tokio::test]
async fn guarded_writes_commit_together_and_roll_back_on_error_or_policy_mutation() {
    with_temp_db("profile_writes", Fixtures::Empty, |pool| async move {
        let (store, snapshot) = ready(&pool).await;
        store
            .guarded_write(
                &snapshot,
                vec!["fixture_vectors".into()],
                |client| async move {
                    client
                        .post(
                            "document/fixture_vectors",
                            &json!({"_key":"accepted","embedding":[1.0, 0.0]}),
                        )
                        .await?;
                    Ok(())
                },
            )
            .await
            .unwrap();
        let error = store
            .guarded_write(
                &snapshot,
                vec!["fixture_vectors".into()],
                |client| async move {
                    client
                        .post("document/fixture_vectors", &json!({"_key":"rejected"}))
                        .await?;
                    Err::<(), _>(ArangoError::Request("fixture failure after write".into()))
                },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("fixture failure"));
        assert_eq!(rows(&pool).await, 1);
        // Accidental metadata mutation aborts the data writes and the mutation together.
        assert!(
            store
                .guarded_write(
                    &snapshot,
                    vec!["fixture_vectors".into()],
                    |client| async move {
                        client
                            .post("document/fixture_vectors", &json!({"_key":"also_rejected"}))
                            .await?;
                        client
                            .patch(
                                &format!("document/{COLLECTION}/embedding_profile"),
                                &json!({"payload":{"generation":"unexpected"}}),
                            )
                            .await?;
                        Ok(())
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(rows(&pool).await, 1);
        assert_eq!(store.load().await.unwrap(), snapshot);
    })
    .await;
}

/// Separate database bindings and detect a metadata change even after its content is restored.
#[tokio::test]
async fn stale_pins_and_cross_database_snapshots_never_enter_the_callback() {
    with_temp_db("profile_one", Fixtures::Empty, |one| async move {
        let (store_one, snapshot_one) = ready(&one).await;
        with_temp_db("profile_two", Fixtures::Empty, move |two| async move {
            let store_two = ProfileStore::new(two.clone()).unwrap();
            let (mut profile, manifest, _) = fixture();
            profile.model.id = "fixture/other-model".into();
            profile.runtime_binding.provider_id = "fixture-other-provider".into();
            profile
                .workloads
                .documents
                .as_mut()
                .unwrap()
                .space
                .representation
                .dimension = 2;
            let initial = store_two.initialize(profile, manifest).await.unwrap();
            let snapshot_two = store_two.mark_ready(&initial).await.unwrap();
            crud::create_collection(&two, "fixture_vectors", Some(2))
                .await
                .unwrap();
            assert_eq!(
                snapshot_one
                    .profile()
                    .workloads
                    .documents
                    .as_ref()
                    .unwrap()
                    .space
                    .representation
                    .dimension,
                384
            );
            assert_eq!(
                snapshot_two
                    .profile()
                    .workloads
                    .documents
                    .as_ref()
                    .unwrap()
                    .space
                    .representation
                    .dimension,
                2
            );
            assert!(
                refused(&store_two, &snapshot_one)
                    .await
                    .to_string()
                    .contains("stale database profile")
            );
            // A real change-and-restore preserves content but invalidates the old revision pin.
            let old = crud::get_document(&one, COLLECTION, "embedding_profile")
                .await
                .unwrap();
            crud::update_document(
                &one,
                COLLECTION,
                "embedding_profile",
                &json!({"payload":{"generation":"temporary"}}),
            )
            .await
            .unwrap();
            crud::replace_document(&one, COLLECTION, "embedding_profile", &old)
                .await
                .unwrap();
            assert!(store_one.load().await.is_ok());
            assert!(
                refused(&store_one, &snapshot_one)
                    .await
                    .to_string()
                    .contains("stale database profile")
            );
            assert_eq!(rows(&one).await, 0);
            assert_eq!(rows(&two).await, 0);
            assert_eq!(store_two.load().await.unwrap(), snapshot_two);
        })
        .await;
    })
    .await;
}

/// Pin immutable record revisions as well as the active pointer revision.
#[tokio::test]
async fn restored_immutable_records_invalidate_admitted_work() {
    with_temp_db(
        "profile_immutable_pins",
        Fixtures::Empty,
        |pool| async move {
            let (store, mut snapshot) = ready(&pool).await;
            let keys = [
                "profile_2".to_owned(),
                format!(
                    "manifest_{}",
                    snapshot.profile().model.manifest_md5.as_str()
                ),
                format!(
                    "generation_{}",
                    Sha256::digest(snapshot.profile().generation.as_bytes())
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                ),
            ];
            for key in keys {
                let old = crud::get_document(&pool, COLLECTION, &key).await.unwrap();
                crud::update_document(&pool, COLLECTION, &key, &json!({"temporary":true}))
                    .await
                    .unwrap();
                crud::replace_document(&pool, COLLECTION, &key, &old)
                    .await
                    .unwrap();
                let fresh = store.load().await.unwrap();
                assert_eq!(fresh.profile(), snapshot.profile());
                assert_eq!(fresh.manifest(), snapshot.manifest());
                assert!(
                    refused(&store, &snapshot)
                        .await
                        .to_string()
                        .contains("stale database profile")
                );
                snapshot = fresh;
            }
            assert_eq!(rows(&pool).await, 0);
        },
    )
    .await;
}

/// Refuse malformed or incomplete persistence without guessing or admitting a write.
#[tokio::test]
async fn corrupted_missing_or_oversize_records_fail_closed() {
    with_temp_db("profile_corrupt", Fixtures::Empty, |pool| async move {
        let (store, snapshot) = ready(&pool).await;
        let manifest_key = format!(
            "manifest_{}",
            snapshot.profile().model.manifest_md5.as_str()
        );
        let generation_key = format!(
            "generation_{}",
            Sha256::digest(snapshot.profile().generation.as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        for (key, patch) in [
            ("embedding_profile", json!({"format":"unknown-v2"})),
            ("embedding_profile", json!({"unexpected":"field"})),
            ("embedding_profile", json!({"payload":{"revision":99}})),
            (
                "profile_2",
                json!({"payload":{"generation":"other-generation"}}),
            ),
            (
                "profile_2",
                json!({"payload":{"workloads":{"documents":{"space":{
                "representation":{"dimension":0}}}}}}),
            ),
            (manifest_key.as_str(), json!({"payload":{"files":[]}})),
            (
                generation_key.as_str(),
                json!({"payload":{"documents":null}}),
            ),
            ("profile_2", json!({"padding":"x".repeat(80 * 1024)})),
        ] {
            let old = crud::get_document(&pool, COLLECTION, key).await.unwrap();
            crud::update_document(&pool, COLLECTION, key, &patch)
                .await
                .unwrap();
            assert!(store.load().await.is_err(), "accepted corrupt {key}");
            crud::replace_document(&pool, COLLECTION, key, &old)
                .await
                .unwrap();
            assert!(store.load().await.is_ok());
        }
        crud::delete_document(&pool, COLLECTION, &manifest_key)
            .await
            .unwrap();
        assert!(store.load().await.is_err());
        refused(&store, &snapshot).await;
        assert_eq!(rows(&pool).await, 0);
    })
    .await;
}

/// Drain cancellation through the transaction owner before allowing another profile snapshot.
#[tokio::test]
async fn cancelled_guarded_work_rolls_back_before_the_next_snapshot() {
    with_temp_db("profile_cancel", Fixtures::Empty, |pool| async move {
        let (store, snapshot) = ready(&pool).await;
        let writer = store.clone();
        let expected = snapshot.clone();
        let (entered, wait_entered) = oneshot::channel();
        let task = tokio::spawn(async move {
            writer
                .guarded_write(
                    &expected,
                    vec!["fixture_vectors".into()],
                    |client| async move {
                        client
                            .post("document/fixture_vectors", &json!({"_key":"interrupted"}))
                            .await?;
                        entered.send(()).unwrap();
                        std::future::pending::<()>().await;
                        Ok(())
                    },
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), wait_entered)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        // The next lock acquisition waits for the transaction owner's abort.
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(10), store.load())
                .await
                .unwrap()
                .unwrap(),
            snapshot
        );
        assert_eq!(rows(&pool).await, 0);
    })
    .await;
}
