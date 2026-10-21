//! Public dispatch uses database identity; legacy paths cannot bypass a saved pairing.
//! All writes use disposable databases and synthetic private providers.
use hades_core::{
    config::HadesConfig,
    db::{ArangoPool, crud, query},
    dispatch::{DaemonCommand, dispatch},
    embedding_profile::store::{COLLECTION, ProfileStore},
    pipeline::profiled_documents::DocumentInput,
    service::{ConnectionPolicy, handle_request},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::time::Duration;

#[allow(dead_code)]
#[path = "common/profiled_provider.rs"]
mod provider;
use provider::{Fixture, Peer};

// These tests share dispatch's four-search process budget, not database state.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn config(pool: &ArangoPool, peer: &Peer) -> HadesConfig {
    let mut config = HadesConfig::with_database(pool.database());
    config.registered_providers = Some(peer.configuration());
    config.embedding.service.socket = "/absent/legacy-provider.sock".into();
    config.embedding.model.name = "wrong/legacy-model".into();
    config.embedding.model.dimension = 777;
    config
}

async fn request(
    pool: &ArangoPool,
    config: &HadesConfig,
    params: Value,
) -> hades_core::dispatch::DaemonResponse {
    handle_request(
        pool,
        config,
        ConnectionPolicy::agent_only(),
        &serde_json::to_vec(&json!({"command":"db.query", "params":params})).unwrap(),
        Duration::from_secs(15),
    )
    .await
}

async fn search(pool: &ArangoPool, config: &HadesConfig) -> Value {
    let response = request(pool, config, json!({"text":"apple","limit":1})).await;
    assert!(response.success, "{response:?}");
    response.data.unwrap()
}

async fn insert(pool: &ArangoPool, fixture: &Fixture, peer: &Peer, source: &str) -> String {
    let documents = fixture.connect(pool, peer).await;
    let prepared = documents
        .prepare(
            DocumentInput {
                source_id: source.into(),
                text: "apple café\nbanana 東京".into(),
            },
            false,
        )
        .await
        .unwrap();
    documents.commit(prepared).await.unwrap().document_key
}

async fn rows(pool: &ArangoPool) -> Value {
    let mut saved = json!({});
    for collection in [COLLECTION, "documents", "chunks", "embeddings"] {
        saved[collection] = json!(
            query::query(
                pool,
                "FOR r IN @@c SORT r._key RETURN r",
                Some(&json!({"@c":collection})),
                None,
                false,
                query::ExecutionTarget::Writer
            )
            .await
            .unwrap()
            .results
        );
    }
    saved
}

#[tokio::test]
async fn service_search_returns_verified_identity_and_matching_source_text() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_search", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let key = insert(&pool, &fixture, &peer, "notes/日本語.md").await;
        let before = rows(&pool).await;
        let result = search(&pool, &config(&pool, &peer)).await;
        assert_eq!(result["dimension"], 2);
        assert_eq!(result["model"], fixture.profile.model.id);
        assert_eq!(result["embedding_profile_state"], "ready");
        assert_eq!(
            result["embedding_binding"]["expected"],
            json!(fixture.receipt)
        );
        assert_eq!(result["results"][0]["parent_key"], key);
        assert_eq!(result["results"][0]["source_id"], "notes/日本語.md");
        assert_eq!(result["results"][0]["text"], "apple café\n");
        assert_eq!(
            result["results"][0]["span"]["end_byte"],
            "apple café\n".len()
        );
        assert_eq!(result["results"][0]["total_chunks"], 2);
        assert_eq!(result["results"][0]["score"], 1.0);
        assert_eq!(result["scanned"], 2);
        assert_eq!(
            rows(&pool).await,
            before,
            "search must not rewrite database records"
        );
        assert_eq!(peer.requests().last().unwrap()["task"], "retrieval.query");
    })
    .await;
}

#[tokio::test]
async fn missing_routes_and_unsupported_modes_do_not_contact_any_provider() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_policy", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let mut config = config(&pool, &peer);
        for params in [
            json!({"text":"apple","collection":"codebase"}),
            json!({"text":"apple","hybrid":true}),
            json!({"text":"apple","structural":true}),
            json!({"text":"apple","rerank":true}),
        ] {
            let response = request(&pool, &config, params).await;
            assert!(!response.success);
            assert_eq!(response.error_code.as_deref(), Some("INVALID_PARAMS"));
        }
        config.registered_providers = None;
        let response = request(&pool, &config, json!({"text":"apple"})).await;
        assert!(response.error.unwrap().contains("registered_providers"));
        config.registered_providers = Some(peer.configuration());
        config
            .registered_providers
            .as_mut()
            .unwrap()
            .providers
            .clear();
        let response = request(&pool, &config, json!({"text":"apple"})).await;
        assert!(response.error.unwrap().contains("allowlist"));
        assert_eq!(peer.observations(), 0);
        assert!(peer.requests().is_empty());
    })
    .await;
}

#[tokio::test]
async fn incomplete_or_corrupt_profiles_never_reenter_legacy_search_or_jobs() {
    let _serial = SERIAL.lock().await;
    for stage in [
        "marker_only",
        "empty_metadata",
        "initializing",
        "missing_pointer",
        "unknown_format",
    ] {
        with_temp_db(
            "public_bound_broken",
            Fixtures::Empty,
            move |pool| async move {
                let fixture = Fixture::new(2);
                let peer = Peer::new(&fixture);
                match stage {
                    "marker_only" => {
                        crud::create_collection(&pool, "hades_setup", Some(2))
                            .await
                            .unwrap();
                    }
                    "empty_metadata" => {
                        crud::create_collection(&pool, COLLECTION, Some(2))
                            .await
                            .unwrap();
                    }
                    "initializing" => {
                        ProfileStore::new(pool.clone())
                            .unwrap()
                            .initialize(fixture.profile.clone(), fixture.manifest.clone())
                            .await
                            .unwrap();
                    }
                    _ => {
                        fixture.install(&pool).await;
                        let path = format!("document/{COLLECTION}/embedding_profile");
                        if stage == "missing_pointer" {
                            pool.writer().delete(&path).await.unwrap();
                        } else {
                            pool.writer()
                                .patch(&path, &json!({"format":"future-unknown"}))
                                .await
                                .unwrap();
                        }
                    }
                }
                let config = config(&pool, &peer);
                let response = request(&pool, &config, json!({"text":"apple"})).await;
                assert!(!response.success, "{stage}: {response:?}");
                let directory = tempfile::tempdir().unwrap();
                let command: DaemonCommand =
                    serde_json::from_value(json!({"command":"ingest.start",
                "params":{"path":directory.path()}}))
                    .unwrap();
                let error = dispatch(&pool, &config, command)
                    .await
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains("model-bound or initializing"),
                    "{stage}: {error}"
                );
                assert!(
                    !crud::list_collections(&pool, false)
                        .await
                        .unwrap()
                        .iter()
                        .any(|c| c.name == "hades_ingest_jobs")
                );
                assert_eq!(peer.observations(), 0);
                assert!(peer.requests().is_empty());
            },
        )
        .await;
    }
}

#[tokio::test]
async fn provider_reload_is_reverified_and_mid_request_drift_discards_search() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_reload", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        insert(&pool, &fixture, &peer, "source").await;
        let config = config(&pool, &peer);
        search(&pool, &config).await;
        let mut reload = peer.receipt();
        reload.serving_generation = "second-load".into();
        peer.set_receipt(reload.clone());
        let result = search(&pool, &config).await;
        assert_eq!(
            result["embedding_binding"]["expected"]["serving_generation"],
            "second-load"
        );
        let (entered, release) = peer.hold_next_verification();
        let other_pool = pool.clone();
        let task =
            tokio::spawn(
                async move { request(&other_pool, &config, json!({"text":"apple"})).await },
            );
        entered.await.unwrap();
        reload.serving_generation = "third-load".into();
        peer.set_receipt(reload);
        release.send(()).unwrap();
        let response = task.await.unwrap();
        assert!(!response.success);
        assert!(response.data.is_none());
    })
    .await;
}

#[tokio::test]
async fn profile_drift_during_inference_cannot_return_results() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        insert(&pool, &fixture, &peer, "source").await;
        let config = config(&pool, &peer);
        let (entered, release) = peer.hold_next();
        let other_pool = pool.clone();
        let task =
            tokio::spawn(
                async move { request(&other_pool, &config, json!({"text":"apple"})).await },
            );
        entered.await.unwrap();
        let path = format!("document/{COLLECTION}/embedding_profile");
        let original = pool.writer().get(&path).await.unwrap();
        pool.writer()
            .patch(&path, &json!({"payload":original["payload"]}))
            .await
            .unwrap();
        release.send(()).unwrap();
        let response = task.await.unwrap();
        assert!(!response.success);
        assert!(response.data.is_none());
    })
    .await;
}

#[tokio::test]
async fn mismatched_stored_vectors_and_source_text_fail_without_partial_results() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_rows", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let key = insert(&pool, &fixture, &peer, "source").await;
        let config = config(&pool, &peer);
        let result = search(&pool, &config).await;
        let chunk_key = result["results"][0]["chunk_key"].as_str().unwrap();
        for (collection, key, patch) in [
            ("chunks", chunk_key, json!({"text":"altered text"})),
            (
                "documents",
                key.as_str(),
                json!({"full_text":"altered source"}),
            ),
            (
                "embeddings",
                chunk_key,
                json!({"model":"same-width-foreign-model"}),
            ),
        ] {
            let path = format!("document/{collection}/{key}");
            let original = pool.writer().get(&path).await.unwrap();
            pool.writer().patch(&path, &patch).await.unwrap();
            let response = request(&pool, &config, json!({"text":"apple"})).await;
            assert!(!response.success, "{collection}");
            assert!(response.data.is_none());
            pool.writer().put(&path, &original).await.unwrap();
        }
    })
    .await;
}

#[tokio::test]
async fn nonvector_reads_work_without_a_provider_and_legacy_vector_helpers_are_blocked() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_reads", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let key = insert(&pool, &fixture, &peer, "source").await;
        let config = HadesConfig::with_database(pool.database());
        let command: DaemonCommand = serde_json::from_value(json!({"command":"db.get",
            "params":{"collection":"documents", "key":key, "fields":["source_id","full_text"]}})).unwrap();
        let data = dispatch(&pool, &config, command).await.unwrap();
        assert!(data.to_string().contains("apple café"));
        for command in [
            json!({"command":"smell.report","params":{"path":"/absent/source"}}),
            json!({"command":"graph_embed.neighbors","params":{"node_id":"documents/source","limit":1}}),
            json!({"command":"db.create_index","params":{"collection":"embeddings","dimension":2,"metric":"cosine"}}),
        ] {
            let command: DaemonCommand = serde_json::from_value(command).unwrap();
            let error = format!("{:#}", anyhow::Error::new(dispatch(&pool, &config, command).await.unwrap_err()));
            assert!(error.contains("model-bound or initializing"), "{error}");
        }
    }).await;
}

#[tokio::test]
async fn alternating_and_concurrent_databases_resolve_independent_models() {
    let _serial = SERIAL.lock().await;
    with_temp_db("public_bound_first", Fixtures::Empty, |first| async move {
        with_temp_db(
            "public_bound_second",
            Fixtures::Empty,
            |second| async move {
                let a = Fixture::new(2);
                let b = Fixture::new(3);
                a.install(&first).await;
                b.install(&second).await;
                let peer_a = Peer::new(&a);
                let peer_b = Peer::new(&b);
                insert(&first, &a, &peer_a, "first/source").await;
                insert(&second, &b, &peer_b, "second/source").await;
                let mut config = config(&first, &peer_a);
                let routes = config.registered_providers.as_mut().unwrap();
                let other = peer_b.configuration();
                // Both synthetic registrations deliberately share the fixture artifact
                // manifest but expose different spaces/providers, as one installation.
                let name = format!("{}.json", b.profile.runtime_binding.provider_id);
                std::fs::copy(
                    other.registry_root.join("records").join(&name),
                    routes.registry_root.join("records").join(name),
                )
                .unwrap();
                routes.providers.extend(other.providers);
                let mut second_config = config.clone();
                second_config.database.name = Some(second.database().into());
                for _ in 0..2 {
                    let (one, two) =
                        tokio::join!(search(&first, &config), search(&second, &second_config));
                    assert_eq!(one["dimension"], 2);
                    assert_eq!(two["dimension"], 3);
                    assert_eq!(one["results"][0]["source_id"], "first/source");
                    assert_eq!(two["results"][0]["source_id"], "second/source");
                }
            },
        )
        .await;
    })
    .await;
}
