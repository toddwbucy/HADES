//! Portable job pins are validated against fresh disposable database snapshots.
use hades_core::{
    config::{HadesConfig, snapshot::Snapshot},
    embedding_profile::store::{COLLECTION, ProfilePin, ProfileStore},
    ingest_jobs::Binding,
    pipeline::profiled_documents::ProfiledDocuments,
    test_support::{Fixtures, with_temp_db},
};
use serde_json::json;

#[allow(dead_code)]
#[path = "common/profiled_provider.rs"]
mod provider;
use provider::{Fixture, Peer};

/// The manifest stays in the database; compact sealed pins survive only explicit handoff.
#[tokio::test]
async fn pins_are_compact_strict_and_excluded_from_ordinary_configuration() {
    with_temp_db("profiled_job_wire", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let pin = documents.job_pin().unwrap();
        let value = json!(pin);
        assert_eq!(value["profile_revision"], 2);
        assert_eq!(value["manifest_md5"], json!(fixture.profile.model.manifest_md5));
        assert!(value.get("manifest").is_none());
        assert!(value.to_string().len() < 4096);
        assert!(!value.to_string().contains(pool.writer().socket_path().unwrap().to_str().unwrap()));
        let decoded: ProfilePin = serde_json::from_value(value.clone()).unwrap();
        ProfiledDocuments::connect_pinned(pool.clone(), &peer.catalog(), &decoded).await.unwrap();
        let mut extra = value.clone();
        extra["untrusted"] = json!(true);
        assert!(serde_json::from_value::<ProfilePin>(extra).is_err());
        for field in ["format", "database_id", "snapshot_sha256", "endpoint_sha256"] {
            let mut bad = value.clone();
            bad[field] = json!("invalid");
            assert!(serde_json::from_value::<ProfilePin>(bad).unwrap().validate().is_err());
        }
        let mut config = HadesConfig::with_database(pool.database());
        config.ingest_binding = Some(Binding::Profiled(Box::new(pin.clone())));
        let encoded = serde_json::to_value(&config).unwrap();
        assert!(encoded.get("ingest_binding").is_none());
        let mut injected = encoded;
        injected["ingest_binding"] = value;
        assert!(serde_json::from_value::<HadesConfig>(injected).unwrap().ingest_binding.is_none());
        let snapshot = Snapshot::new(&config).unwrap();
        let mut command = tokio::process::Command::new("/bin/true");
        let fd = snapshot.inherit(&mut command);
        let inherited = hades_core::config::snapshot::load_inherited(fd).unwrap();
        assert!(matches!(inherited.ingest_binding, Some(Binding::Profiled(ref actual)) if **actual == pin));
    }).await;
}

/// Neither identical model bytes in another database nor a restored record rev is admission.
#[tokio::test]
async fn pins_refuse_other_databases_and_changed_records_before_provider_observation() {
    with_temp_db("profiled_job_scope", Fixtures::Empty, |one| async move {
        let fixture = Fixture::new(2);
        fixture.install(&one).await;
        let peer = Peer::new(&fixture);
        let pin = fixture.connect(&one, &peer).await.job_pin().unwrap();
        with_temp_db("profiled_job_other", Fixtures::Empty, |two| async move {
            fixture.install(&two).await;
            let observations = peer.observations();
            assert!(
                ProfiledDocuments::connect_pinned(two, &peer.catalog(), &pin)
                    .await
                    .is_err()
            );
            assert_eq!(peer.observations(), observations);
            let before = ProfileStore::new(one.clone())
                .unwrap()
                .load()
                .await
                .unwrap();
            let path = format!("document/{COLLECTION}/embedding_profile");
            let original = one.writer().get(&path).await.unwrap();
            one.writer()
                .patch(&path, &json!({"payload":original["payload"]}))
                .await
                .unwrap();
            let after = ProfileStore::new(one.clone())
                .unwrap()
                .load()
                .await
                .unwrap();
            assert_eq!(before.profile(), after.profile());
            assert_ne!(before, after);
            let error = ProfiledDocuments::connect_pinned(one, &peer.catalog(), &pin)
                .await
                .err()
                .unwrap();
            assert!(error.to_string().contains("ingestion job profile changed"));
            assert_eq!(peer.observations(), observations);
            assert!(peer.requests().is_empty());
        })
        .await;
    })
    .await;
}
