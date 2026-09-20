//! Synthetic-only authorization fixtures; never read real server files.
use hades_core::config::HadesConfig;
use hades_core::db::{ArangoClient, ArangoPool};
use hades_core::service::{ConnectionPolicy, handle_request};
use serde_json::json;
use std::time::Duration;
#[allow(dead_code)]
#[path = "common/cursor_mock.rs"]
mod cursor_mock;

#[tokio::test]
async fn agent_cannot_scan_files_or_symlink_descendants() {
    let files = tempfile::tempdir().unwrap();
    let secret = files.path().join("private.txt");
    std::fs::write(&secret, "password = synthetic-audit-secret").unwrap();
    let tree = files.path().join("tree");
    std::fs::create_dir(&tree).unwrap();
    std::os::unix::fs::symlink(&secret, tree.join("linked.rs")).unwrap();
    let config = HadesConfig::with_database("fixture");
    let mut mock = cursor_mock::Mock::new(vec![]).await;
    for command in ["smell.check", "smell.verify", "smell.report"] {
        for path in [&secret, &tree] {
            let request = serde_json::to_vec(&json!({
                "command":command, "params":{"path":path.to_str().unwrap()}
            }))
            .unwrap();
            let response = handle_request(
                &mock.pool,
                &config,
                ConnectionPolicy::agent_only(),
                &request,
                Duration::from_secs(2),
            )
            .await;
            assert!(!response.success, "{command}: {response:?}");
            assert_eq!(response.error_code.as_deref(), Some("ACCESS_DENIED"));
        }
    }
    assert!(
        mock.events.try_recv().is_err(),
        "denied scans reached the database"
    );
}

#[tokio::test]
async fn stored_report_uses_bound_database_identity_without_reading_a_file() {
    // The database can report a file that does not exist on this server.
    let files = tempfile::tempdir().unwrap();
    let path = files.path().join("does-not-exist' RETURN 1");
    let config = HadesConfig::with_database("fixture");
    let row = json!({"file_id":"codebase_files/one", "smell_id":"smell_specs/one"});
    let mut mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(json!({
        "result":[row], "hasMore":false
    }))])
    .await;
    let request = serde_json::to_vec(&json!({
        "command":"smell.stored_report", "params":{"path":path.to_str().unwrap()}
    }))
    .unwrap();
    let response = handle_request(
        &mock.pool,
        &config,
        ConnectionPolicy::agent_only(),
        &request,
        Duration::from_secs(2),
    )
    .await;
    assert!(response.success, "{response:?}");
    let data = response.data.unwrap();
    assert_eq!(data["recorded_smells"], json!([row]));
    assert_eq!(data["source"], "stored_graph");
    assert_eq!(data["truncated"], false);
    let query = mock.event("POST cursor").await;
    assert_eq!(query["bindVars"]["path"], path.to_str().unwrap());
    assert!(
        !query["query"]
            .as_str()
            .unwrap()
            .contains(path.to_str().unwrap())
    );
    assert_eq!(query["memoryLimit"], 32 * 1024 * 1024);
    assert!(!path.exists());
}

#[tokio::test]
async fn stored_report_marks_truncated_results() {
    let config = HadesConfig::with_database("fixture");
    let mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(json!({
        "result": (0..101).map(|n| json!({"edge_id":n})).collect::<Vec<_>>(),
        "hasMore":false
    }))])
    .await;
    let request = serde_json::to_vec(&json!({
        "command":"smell.stored_report", "params":{"path":"src/lib.rs"}
    }))
    .unwrap();
    let response = handle_request(
        &mock.pool,
        &config,
        ConnectionPolicy::agent_only(),
        &request,
        Duration::from_secs(2),
    )
    .await;
    assert!(response.success, "{response:?}");
    let data = response.data.unwrap();
    assert_eq!(data["recorded_smells"].as_array().unwrap().len(), 100);
    assert_eq!(data["truncated"], true);
}

#[tokio::test]
async fn trusted_admin_can_still_scan_synthetic_file() {
    let files = tempfile::tempdir().unwrap();
    let path = files.path().join("fixture.rs");
    let marker = "password = synthetic-audit-secret";
    std::fs::write(&path, marker).unwrap();
    let config = HadesConfig::with_database("fixture");
    let mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(json!({
        "result":[{"_key":"fixture","smell_id":1,"name":"fixture", "tier":"static", "forbidden_patterns":["password"]}],
        "hasMore":false
    }))]).await;
    let request = serde_json::to_vec(&json!({
        "command":"smell.check", "params":{"path":path.to_str().unwrap(),"verbose":true}
    }))
    .unwrap();
    let response = handle_request(
        &mock.pool,
        &config,
        ConnectionPolicy::local_admin(),
        &request,
        Duration::from_secs(2),
    )
    .await;
    assert!(response.success, "{response:?}");
    assert!(response.data.unwrap().to_string().contains(marker));
}
