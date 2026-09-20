//! Stored structural-vector validation through the actual service envelope.
use hades_core::config::HadesConfig;
use hades_core::db::{ArangoClient, ArangoPool};
use hades_core::service::{ConnectionPolicy, handle_request};
use serde_json::{Value, json};
use std::time::Duration;
#[allow(dead_code)]
#[path = "common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};

async fn response(command: &str, vector: Value) -> hades_core::dispatch::DaemonResponse {
    let target = if command == "graph_embed.embed" {
        json!({"_id":"nodes/a","structural_embedding":vector})
    } else {
        vector
    };
    let mock = Mock::new(vec![
        Reply::page(json!({"result":[target],"hasMore":false})),
        Reply::page(json!({"result":[]})),
    ])
    .await;
    handle_request(
        &mock.pool,
        &HadesConfig::with_database("fixture"),
        ConnectionPolicy::agent_only(),
        &serde_json::to_vec(&json!({"command":command,"params":{"node_id":"nodes/a"}})).unwrap(),
        Duration::from_secs(3),
    )
    .await
}

#[tokio::test]
async fn lookup_rejects_invalid_numeric_components() {
    for vector in [
        json!([null]),
        json!(["bad"]),
        json!([true]),
        json!([1e100]),
        json!([]),
        json!("bad"),
        json!(null),
    ] {
        let result = response("graph_embed.embed", vector).await;
        assert!(!result.success, "malformed vector accepted: {result:?}");
        assert_eq!(result.error_code.as_deref(), Some("QUERY_FAILED"));
    }
}

#[tokio::test]
async fn neighbors_rejects_invalid_numeric_target() {
    for vector in [
        json!([null]),
        json!(["bad"]),
        json!([true]),
        json!([1e100]),
        json!([]),
        json!("bad"),
        json!(null),
    ] {
        let result = response("graph_embed.neighbors", vector).await;
        assert!(!result.success, "malformed target accepted: {result:?}");
        assert_eq!(result.error_code.as_deref(), Some("QUERY_FAILED"));
    }
}

#[tokio::test]
async fn valid_stored_vectors_remain_readable() {
    for command in ["graph_embed.embed", "graph_embed.neighbors"] {
        let result = response(command, json!([0.25, -0.5])).await;
        assert!(result.success, "valid vector rejected: {result:?}");
    }
}
