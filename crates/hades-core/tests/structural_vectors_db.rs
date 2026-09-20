//! Real AQL validation, admitted only by the disposable-server test harness.
use hades_core::config::HadesConfig;
use hades_core::db::{ArangoPool, crud};
use hades_core::service::{ConnectionPolicy, handle_request};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;
use std::time::Duration;

async fn request(
    pool: &ArangoPool,
    command: &str,
    node: &str,
) -> hades_core::dispatch::DaemonResponse {
    handle_request(
        pool,
        &HadesConfig::with_database("fixture"),
        ConnectionPolicy::agent_only(),
        &serde_json::to_vec(&json!({"command":command,"params":{"node_id":node}})).unwrap(),
        Duration::from_secs(5),
    )
    .await
}

#[tokio::test]
async fn malformed_candidates_are_excluded_and_invalid_targets_fail() {
    with_temp_db("structural_vectors", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "nodes", Some(2))
            .await
            .unwrap();
        crud::create_collection(&pool, "other", Some(2))
            .await
            .unwrap();
        crud::insert_documents(
            &pool,
            "nodes",
            &[
                json!({"_key":"target","structural_embedding":[1,0]}),
                json!({"_key":"valid","structural_embedding":[0.5,0.5]}),
                json!({"_key":"null_component","structural_embedding":[null,1]}),
                json!({"_key":"string_component","structural_embedding":["3",1]}),
                json!({"_key":"boolean_component","structural_embedding":[true,1]}),
                json!({"_key":"large_component","structural_embedding":[1e100,1]}),
                json!({"_key":"negative_large","structural_embedding":[-1e100,1]}),
                json!({"_key":"wrong_dimension","structural_embedding":[1]}),
                json!({"_key":"non_array","structural_embedding":{"a":1,"b":2}}),
                json!({"_key":"empty","structural_embedding":[]}),
                json!({"_key":"missing"}),
            ],
            false,
        )
        .await
        .unwrap();
        crud::insert_documents(
            &pool,
            "other",
            &[json!({"_key":"best","structural_embedding":[0.75,0.25]})],
            false,
        )
        .await
        .unwrap();
        let response = request(&pool, "graph_embed.neighbors", "nodes/target").await;
        assert!(response.success, "{response:?}");
        let data = response.data.unwrap();
        let neighbors = data["neighbors"].as_array().unwrap();
        assert_eq!(neighbors.len(), 2, "{data}");
        assert_eq!(neighbors[0]["id"], "other/best");
        assert_eq!(neighbors[0]["similarity"].as_f64(), Some(0.75));
        assert_eq!(neighbors[1]["id"], "nodes/valid");
        assert_eq!(neighbors[1]["similarity"].as_f64(), Some(0.5));
        for command in ["graph_embed.embed", "graph_embed.neighbors"] {
            for key in [
                "null_component",
                "string_component",
                "boolean_component",
                "large_component",
                "negative_large",
                "non_array",
                "empty",
                "missing",
            ] {
                let response = request(&pool, command, &format!("nodes/{key}")).await;
                assert!(!response.success, "{command} accepted {key}: {response:?}");
                assert_eq!(response.error_code.as_deref(), Some("QUERY_FAILED"));
            }
            let response = request(&pool, command, "nodes/target").await;
            assert!(response.success, "{response:?}");
        }
    })
    .await;
}
