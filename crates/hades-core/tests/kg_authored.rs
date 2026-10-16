//! Authored MCP graph writes against a disposable database.
use std::time::Duration;

use hades_core::db::crud;
use hades_core::service::{ConnectionPolicy, ProvisioningLimits, handle_request};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::{Value, json};

#[tokio::test]
async fn authored_graph_can_link_processed_documents_without_owning_them() {
    with_temp_db("kg_authored", Fixtures::Empty, |pool| async move {
        let request = |policy: ConnectionPolicy, command: Value| {
            let pool = pool.clone();
            async move {
                handle_request(
                    &pool,
                    &Default::default(),
                    policy,
                    &serde_json::to_vec(&command).unwrap(),
                    Duration::from_secs(10),
                )
                .await
            }
        };
        let allowed = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
            database_prefixes: vec!["hades_test_kg_authored_".into()],
            ingest_roots: vec![],
        });
        let create = json!({"command":"kg.create","params":{"name":"research"}});
        let denied = request(ConnectionPolicy::agent_only(), create.clone()).await;
        assert_eq!(denied.error_code.as_deref(), Some("ACCESS_DENIED"));
        let wrong_prefix = request(
            ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
                database_prefixes: vec!["other_".into()],
                ingest_roots: vec![],
            }),
            create.clone(),
        )
        .await;
        assert_eq!(wrong_prefix.error_code.as_deref(), Some("ACCESS_DENIED"));

        let created = request(allowed.clone(), create).await;
        assert!(created.success, "{created:?}");
        assert_eq!(created.data.unwrap()["nodes"], "research_nodes");
        let racing = json!({"command":"kg.create","params":{"name":"racing"}});
        let (first, second) = tokio::join!(
            request(allowed.clone(), racing.clone()),
            request(allowed.clone(), racing)
        );
        assert_ne!(first.success, second.success, "{first:?} {second:?}");
        let loser = if first.success { second } else { first };
        assert_eq!(loser.error_code.as_deref(), Some("CONFLICT"), "{loser:?}");
        crud::insert_document(
            &pool,
            "documents",
            &json!({"_key":"paper-1","title":"Original document"}),
        )
        .await
        .unwrap();

        let node = json!({"command":"kg.add_node","params":{
            "graph":"research","key":"claim-1","data":{"title":"Claim", "confidence":0.9}
        }});
        let added = request(allowed.clone(), node.clone()).await;
        assert!(added.success, "{added:?}");
        assert_eq!(added.data.unwrap()["_id"], "research_nodes/claim-1");
        let duplicate = request(allowed.clone(), node).await;
        assert!(!duplicate.success);
        let reserved = request(
            allowed.clone(),
            json!({"command":"kg.add_node","params":{
                "graph":"research","key":"bad", "data":{"_from":"documents/paper-1"}
            }}),
        )
        .await;
        assert!(!reserved.success);

        let missing = request(
            allowed.clone(),
            json!({"command":"kg.add_edge","params":{
                "graph":"research","key":"supports-missing", "from":"research_nodes/claim-1",
                "to":"documents/absent", "data":{"relation":"supported_by"}
            }}),
        )
        .await;
        assert!(!missing.success);
        let outside = request(
            allowed.clone(),
            json!({"command":"kg.add_edge","params":{
                "graph":"research","key":"outside", "from":"research_nodes/claim-1",
                "to":"hades_schema/meta", "data":{}
            }}),
        )
        .await;
        assert!(!outside.success);
        let edge = request(
            allowed.clone(),
            json!({"command":"kg.add_edge","params":{
                "graph":"research","key":"supports-1", "from":"research_nodes/claim-1",
                "to":"documents/paper-1", "data":{"relation":"supported_by"}
            }}),
        )
        .await;
        assert!(edge.success, "{edge:?}");
        assert_eq!(edge.data.unwrap()["_id"], "research_edges/supports-1");

        let neighbors = request(
            ConnectionPolicy::agent_only(),
            json!({
                "command":"db.graph.neighbors","params":{
                    "vertex":"research_nodes/claim-1", "direction":"outbound",
                    "limit":10,"graph":"research"
                }
            }),
        )
        .await;
        assert!(neighbors.success, "{neighbors:?}");
        assert!(
            serde_json::to_string(&neighbors.data)
                .unwrap()
                .contains("documents/paper-1")
        );

        // A document refresh owns only the document row. Authored graph rows
        // stay in their own collections and retain the document link by ID.
        crud::replace_document(
            &pool,
            "documents",
            "paper-1",
            &json!({"title":"Updated document"}),
        )
        .await
        .unwrap();
        assert_eq!(
            crud::get_document(&pool, "research_nodes", "claim-1")
                .await
                .unwrap()["title"],
            "Claim"
        );
        assert_eq!(
            crud::get_document(&pool, "research_edges", "supports-1")
                .await
                .unwrap()["_to"],
            "documents/paper-1"
        );
    })
    .await;
}
