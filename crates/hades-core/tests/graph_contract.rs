//! Isolated database tests for actual graph-loader schema and feature provenance.
use hades_core::db::crud;
use hades_core::graph::{RuntimeSchema, load};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;

fn schema(relation: &str, collections: &[&str]) -> RuntimeSchema {
    RuntimeSchema {
        meta: serde_json::from_value(json!({"relation_order":[relation], "num_relations":1,
            "feature_dim":2, "model_type":"hetero_sage"}))
        .unwrap(),
        edge_definitions: vec![
            serde_json::from_value(json!({"name":relation,
            "from_collections":collections,"to_collections":collections}))
            .unwrap(),
        ],
        named_graphs: vec![],
        from_database: true,
    }
}

#[tokio::test]
async fn absent_collection_keeps_its_index_and_unknown_provenance_fails() {
    with_temp_db("graph_contract", Fixtures::Empty, |pool| async move {
        for collection in ["alpha", "middle", "zeta"] {
            crud::create_collection(&pool, collection, Some(2))
                .await
                .unwrap();
        }
        crud::create_collection(&pool, "links", Some(3))
            .await
            .unwrap();
        for collection in ["alpha", "zeta"] {
            crud::insert_documents(
                &pool,
                collection,
                &[json!({"_key":"node", "embedding":[1.,0.], "model":"fixture:v1"})],
                false,
            )
            .await
            .unwrap();
        }
        crud::insert_documents(
            &pool,
            "links",
            &[json!({"_key":"first", "_from":"alpha/node", "_to":"zeta/node"})],
            false,
        )
        .await
        .unwrap();
        let schema = schema("links", &["zeta", "middle", "alpha"]);
        let (before, ids) = load(&pool, &schema).await.unwrap();
        assert_eq!(before.collection_names, ["alpha", "middle", "zeta"]);
        assert_eq!(
            before.node_collections[ids.get_index("zeta/node").unwrap()],
            2
        );
        crud::insert_documents(&pool, "middle", &[json!({"_key":"node"})], false)
            .await
            .unwrap();
        crud::insert_documents(
            &pool,
            "links",
            &[json!({"_key":"second", "_from":"middle/node", "_to":"zeta/node"})],
            false,
        )
        .await
        .unwrap();
        let (after, ids) = load(&pool, &schema).await.unwrap();
        assert_eq!(after.contract, before.contract);
        assert_eq!(
            after.node_collections[ids.get_index("zeta/node").unwrap()],
            2
        );
        assert_eq!(
            after.node_collections[ids.get_index("middle/node").unwrap()],
            1
        );
        crud::insert_documents(
            &pool,
            "alpha",
            &[json!({"_key":"node", "embedding":[1.,0.]})],
            true,
        )
        .await
        .unwrap();
        assert!(
            load(&pool, &schema)
                .await
                .unwrap_err()
                .to_string()
                .contains("lacks a model identity")
        );
    })
    .await;
}

#[tokio::test]
async fn code_features_carry_pooling_provenance_and_reject_mixed_models() {
    with_temp_db("graph_features", Fixtures::Codebase, |pool| async move {
        crud::insert_documents(&pool, "codebase_files", &[json!({"_key":"file"})], false).await.unwrap();
        crud::insert_documents(&pool, "codebase_symbols", &[json!({"_key":"symbol", "file_key":"file"})], false).await.unwrap();
        crud::insert_documents(&pool, "codebase_defines_edges", &[json!({"_key":"defines", "_from":"codebase_files/file", "_to":"codebase_symbols/symbol"})], false).await.unwrap();
        crud::insert_documents(&pool, "codebase_embeddings", &[
            json!({"_key":"one", "file_key":"file", "embedding":[0.,2.], "model":"fixture:v1"}),
            json!({"_key":"two", "file_key":"file", "embedding":[2.,2.], "model":"fixture:v1"}),
        ], false).await.unwrap();
        let schema = schema("codebase_defines_edges", &["codebase_files", "codebase_symbols"]);
        let (graph, _) = load(&pool, &schema).await.unwrap();
        assert_eq!(graph.node_features, vec![1.,2.,1.,2.]);
        assert!(graph.contract.as_ref().unwrap().feature_models.values().all(|models| models.len() == 1));
        crud::insert_documents(&pool, "codebase_embeddings", &[
            json!({"_key":"two", "file_key":"file", "embedding":[2.,2.], "model":"different:v2"}),
        ], true).await.unwrap();
        assert!(load(&pool, &schema).await.unwrap_err().to_string().contains("mixes embedding model identities"));
    }).await;
}

#[tokio::test]
async fn recorded_smells_match_exact_files_across_roots_without_scanning() {
    use hades_core::config::HadesConfig;
    use hades_core::service::{ConnectionPolicy, handle_request};
    use std::time::Duration;
    with_temp_db("stored_smells", Fixtures::Empty, |pool| async move {
        for (name, kind) in [
            ("codebase_files", 2),
            ("smell_specs", 2),
            ("compliance_edges", 3),
        ] {
            crud::create_collection(&pool, name, Some(kind))
                .await
                .unwrap();
        }
        for key in ["root_a", "root_b"] {
            crud::insert_document(
                &pool,
                "codebase_files",
                &json!({
                    "_key":key,"path":"synthetic/not-on-disk.rs"
                }),
            )
            .await
            .unwrap();
        }
        crud::insert_document(
            &pool,
            "smell_specs",
            &json!({
                "_key":"one","name":"fixture smell"
            }),
        )
        .await
        .unwrap();
        for key in ["root_a", "root_b"] {
            crud::insert_document(
                &pool,
                "compliance_edges",
                &json!({
                    "_key":key,"_from":format!("codebase_files/{key}"),
                    "_to":"smell_specs/one","enforcement_type":"static"
                }),
            )
            .await
            .unwrap();
        }
        let config = HadesConfig::with_database("unused-config-db");
        for (path, count) in [
            ("synthetic/not-on-disk.rs", 2),
            ("codebase_files/root_a", 1),
            ("absent.rs", 0),
            ("' RETURN 1", 0),
        ] {
            let request = serde_json::to_vec(&json!({
                "command":"smell.stored_report","params":{"path":path}
            }))
            .unwrap();
            let response = handle_request(
                &pool,
                &config,
                ConnectionPolicy::agent_only(),
                &request,
                Duration::from_secs(5),
            )
            .await;
            assert!(response.success, "{path}: {response:?}");
            let data = response.data.unwrap();
            let rows = data["recorded_smells"].as_array().unwrap();
            assert_eq!(rows.len(), count);
            assert_eq!(data["truncated"], false);
            for row in rows {
                assert_eq!(row["name"], "fixture smell");
                assert_eq!(row["enforcement"], "static");
                assert_eq!(row["smell_id"], "smell_specs/one");
            }
            if count == 2 {
                assert_ne!(rows[0]["file_id"], rows[1]["file_id"]);
            }
        }
    })
    .await;
}
