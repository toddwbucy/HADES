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
        for (key, path) in [
            ("collision", "codebase_files/root_a"),
            ("fallback", "codebase_files/not-an-existing-id"),
        ] {
            crud::insert_document(&pool, "codebase_files", &json!({"_key":key,"path":path}))
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
        for key in ["root_a", "root_b", "collision", "fallback"] {
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
            ("codebase_files/not-an-existing-id", 1),
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
            if path == "codebase_files/root_a" {
                assert_eq!(rows[0]["file_id"], "codebase_files/root_a");
            }
            if path == "codebase_files/not-an-existing-id" {
                assert_eq!(rows[0]["file_id"], "codebase_files/fallback");
            }
        }
    })
    .await;
}

#[tokio::test]
async fn vertex_read_class_projects_bulk_fields_and_explicit_opt_ins() {
    use hades_core::dispatch::{AccessTier, DaemonCommand};
    use hades_core::service::{ConnectionPolicy, handle_request};
    use serde_json::Value;
    with_temp_db("vertex_projection", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool,"documents",Some(2)).await.unwrap();
        crud::create_collection(&pool,"links",Some(3)).await.unwrap();
        let documents=vec![
            json!({"_key":"plain","label":"ordinary vertex","path":"plain"}),
            json!({"_key":"paper","label":"paper vertex","full_text":"entire paper","body":"paper body"}),
            json!({"_key":"code","label":"code vertex","text":"fn example() {}","embedding":[1,0]}),
        ];
        crud::insert_documents(&pool,"documents",&documents,false).await.unwrap();
        crud::insert_documents(&pool,"links",&[
            json!({"_key":"paper","_from":"documents/plain","_to":"documents/paper","label":"edge retained"}),
            json!({"_key":"code","_from":"documents/plain","_to":"documents/code","label":"edge retained"}),
        ],false).await.unwrap();
        pool.writer().post("gharial",&json!({"name":"fixture","edgeDefinitions":[{"collection":"links","from":["documents"],"to":["documents"]}]})).await.unwrap();
        for fields in [None,Some(json!(["_key","label","full_text","embedding","text","body"])),Some(json!([]))] {
            let mut cases=vec![
                ("db.graph.traverse",json!({"start":"documents/plain","graph":"fixture","max_depth":2})),
                ("db.graph.neighbors",json!({"vertex":"documents/plain","graph":"fixture"})),
                ("db.recent",json!({})),
                ("db.list",json!({})),
            ];
            for key in ["plain","paper","code"] {
                cases.push(("db.get",json!({"collection":"documents","key":key})));
                cases.push(("db.graph.shortest_path",json!({"source":"documents/plain","target":format!("documents/{key}"),"graph":"fixture"})));
            }
            for (operation, mut params) in cases {
                if let Some(fields)=&fields {params["fields"]=fields.clone();}
                let payload=json!({"command":operation,"params":params});
                let command:DaemonCommand=serde_json::from_value(payload.clone()).unwrap();
                assert_eq!(command.access_tier(),AccessTier::Agent);
                let response=handle_request(&pool,&Default::default(),ConnectionPolicy::agent_only(),&serde_json::to_vec(&payload).unwrap(),std::time::Duration::from_secs(5)).await;
                assert!(response.success,"{operation}: {response:?}");
                let data=response.data.unwrap();
                let vertices:Vec<&Value>=if operation=="db.get" {vec![&data]}
                else if operation.starts_with("db.graph.") {
                    data["results"].as_array().unwrap().iter().map(|row| {
                        if !row["edge"].is_null() {assert_eq!(row["edge"]["label"],"edge retained");}
                        &row["vertex"]
                    }).collect()
                } else {data["documents"].as_array().unwrap().iter().collect()};
                assert!(!vertices.is_empty(),"{operation}");
                for vertex in vertices {
                    if fields.as_ref()==Some(&json!([])) {
                        assert!(vertex.as_object().unwrap().keys().all(|key|key=="_collection"));
                        continue;
                    }
                    let original=documents.iter().find(|d|d["_key"]==vertex["_key"]).unwrap();
                    assert_eq!(vertex["label"],original["label"]);
                    for field in ["full_text","embedding","text","body"] {
                        if fields.is_none() { assert!(vertex.get(field).is_none(),"{operation}: {vertex}"); }
                        else { assert_eq!(vertex.get(field),original.get(field),"{operation}: {field}"); }
                    }
                }
            }
        }
    }).await;
}

#[test]
fn weavertools_edge_endpoints_follow_document_format() {
    // Document Format v0.24 sections 3–4 at WeaverTools 13dd7db (#175).
    // Read the actual YAML, not a parallel runtime table or a text-pattern proxy.
    let schema: serde_json::Value = serde_yaml::from_str(include_str!(
        "../../../services/adapters/weavertools/schema.yaml"
    ))
    .unwrap();
    let expected = [
        ("cites", vec!["codebase_files"], vec!["wt_assertions"]),
        (
            "declared_in",
            vec![
                "wt_artifacts",
                "wt_assertions",
                "wt_axioms",
                "wt_crates",
                "wt_documents",
                "wt_systems",
                "wt_terms",
                "wt_vocabulary",
            ],
            vec!["documents"],
        ),
        ("asserts", vec!["wt_crates"], vec!["wt_assertions"]),
        (
            "defines",
            vec!["wt_crates", "wt_documents"],
            vec!["wt_vocabulary", "wt_terms"],
        ),
        (
            "draws",
            vec!["wt_documents"],
            vec!["wt_vocabulary", "wt_terms"],
        ),
        ("elects", vec!["wt_vocabulary"], vec!["wt_vocabulary"]),
        ("floor_link", vec!["wt_crates"], vec!["wt_crates"]),
        ("grounds", vec!["wt_assertions"], vec!["wt_axioms"]),
        ("holds", vec!["wt_artifacts"], vec!["wt_vocabulary"]),
        ("parent", vec!["wt_crates"], vec!["wt_crates", "wt_systems"]),
        ("party", vec!["wt_documents"], vec!["wt_crates"]),
        ("reads", vec!["wt_crates"], vec!["wt_artifacts"]),
        ("seam", vec!["wt_crates"], vec!["wt_crates"]),
        ("writes", vec!["wt_crates"], vec!["wt_artifacts"]),
    ];
    let definitions = schema["edge_definitions"].as_array().unwrap();
    assert_eq!(
        definitions
            .iter()
            .filter(|row| row["name"].as_str().unwrap().starts_with("wt_"))
            .count(),
        expected.len()
    );
    for (relation, from, to) in expected {
        let name = format!("wt_{relation}_edges");
        let row = definitions.iter().find(|row| row["name"] == name).unwrap();
        assert_eq!(row["from_collections"], json!(from), "{name}");
        assert_eq!(row["to_collections"], json!(to), "{name}");
    }
}

#[tokio::test]
async fn indexed_lookup_is_agent_bounded_projected_and_fails_closed() {
    use hades_core::dispatch::{AccessTier, DaemonCommand};
    use hades_core::service::{ConnectionPolicy, handle_request};
    with_temp_db("indexed_lookup", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "nodes", Some(2)).await.unwrap();
        pool.writer().post("index?collection=nodes", &json!({"type":"persistent","fields":["ident","other"],"name":"by_ident"})).await.unwrap();
        pool.writer().post("index?collection=nodes", &json!({"type":"persistent","fields":["first","second"],"name":"by_first"})).await.unwrap();
        pool.writer().post("index?collection=nodes", &json!({"type":"persistent","fields":["nested.ident"],"name":"by_nested"})).await.unwrap();
        pool.writer().post("index?collection=nodes", &json!({"type":"persistent","fields":["sparse"],"sparse":true,"name":"by_sparse"})).await.unwrap();
        let docs: Vec<_> = (0..1002).map(|n| json!({"_key":format!("node{n}"),"ident":"same","other":n,"nested":{"ident":true},"full_text":"bulk","text":"bulk","body":"bulk","embedding":[1,0]})).collect();
        crud::insert_documents(&pool,"nodes",&docs,false).await.unwrap();
        let call = |params| {
            let pool=pool.clone();
            async move {
                let request=json!({"command":"db.lookup","params":params});
                let cmd:DaemonCommand=serde_json::from_value(request.clone()).unwrap();
                assert_eq!(cmd.access_tier(),AccessTier::Agent);
                handle_request(&pool,&Default::default(),ConnectionPolicy::agent_only(),&serde_json::to_vec(&request).unwrap(),std::time::Duration::from_secs(5)).await
            }
        };
        for (limit,count) in [(None,10),(Some(2),2),(Some(5000),1000)] {
            let response=call(json!({"collection":"nodes","field":"ident","value":"same","limit":limit})).await;
            assert!(response.success,"{response:?}");
            let data=response.data.unwrap();
            assert_eq!(data["count"],count);
            for row in data["documents"].as_array().unwrap() {
                for field in ["full_text","text","body","embedding"] {assert!(row.get(field).is_none());}
            }
        }
        let projected=call(json!({"collection":"nodes","field":"ident","value":"same","limit":1,"fields":["ident","full_text","embedding"]})).await;
        assert!(projected.success,"{projected:?}");
        let row=&projected.data.unwrap()["documents"][0];
        assert_eq!(row.as_object().unwrap().len(),3);
        assert_eq!(row["full_text"],"bulk");
        assert_eq!(row["embedding"],json!([1,0]));
        let nested=call(json!({"collection":"nodes","field":"nested.ident","value":true})).await;
        assert!(nested.success,"{nested:?}");
        assert_eq!(nested.data.unwrap()["count"],10);
        for field in ["unindexed","second","_key"] {
            let response=call(json!({"collection":"nodes","field":field,"value":"same"})).await;
            assert!(!response.success,"{response:?}");
            let error=serde_json::to_string(&response).unwrap();
            assert!(error.contains("indexed fields available") && error.contains("ident") && error.contains("first"),"{error}");
        }
        let missing=call(json!({"collection":"absent_nodes","field":"ident","value":"x"})).await;
        assert!(!missing.success);
        assert!(serde_json::to_string(&missing).unwrap().contains("absent_nodes"));
        // A sparse index cannot answer equality to null. Forcing it must fail,
        // rather than falling back to scanning the collection (#173).
        let sparse=call(json!({"collection":"nodes","field":"sparse","value":null})).await;
        assert!(!sparse.success,"{sparse:?}");
        let zero=call(json!({"collection":"nodes","field":"ident","value":"same","limit":0})).await;
        assert!(!zero.success);
    }).await;
}
