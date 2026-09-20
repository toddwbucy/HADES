//! Opt-in real DB -> Rust safetensors -> CPU Python RPC -> Rust export contract.
use hades_core::db::crud;
use hades_core::graph::{self, ExportConfig, RuntimeSchema};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::{Value, json};

#[tokio::test]
#[ignore = "requires the private database runner and an explicit CPU Python interpreter"]
async fn training_rows_remain_attached_to_qualified_node_ids() {
    let python =
        std::env::var("HADES_ALIGNMENT_PYTHON").expect("explicit CPU interpreter required");
    with_temp_db(
        "training_alignment",
        Fixtures::Empty,
        move |pool| async move {
            for name in ["papers", "concepts", "hades_schema"] {
                crud::create_collection(&pool, name, Some(2)).await.unwrap();
            }
            crud::create_collection(&pool, "links", Some(3))
                .await
                .unwrap();
            crud::insert_documents(
                &pool,
                "papers",
                &[
                    json!({"_key":"same", "embedding":[3.,1.], "model":"fixture:v1"}),
                    json!({"_key":"missing"}),
                ],
                false,
            )
            .await
            .unwrap();
            crud::insert_documents(
                &pool,
                "concepts",
                &[json!({"_key":"same", "embedding":[1.,7.], "model":"fixture:v1"})],
                false,
            )
            .await
            .unwrap();
            crud::insert_documents(
                &pool,
                "links",
                &[
                    json!({"_key":"a", "_from":"papers/same", "_to":"concepts/same"}),
                    json!({"_key":"b", "_from":"papers/missing", "_to":"concepts/same"}),
                ],
                false,
            )
            .await
            .unwrap();
            crud::insert_documents(
                &pool,
                "hades_schema",
                &[
                    json!({"_key":"meta", "schema_type":"schema_meta", "relation_order":["links"],
                "num_relations":1, "feature_dim":2, "model_type":"hetero_sage"}),
                    json!({"_key":"links", "schema_type":"edge_definition", "name":"links",
                "from_collections":["papers"], "to_collections":["concepts"]}),
                ],
                false,
            )
            .await
            .unwrap();
            let schema = RuntimeSchema::load(&pool).await.unwrap();
            let (graph, ids) = graph::load(&pool, &schema).await.unwrap();
            assert_eq!((graph.num_nodes, graph.num_edges), (3, 2));
            let root = tempfile::tempdir().unwrap();
            hades_prefetch::serialize_to_file(
                &root.path().join("graph.safetensors"),
                &graph,
                &hades_prefetch::EdgeSplit {
                    train_idx: vec![0, 1],
                    val_idx: vec![],
                    test_idx: vec![],
                },
                &hades_prefetch::NegativeSamples {
                    src: vec![0, 2],
                    dst: vec![0, 2],
                },
                &hades_prefetch::SplitConfig {
                    val_ratio: 0.,
                    test_ratio: 0.,
                    ..Default::default()
                },
            )
            .unwrap();
            let names: Vec<_> = (0..3).map(|i| ids.get_arango_id(i).unwrap()).collect();
            let features: Vec<[f32; 2]> = names
                .iter()
                .map(|id| match *id {
                    "papers/same" => [3., 1.],
                    "concepts/same" => [1., 7.],
                    "papers/missing" => [0., 0.],
                    other => panic!("unexpected node {other}"),
                })
                .collect();
            let collections: Vec<_> = names
                .iter()
                .map(|id| u32::from(id.starts_with("papers/")))
                .collect();
            let edge_ids: Vec<_> = graph
                .edge_src
                .iter()
                .zip(&graph.edge_dst)
                .map(|(&s, &d)| {
                    let source = ids.get_arango_id(s as usize).unwrap();
                    let target = ids.get_arango_id(d as usize).unwrap();
                    assert!(matches!(source, "papers/same" | "papers/missing"));
                    assert_eq!(target, "concepts/same");
                    [s, d]
                })
                .collect();
            let mut named_edges: Vec<_> = edge_ids
                .iter()
                .map(|[s, d]| {
                    (
                        ids.get_arango_id(*s as usize).unwrap(),
                        ids.get_arango_id(*d as usize).unwrap(),
                    )
                })
                .collect();
            named_edges.sort();
            assert_eq!(
                named_edges,
                vec![
                    ("papers/missing", "concepts/same"),
                    ("papers/same", "concepts/same")
                ]
            );
            std::fs::write(
                root.path().join("manifest.json"),
                serde_json::to_vec(&json!({
                    "ids":names, "features":features, "collections":collections, "edges":edge_ids,
                }))
                .unwrap(),
            )
            .unwrap();
            let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../services/tests/training_alignment_peer.py");
            let output = std::process::Command::new("timeout")
                .args(["--kill-after=5", "90", &python])
                .arg(peer)
                .arg(root.path())
                .env("CUDA_VISIBLE_DEVICES", "")
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "CPU peer failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let expected: Value =
                serde_json::from_slice(&std::fs::read(root.path().join("expected.json")).unwrap())
                    .unwrap();
            let full =
                graph::decode_f32_embeddings(&std::fs::read(root.path().join("full.bin")).unwrap())
                    .unwrap();
            graph::export_embeddings(&pool, &ids, &full, 4, &ExportConfig { chunk_size: 1 })
                .await
                .unwrap();
            for id in &names {
                let (collection, key) = id.split_once('/').unwrap();
                let row = crud::get_document(&pool, collection, key).await.unwrap();
                let actual: Vec<f32> =
                    serde_json::from_value(row["structural_embedding"].clone()).unwrap();
                let wanted: Vec<f32> =
                    serde_json::from_value(expected["before"][*id].clone()).unwrap();
                assert_eq!(actual, wanted, "full export identity {id}");
            }
            let subset = graph::decode_f32_embeddings(
                &std::fs::read(root.path().join("subset.bin")).unwrap(),
            )
            .unwrap();
            graph::export_embeddings_subset(
                &pool,
                &ids,
                &[2, 0],
                &subset,
                4,
                &ExportConfig { chunk_size: 1 },
            )
            .await
            .unwrap();
            for (index, id) in names.iter().enumerate() {
                let (collection, key) = id.split_once('/').unwrap();
                let row = crud::get_document(&pool, collection, key).await.unwrap();
                let generation = if index == 1 { "before" } else { "after" };
                let actual: Vec<f32> =
                    serde_json::from_value(row["structural_embedding"].clone()).unwrap();
                let wanted: Vec<f32> =
                    serde_json::from_value(expected[generation][*id].clone()).unwrap();
                assert_eq!(actual, wanted, "subset export identity {id}");
            }
        },
    )
    .await;
}
