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
            // Preserve the complete node order while varying one input at a time.
            let mut changed_features = graph.clone();
            changed_features.set_node_features(ids.get_index("papers/same").unwrap(), &[9., -3.]);
            hades_prefetch::serialize_graph_for_inference_to_file(
                &root.path().join("features.safetensors"),
                &changed_features,
            )
            .unwrap();
            let mut changed_neighbors = graph.clone();
            changed_neighbors.edge_src.clear();
            changed_neighbors.edge_dst.clear();
            changed_neighbors.edge_type.clear();
            changed_neighbors.num_edges = 0;
            hades_prefetch::serialize_graph_for_inference_to_file(
                &root.path().join("neighbors.safetensors"),
                &changed_neighbors,
            )
            .unwrap();
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
                    "database":pool.database(), "database_socket":std::env::var("ARANGO_SOCKET").unwrap(),
                    "cli_binary":env!("CARGO_BIN_EXE_hades"),
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
            println!(
                "CPU generation evidence: {}",
                String::from_utf8_lossy(&output.stdout)
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
            // Reset to one generation, then fail a later batch after one real
            // new-checkpoint vector is acknowledged. Same collection makes
            // the batch ordering explicit rather than HashMap-dependent.
            graph::export_embeddings(&pool, &ids, &full, 4, &ExportConfig { chunk_size: 1 })
                .await.unwrap();
            let mut interrupted_ids = graph::IDMap::new();
            interrupted_ids.get_or_create("papers/same");
            interrupted_ids.get_or_create("papers/absent");
            let new_vector: Vec<f32> = serde_json::from_value(expected["after"]["papers/same"].clone()).unwrap();
            let attempted = [new_vector.clone(), new_vector].concat();
            let failure = graph::export_embeddings(&pool, &interrupted_ids, &attempted, 4,
                &ExportConfig { chunk_size: 1 }).await.unwrap_err();
            assert!(matches!(failure, graph::ExportError::BatchFailed { acknowledged: 1, .. }));
            for id in &names {
                let (collection, key) = id.split_once('/').unwrap();
                let row = crud::get_document(&pool, collection, key).await.unwrap();
                let generation = if *id == "papers/same" { "after" } else { "before" };
                let actual: Vec<f32> = serde_json::from_value(row["structural_embedding"].clone()).unwrap();
                let wanted: Vec<f32> = serde_json::from_value(expected[generation][*id].clone()).unwrap();
                assert_eq!(actual, wanted, "failed refresh generation {id}");
            }
            println!("Failed refresh: one acknowledged new-checkpoint vector; two old-checkpoint vectors retained; explicit export error.");
            // The actual CLI must be unable to contact the fixed live training
            // socket even if the expected no-op branch regresses. Bubblewrap
            // hides /run and devices, while the private DB socket stays visible.
            crud::update_document(&pool, "papers", "same", &json!({"embedding":[9., -3.]}))
                .await.unwrap();
            let mut before_noop = Vec::new();
            for id in &names {
                let (collection, key) = id.split_once('/').unwrap();
                before_noop.push(crud::get_document(&pool, collection, key).await.unwrap());
            }
            let absent_checkpoint = root.path().join("absent-checkpoint");
            let cli = |missing_only: bool| {
                let mut command = std::process::Command::new("timeout");
                command.args(["--kill-after=5", "30", "bwrap", "--ro-bind", "/", "/",
                    "--tmpfs", "/run", "--dev", "/dev", "--unshare-net", "--",
                    env!("CARGO_BIN_EXE_hades"), "--db", pool.database(), "--gpu", "0",
                    "graph-embed", "update", "--checkpoint-dir"])
                    .arg(&absent_checkpoint);
                if missing_only { command.arg("--new-nodes"); }
                command.output().unwrap()
            };
            let noop = cli(true);
            assert!(noop.status.success(), "no-op failed: {}", String::from_utf8_lossy(&noop.stderr));
            let reply: Value = serde_json::from_slice(&noop.stdout).unwrap();
            assert_eq!(reply["success"], true);
            assert_eq!(reply["data"]["model"]["checkpoint_validated"], false);
            assert_eq!(reply["data"]["model"]["service_contacted"], false);
            assert_eq!(reply["data"]["export"]["count"], 0);
            for (id, expected_row) in names.iter().zip(&before_noop) {
                let (collection, key) = id.split_once('/').unwrap();
                assert_eq!(crud::get_document(&pool, collection, key).await.unwrap(), *expected_row);
            }
            assert!(!absent_checkpoint.exists());
            let full_without_checkpoint = cli(false);
            assert!(!full_without_checkpoint.status.success());
            assert!(String::from_utf8_lossy(&full_without_checkpoint.stderr).contains("checkpoint"));
            crud::update_document(&pool, "papers", "same", &json!({"structural_embedding":null}))
                .await.unwrap();
            let missing_row = crud::get_document(&pool, "papers", "same").await.unwrap();
            let needs_work = cli(true);
            assert!(!needs_work.status.success());
            assert!(String::from_utf8_lossy(&needs_work.stderr).contains("checkpoint"));
            assert_eq!(crud::get_document(&pool, "papers", "same").await.unwrap(), missing_row);
            assert!(!absent_checkpoint.exists());
            println!("CLI missing-only: changed database features plus mixed existing vectors produce no-op with exact row/revision preservation; full mode and a null vector require the absent checkpoint. Live training socket and GPU devices hidden.");
        },
    )
    .await;
}

#[tokio::test]
#[ignore = "requires the private database runner and an explicit CPU Python interpreter"]
async fn actual_cli_training_exports_best_checkpoint_vectors() {
    let python =
        std::env::var("HADES_ALIGNMENT_PYTHON").expect("explicit CPU interpreter required");
    with_temp_db("cli_training_lifecycle", Fixtures::Empty, move |pool| async move {
        for name in ["papers", "concepts", "hades_schema"] {
            crud::create_collection(&pool, name, Some(2)).await.unwrap();
        }
        crud::create_collection(&pool, "links", Some(3)).await.unwrap();
        for collection in ["papers", "concepts"] {
            let rows: Vec<_> = (0..4).map(|i| json!({"_key":format!("n{i}"), "embedding":[i as f32 + 1., 0.5], "model":"fixture:v1"})).collect();
            crud::insert_documents(&pool, collection, &rows, false).await.unwrap();
        }
        let edges: Vec<_> = (0..12).map(|i| json!({"_key":format!("e{i}"), "_from":format!("papers/n{}", i/3), "_to":format!("concepts/n{}", i%3)})).collect();
        crud::insert_documents(&pool, "links", &edges, false).await.unwrap();
        crud::insert_documents(&pool, "hades_schema", &[
            json!({"_key":"meta", "schema_type":"schema_meta", "relation_order":["links"], "num_relations":1, "feature_dim":2, "model_type":"hetero_sage"}),
            json!({"_key":"links", "schema_type":"edge_definition", "name":"links", "from_collections":["papers"], "to_collections":["concepts"]}),
        ], false).await.unwrap();
        let schema = RuntimeSchema::load(&pool).await.unwrap();
        let (graph, ids) = graph::load(&pool, &schema).await.unwrap();
        let names: Vec<_> = (0..graph.num_nodes).map(|i| ids.get_arango_id(i).unwrap()).collect();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("manifest.json"), serde_json::to_vec(&json!({
            "ids":names, "database":pool.database(), "database_socket":std::env::var("ARANGO_SOCKET").unwrap(), "cli_binary":env!("CARGO_BIN_EXE_hades"),
        })).unwrap()).unwrap();
        let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services/tests/cli_training_lifecycle_peer.py");
        let output = std::process::Command::new("timeout").args(["--kill-after=5", "90", &python]).arg(peer).arg(root.path()).env("CUDA_VISIBLE_DEVICES", "").env("PYTHONDONTWRITEBYTECODE", "1").output().unwrap();
        assert!(output.status.success(), "CPU training failed: {}", String::from_utf8_lossy(&output.stderr));
        println!("CLI training lifecycle: {}", String::from_utf8_lossy(&output.stdout));
    }).await;
}
