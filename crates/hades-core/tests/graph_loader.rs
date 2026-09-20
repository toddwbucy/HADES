//! Graph loader contracts with a freshly seeded schema and graph per test.
//! Use a separate ArangoDB server; ARANGO_TESTS=1 makes prerequisites strict.
use hades_core::db::{ArangoPool, crud};
use hades_core::graph::{self, RuntimeSchema};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;

async fn with_graph<F, Fut>(tag: &str, f: F)
where
    F: FnOnce(ArangoPool, RuntimeSchema) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    with_temp_db(tag, Fixtures::Empty, |pool| async move {
        for collection in ["papers", "concepts", "hades_schema"] {
            crud::create_collection(&pool, collection, Some(2))
                .await
                .unwrap();
        }
        crud::create_collection(&pool, "links", Some(3))
            .await
            .unwrap();
        crud::insert_documents(
            &pool,
            "papers",
            &[
                json!({"_key":"one", "embedding":[1.,0.], "model":"fixture:v1"}),
                json!({"_key":"two"}),
            ],
            false,
        )
        .await
        .unwrap();
        crud::insert_documents(
            &pool,
            "concepts",
            &[json!({"_key":"concept", "embedding":[0.,1.], "model":"fixture:v1"})],
            false,
        )
        .await
        .unwrap();
        crud::insert_documents(
            &pool,
            "links",
            &[
                json!({"_key":"a", "_from":"papers/one", "_to":"concepts/concept"}),
                json!({"_key":"b", "_from":"papers/two", "_to":"concepts/concept"}),
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
        let schema = RuntimeSchema::load(&pool)
            .await
            .expect("fixture schema must load");
        f(pool, schema).await;
    })
    .await;
}

#[tokio::test]
async fn test_load_full_graph() {
    with_graph("test_load_full_graph", |pool, schema| async move {
        let (graph, id_map) = graph::load(&pool, &schema)
            .await
            .expect("graph load failed");

        // Structural invariants — graph dimensions must match the loaded schema.
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 2);
        assert_eq!(graph.num_relations, schema.meta.num_relations);
        assert_eq!(graph.feature_dim, schema.meta.feature_dim);

        // Validate passes (already called internally, but double-check)
        graph.validate().unwrap();

        // Edge arrays match counts
        assert_eq!(graph.edge_src.len(), graph.num_edges);
        assert_eq!(graph.edge_dst.len(), graph.num_edges);
        assert_eq!(graph.edge_type.len(), graph.num_edges);

        // Node features are sized correctly
        assert_eq!(
            graph.node_features.len(),
            graph.num_nodes * graph.feature_dim
        );
        assert_eq!(graph.has_embedding.len(), graph.num_nodes);
        assert_eq!(graph.node_collections.len(), graph.num_nodes);

        // IDMap is consistent with graph
        assert_eq!(id_map.len(), graph.num_nodes);

        // Collection names are sorted and non-empty
        assert!(!graph.collection_names.is_empty());
        let mut sorted = graph.collection_names.clone();
        sorted.sort();
        assert_eq!(
            graph.collection_names, sorted,
            "collection_names must be sorted"
        );

        // Exactly two fixture nodes carry verified embedding provenance.
        assert_eq!(graph.embedded_count(), 2);

        // All edge relation types should be valid indices
        for &rel in &graph.edge_type {
            assert!(
                (rel as usize) < schema.meta.num_relations,
                "edge relation type {rel} out of bounds"
            );
        }

        // All edge node indices should be valid
        for (&src, &dst) in graph.edge_src.iter().zip(&graph.edge_dst) {
            assert!(
                (src as usize) < graph.num_nodes,
                "edge src {src} out of bounds"
            );
            assert!(
                (dst as usize) < graph.num_nodes,
                "edge dst {dst} out of bounds"
            );
        }

        // Print summary for manual inspection
        tracing::info!(
            num_nodes = graph.num_nodes,
            num_edges = graph.num_edges,
            num_relations = graph.num_relations,
            num_collections = graph.collection_names.len(),
            embedded = graph.embedded_count(),
            total = graph.num_nodes,
            coverage_pct = format_args!("{:.1}", graph.embedding_coverage() * 100.0),
            "graph load summary"
        );
    })
    .await;
}

#[tokio::test]
async fn test_idmap_collection_grouping() {
    with_graph(
        "test_idmap_collection_grouping",
        |pool, schema| async move {
            let (_graph, id_map) = graph::load(&pool, &schema)
                .await
                .expect("graph load failed");

            let groups = id_map.nodes_by_collection();

            // Every node should be in exactly one group
            let total: usize = groups.values().map(|v| v.len()).sum();
            assert_eq!(total, id_map.len());

            // Every node's _id prefix (before '/') must exactly match its group key
            for (col, nodes) in &groups {
                assert!(!col.is_empty(), "collection name should not be empty");
                for (arango_id, _) in nodes {
                    let (prefix, _key) = arango_id
                        .split_once('/')
                        .unwrap_or_else(|| panic!("arango_id {arango_id} missing '/' separator"));
                    assert_eq!(
                        prefix, *col,
                        "node {arango_id} has prefix {prefix}, expected {col}"
                    );
                }
            }
        },
    )
    .await;
}
