//! Write a bounded synthetic Rust graph for Python checkpoint-contract tests.
use hades_core::graph::types::{GraphContract, GraphData};
use hades_prefetch::{EdgeSplit, NegativeSamples, SplitConfig, serialize_to_file};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args()
        .nth(1)
        .ok_or("provide an output .safetensors path")?;
    let mut graph = GraphData::with_schema_capacity(4, 3, 2, 6);
    graph.collection_names = vec!["a".into(), "b".into()];
    graph.node_collections = vec![0, 1, 0, 1];
    for node in 0..4 {
        graph.set_node_features(node, &[node as f32 / 10.0; 6]);
    }
    for (src, dst, rel) in [(0, 1, 0), (1, 2, 1), (2, 3, 0)] {
        assert!(graph.add_edge(src, dst, rel));
    }
    graph.contract = Some(GraphContract {
        version: 1,
        relation_order: vec!["first".into(), "second".into()],
        collection_names: graph.collection_names.clone(),
        feature_dim: 6,
        architecture: "hetero_sage".into(),
        feature_policy: GraphContract::FEATURE_POLICY.into(),
        feature_models: graph
            .collection_names
            .iter()
            .map(|name| (name.clone(), vec!["synthetic:v1".into()]))
            .collect(),
    });
    serialize_to_file(
        Path::new(&output),
        &graph,
        &EdgeSplit {
            train_idx: vec![0],
            val_idx: vec![1],
            test_idx: vec![2],
        },
        &NegativeSamples {
            src: vec![3],
            dst: vec![0],
        },
        &SplitConfig::default(),
    )?;
    Ok(())
}
