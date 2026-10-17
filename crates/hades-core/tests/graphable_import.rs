//! Strict contracts run only through the isolated database harness.
use hades_core::{
    db::{ArangoPool, crud, query},
    graphable::{self, Plan},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::json;
const SAMPLE: &str = include_str!("fixtures/graphable/sample.md");
fn plan(source: &str) -> Plan {
    Plan::from_bytes(source.as_bytes(), "md", "theory").unwrap()
}
async fn count(pool: &ArangoPool, col: &str) -> u64 {
    pool.reader()
        .get(&format!("collection/{col}/count"))
        .await
        .unwrap()["count"]
        .as_u64()
        .unwrap()
}

#[tokio::test]
async fn repeat_new_version_and_conflicting_version_are_distinct() {
    with_temp_db("graphable",Fixtures::Empty,|pool|async move {
        let original=plan(SAMPLE);
        let a=graphable::import(&pool,&original,true).await.unwrap();
        assert_eq!(a["inserted_nodes"],original.nodes().len());
        let b=graphable::import(&pool,&original,false).await.unwrap();
        assert_eq!(b["inserted_nodes"],0);assert_eq!(b["inserted_edges"],0);assert_eq!(b["unchanged"],true);
        let changed=plan(&SAMPLE.replace("holds entries","holds revised entries"));
        assert!(graphable::import(&pool,&changed,false).await.unwrap_err().to_string().contains("different hash"));
        let next=plan(&SAMPLE.replace("'0.10'","'0.11'"));
        graphable::import(&pool,&next,false).await.unwrap();
        assert_eq!(count(&pool,"theory_nodes").await,2*original.nodes().len() as u64);
        assert_eq!(count(&pool,"theory_imports").await,2);
        for row in original.nodes() {
            let stored=crud::get_document(&pool,"theory_nodes",row["_key"].as_str().unwrap()).await.unwrap();
            assert_eq!(stored["source_version"],"0.10");
        }
        let root=a["root_id"].as_str().unwrap();
        let traversal=query::query(&pool,"FOR v,e IN 1..3 OUTBOUND @root GRAPH 'theory' FILTER e.relation == 'contains' RETURN DISTINCT v.logical_id",
            Some(&json!({"root":root})),None,false,query::ExecutionTarget::Reader).await.unwrap();
        assert!(traversal.results.contains(&json!("state-notes/buffer/boundary")));
        assert_eq!(a["embedding"],"not_requested");assert_eq!(a["semantic_search"],"not_verified");
    }).await;
}

#[tokio::test]
async fn late_conflict_rolls_back_nodes_and_retry_recovers_missing_rows() {
    with_temp_db("graphable_retry", Fixtures::Empty, |pool| async move {
        let source = plan(SAMPLE);
        let result = graphable::import(&pool, &source, true).await.unwrap();
        // Simulate an interrupted older/nontransactional loader: only one node
        // survives, there is no receipt, and an existing edge conflicts.
        for row in source.nodes().iter().skip(1) {
            pool.writer()
                .delete(&format!(
                    "document/theory_nodes/{}",
                    row["_key"].as_str().unwrap()
                ))
                .await
                .unwrap();
        }
        for row in source.edges() {
            pool.writer()
                .delete(&format!(
                    "document/theory_edges/{}",
                    row["_key"].as_str().unwrap()
                ))
                .await
                .unwrap();
        }
        pool.writer()
            .delete(&format!(
                "document/{}",
                result["source_archive_id"].as_str().unwrap()
            ))
            .await
            .unwrap();
        let mut conflict = source.edges()[0].clone();
        conflict["label"] = json!("user-authored edit");
        crud::insert_document(&pool, "theory_edges", &conflict)
            .await
            .unwrap();
        assert!(
            graphable::import(&pool, &source, false)
                .await
                .unwrap_err()
                .to_string()
                .contains("payload conflict")
        );
        // Node batches were sent before the edge conflict. None published.
        assert_eq!(count(&pool, "theory_nodes").await, 1);
        assert_eq!(count(&pool, "theory_imports").await, 0);
        let kept = crud::get_document(&pool, "theory_edges", conflict["_key"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(kept["label"], "user-authored edit");
        pool.writer()
            .delete(&format!(
                "document/theory_edges/{}",
                conflict["_key"].as_str().unwrap()
            ))
            .await
            .unwrap();
        let recovered = graphable::import(&pool, &source, false).await.unwrap();
        assert_eq!(recovered["inserted_nodes"], source.nodes().len() - 1);
        assert_eq!(recovered["inserted_edges"], source.edges().len());
        assert_eq!(
            count(&pool, "theory_nodes").await,
            source.nodes().len() as u64
        );
        assert_eq!(
            count(&pool, "theory_edges").await,
            source.edges().len() as u64
        );
        assert_eq!(
            graphable::import(&pool, &source, false).await.unwrap()["unchanged"],
            true
        );
    })
    .await;
}

#[tokio::test]
async fn concurrent_retries_publish_one_revision() {
    with_temp_db("graphable_race", Fixtures::Empty, |pool| async move {
        let source = plan(SAMPLE);
        graphable::import(&pool, &source, true).await.unwrap();
        let (a, b) = tokio::join!(
            graphable::import(&pool, &source, false),
            graphable::import(&pool, &source, false)
        );
        assert_eq!(a.unwrap()["unchanged"], true);
        assert_eq!(b.unwrap()["unchanged"], true);
        assert_eq!(count(&pool, "theory_imports").await, 1);
    })
    .await;
}

#[tokio::test]
async fn batched_retries_check_rows_beyond_the_first_batch() {
    with_temp_db("graphable_batches", Fixtures::Empty, |pool| async move {
        let sections = (0..1001).map(|i| format!(
            "## Section {i}\n<!-- node: state-notes/section-{i} | keywords: buffer -->\nSource content.\n\n"
        )).collect::<String>();
        let source = plan(&SAMPLE.replace("## Glossary", &(sections + "## Glossary")));
        assert!(source.nodes().len() > 1000);
        assert!(source.edges().len() > 1000);
        graphable::import(&pool, &source, true).await.unwrap();
        assert_eq!(graphable::import(&pool, &source, false).await.unwrap()["unchanged"], true);
        let last_node = source.nodes().last().unwrap();
        let node_path = format!("document/theory_nodes/{}", last_node["_key"].as_str().unwrap());
        pool.writer().delete(&node_path).await.unwrap();
        let recovered = graphable::import(&pool, &source, false).await.unwrap();
        assert_eq!(recovered["inserted_nodes"], 1);
        assert_eq!(recovered["inserted_edges"], 0);
        let last_edge = source.edges().last().unwrap();
        let edge_path = format!("document/theory_edges/{}", last_edge["_key"].as_str().unwrap());
        pool.writer().patch(&edge_path, &json!({"label": "reviewed edit"})).await.unwrap();
        assert!(graphable::import(&pool, &source, false).await.unwrap_err().to_string().contains("payload conflict"));
        assert_eq!(pool.reader().get(&edge_path).await.unwrap()["label"], "reviewed edit");
        assert_eq!(count(&pool, "theory_nodes").await, source.nodes().len() as u64);
        assert_eq!(count(&pool, "theory_edges").await, source.edges().len() as u64);
    }).await;
}
