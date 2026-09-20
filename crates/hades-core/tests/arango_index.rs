//! Isolated ArangoDB integration tests. Use a separate test server; ARANGO_TESTS=1
//! makes missing prerequisites fail. Each test creates and drops its own database.

mod common;
use hades_core::db::index::{self, VectorMetric};
use tracing::warn;

#[tokio::test]
async fn test_list_indexes() {
    common::with_tasks_db("index_test_list_indexes", |pool| async move {
        // The fixture owns a seeded task collection.
        let indexes = index::list_indexes(&pool, "persephone_tasks")
            .await
            .unwrap();

        // Every collection has at least a primary index
        assert!(!indexes.is_empty(), "expected at least primary index");

        let primary = indexes.iter().find(|i| i.index_type == "primary");
        assert!(primary.is_some(), "expected a primary index");
    })
    .await;
}

#[tokio::test]
async fn test_list_indexes_nonexistent_collection() {
    common::with_tasks_db(
        "index_test_list_indexes_nonexistent_collection",
        |pool| async move {
            let result = index::list_indexes(&pool, "nonexistent_collection_xyz").await;
            assert!(result.is_err(), "expected error for nonexistent collection");
        },
    )
    .await;
}

#[tokio::test]
async fn test_create_and_drop_vector_index() {
    use hades_core::db::crud;

    common::with_tasks_db("index_test_create_and_drop_vector_index", |pool| async move {

    let col_name = format!("test_vec_idx_{}", std::process::id());

    // Create a test collection
    crud::create_collection(&pool, &col_name, Some(2))
        .await
        .unwrap();

    // Insert a few documents with embedding fields so the index has something
    let docs: Vec<serde_json::Value> = (0..5)
        .map(|i| {
            serde_json::json!({
                "_key": format!("doc_{i}"),
                "embedding": (0..128).map(|j| if j == i { 1.0_f64 } else { 0.0 }).collect::<Vec<_>>(),
                "chunk_key": format!("chunk_{i}"),
            })
        })
        .collect();
    crud::insert_documents(&pool, &col_name, &docs, false)
        .await
        .unwrap();

    // Create a vector index (requires --experimental-vector-index flag)
    let idx = match index::create_vector_index(
        &pool,
        &col_name,
        "embedding",
        128,
        Some(1), // explicit nLists for small collection
        10,
        VectorMetric::Cosine,
    )
    .await
    {
        Ok(idx) => idx,
        Err(e)
            if e.to_string()
                .contains("vector index feature is not enabled") =>
        {
            if std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true") {
                panic!("ARANGO_TESTS requires --experimental-vector-index=true on the test server");
            }
            warn!("skipping: ArangoDB vector index feature not enabled");
            crud::drop_collection(&pool, &col_name, true).await.unwrap();
            return;
        }
        Err(e) => panic!("unexpected error creating vector index: {e}"),
    };

    assert_eq!(idx.index_type, "vector");
    assert!(idx.fields.contains(&"embedding".to_string()));

    // Verify it shows up in list
    let indexes = index::list_indexes(&pool, &col_name).await.unwrap();
    let vector_idx = indexes.iter().find(|i| i.index_type == "vector");
    assert!(vector_idx.is_some(), "expected vector index in list");

    // Detect it
    let metric = index::detect_vector_index(&pool, &col_name).await.unwrap();
    assert_eq!(metric, Some(VectorMetric::Cosine));

    // Drop it
    index::drop_index(&pool, &idx.id).await.unwrap();

    // Verify it's gone
    let metric_after = index::detect_vector_index(&pool, &col_name).await.unwrap();
    assert_eq!(metric_after, None);

    // Cleanup
    crud::drop_collection(&pool, &col_name, true).await.unwrap();
    }).await;
}

#[tokio::test]
async fn test_detect_vector_index_none() {
    common::with_tasks_db("index_test_detect_vector_index_none", |pool| async move {
        // persephone_tasks should not have a vector index
        let metric = index::detect_vector_index(&pool, "persephone_tasks")
            .await
            .unwrap();
        assert_eq!(metric, None);
    })
    .await;
}
