//! Requires the strict disposable-server runner; never discovers a live endpoint.
use hades_core::db::crud;
use hades_core::graph::export::{
    ExportConfig, ExportError, export_embeddings, export_embeddings_subset,
};
use hades_core::graph::types::IDMap;
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;

#[tokio::test]
async fn real_export_and_missing_target_preserve_explicit_partial_progress() {
    with_temp_db("structural_export", Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "nodes", Some(2))
            .await
            .unwrap();
        crud::insert_documents(
            &pool,
            "nodes",
            &[json!({"_key":"a"}), json!({"_key":"b"})],
            false,
        )
        .await
        .unwrap();
        let mut ids = IDMap::new();
        ids.get_or_create("nodes/a");
        ids.get_or_create("nodes/b");
        let config = ExportConfig { chunk_size: 1 };
        let result = export_embeddings(&pool, &ids, &[1., 2., 3., 4.], 2, &config)
            .await
            .unwrap();
        assert_eq!(result.total_exported, 2);
        export_embeddings_subset(&pool, &ids, &[1], &[9., 8.], 2, &config)
            .await
            .unwrap();
        assert_eq!(
            crud::get_document(&pool, "nodes", "a").await.unwrap()["structural_embedding"],
            json!([1., 2.])
        );
        assert_eq!(
            crud::get_document(&pool, "nodes", "b").await.unwrap()["structural_embedding"],
            json!([9., 8.])
        );
        let mut targets = IDMap::new();
        targets.get_or_create("nodes/a");
        targets.get_or_create("nodes/missing");
        targets.get_or_create("nodes/b");
        let failure = export_embeddings(&pool, &targets, &[5., 6., 7., 8., 10., 11.], 2, &config)
            .await
            .unwrap_err();
        assert!(matches!(
            failure,
            ExportError::BatchFailed {
                acknowledged: 1,
                attempted: 1,
                ..
            }
        ));
        assert_eq!(
            crud::get_document(&pool, "nodes", "a").await.unwrap()["structural_embedding"],
            json!([5., 6.])
        );
        assert_eq!(
            crud::get_document(&pool, "nodes", "b").await.unwrap()["structural_embedding"],
            json!([9., 8.])
        );
    })
    .await;
}
