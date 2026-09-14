//! Integration tests for the cache layer.
//!
//! Each test runs in a throwaway database that
//! `hades_core::test_support::with_temp_db` creates and drops, so nothing here
//! touches a corpus and a failed assertion leaks nothing. See that module for
//! the environment it reads, including that `HADES_TEST_USER` needs `rw` on
//! `_system`.

use hades_core::db::cache::CachedPool;
use hades_core::db::crud;
use hades_core::db::query::ExecutionTarget;
use hades_core::test_support::{Fixtures, with_temp_db};

#[tokio::test]
async fn test_cached_get_document() {
    with_temp_db("cache", Fixtures::Empty, |pool| async move {
        let col = format!("test_cache_get_{}", std::process::id());
        crud::create_collection(&pool, &col, Some(2)).await.unwrap();

        let doc = serde_json::json!({"_key": "doc1", "value": 42});
        crud::insert_documents(&pool, &col, &[doc], false)
            .await
            .unwrap();

        let cached = CachedPool::with_defaults(pool.clone());

        // First call: cache miss
        let result = cached.get_document(&col, "doc1").await.unwrap();
        assert_eq!(result["value"], 42);

        let m = cached.metrics();
        assert_eq!(m.doc_hits, 0);
        assert_eq!(m.doc_misses, 1);

        // Second call: cache hit
        let result2 = cached.get_document(&col, "doc1").await.unwrap();
        assert_eq!(result2["value"], 42);

        let m = cached.metrics();
        assert_eq!(m.doc_hits, 1);
        assert_eq!(m.doc_misses, 1);

        crud::drop_collection(&pool, &col, true).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_cached_delete_invalidates() {
    with_temp_db("cache", Fixtures::Empty, |pool| async move {
        let col = format!("test_cache_del_{}", std::process::id());
        crud::create_collection(&pool, &col, Some(2)).await.unwrap();

        let doc = serde_json::json!({"_key": "doc1", "value": 99});
        crud::insert_documents(&pool, &col, &[doc], false)
            .await
            .unwrap();

        let cached = CachedPool::with_defaults(pool.clone());

        // Populate cache
        cached.get_document(&col, "doc1").await.unwrap();
        assert_eq!(cached.metrics().doc_misses, 1);

        // Delete should invalidate
        cached.delete_document(&col, "doc1").await.unwrap();

        // Next get should be a miss (and fail with 404 since doc is deleted)
        let result = cached.get_document(&col, "doc1").await;
        assert!(result.is_err());
        assert_eq!(cached.metrics().doc_misses, 2);

        crud::drop_collection(&pool, &col, true).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_cached_query() {
    with_temp_db("cache", Fixtures::Empty, |pool| async move {
        let cached = CachedPool::with_defaults(pool);

        // First query: miss
        let r1 = cached
            .query("RETURN 42", None, None, false, ExecutionTarget::Reader)
            .await
            .unwrap();
        assert_eq!(r1.results[0], 42);
        assert_eq!(cached.metrics().query_misses, 1);
        assert_eq!(cached.metrics().query_hits, 0);

        // Same query: hit
        let r2 = cached
            .query("RETURN 42", None, None, false, ExecutionTarget::Reader)
            .await
            .unwrap();
        assert_eq!(r2.results[0], 42);
        assert_eq!(cached.metrics().query_hits, 1);
    })
    .await
}

#[tokio::test]
async fn test_cached_query_writer_bypasses_cache() {
    with_temp_db("cache", Fixtures::Empty, |pool| async move {
        let cached = CachedPool::with_defaults(pool);

        // Writer queries should not be cached
        cached
            .query("RETURN 1", None, None, false, ExecutionTarget::Writer)
            .await
            .unwrap();

        cached
            .query("RETURN 1", None, None, false, ExecutionTarget::Writer)
            .await
            .unwrap();

        // No cache activity for writer queries
        assert_eq!(cached.metrics().query_hits, 0);
        assert_eq!(cached.metrics().query_misses, 0);
    })
    .await
}

#[tokio::test]
async fn test_invalidate_all() {
    with_temp_db("cache", Fixtures::Empty, |pool| async move {
        let cached = CachedPool::with_defaults(pool);

        // Populate query cache
        cached
            .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
            .await
            .unwrap();

        assert_eq!(cached.metrics().query_misses, 1);

        // Verify it's cached (hit)
        cached
            .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
            .await
            .unwrap();
        assert_eq!(cached.metrics().query_hits, 1);

        cached.invalidate_all();

        // After invalidation, next call should be a miss
        cached
            .query("RETURN 1", None, None, false, ExecutionTarget::Reader)
            .await
            .unwrap();

        assert_eq!(cached.metrics().query_misses, 2);
    })
    .await
}
