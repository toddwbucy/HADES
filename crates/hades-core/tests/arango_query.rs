//! Isolated ArangoDB integration tests. Use a separate test server; ARANGO_TESTS=1
//! makes missing prerequisites fail. Each test creates and drops its own database.

mod common;
use hades_core::db::query::{self, ExecutionTarget};

#[tokio::test]
async fn test_simple_query() {
    common::with_tasks_db("query_test_simple_query", |pool| async move {
        let result = query::query(
            &pool,
            "RETURN 1 + 1",
            None,
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0], 2);
    })
    .await;
}

#[tokio::test]
async fn test_query_with_bind_vars() {
    common::with_tasks_db("query_test_query_with_bind_vars", |pool| async move {
        let vars = serde_json::json!({"value": 42});
        let result = query::query(
            &pool,
            "RETURN @value",
            Some(&vars),
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0], 42);
    })
    .await;
}

#[tokio::test]
async fn test_query_collection() {
    common::with_tasks_db("query_test_query_collection", |pool| async move {
        let result = query::query(
            &pool,
            "FOR t IN persephone_tasks LIMIT 3 RETURN t._key",
            None,
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await
        .unwrap();
        assert!(!result.results.is_empty(), "expected at least one task key");
        assert!(result.results.len() <= 3);
    })
    .await;
}

#[tokio::test]
async fn test_query_full_count() {
    common::with_tasks_db("query_test_query_full_count", |pool| async move {
        let result = query::query(
            &pool,
            "FOR t IN persephone_tasks LIMIT 2 RETURN t._key",
            None,
            None,
            true,
            ExecutionTarget::Reader,
        )
        .await
        .unwrap();
        assert!(
            result.full_count.is_some(),
            "expected full_count with fullCount=true"
        );
        // full_count should be >= the number of results
        assert!(result.full_count.unwrap() >= result.results.len() as u64);
    })
    .await;
}

#[tokio::test]
async fn test_query_pagination() {
    common::with_tasks_db("query_test_query_pagination", |pool| async move {
        // Use a very small batch_size to force pagination
        let result = query::query(
            &pool,
            "FOR t IN persephone_tasks LIMIT 5 RETURN t._key",
            None,
            Some(2), // batch_size=2, should paginate across 3 pages
            false,
            ExecutionTarget::Reader,
        )
        .await
        .unwrap();
        assert_eq!(
            result.results.len(),
            5,
            "expected 5 results after pagination"
        );
    })
    .await;
}

#[tokio::test]
async fn test_query_single() {
    common::with_tasks_db("query_test_query_single", |pool| async move {
        let result = query::query_single(&pool, "RETURN 'hello'", None, ExecutionTarget::Reader)
            .await
            .unwrap();
        assert_eq!(result, Some(serde_json::json!("hello")));
    })
    .await;
}

#[tokio::test]
async fn test_query_single_empty() {
    common::with_tasks_db("query_test_query_single_empty", |pool| async move {
        let result =
            query::query_single(&pool, "FOR x IN [] RETURN x", None, ExecutionTarget::Reader)
                .await
                .unwrap();
        assert_eq!(result, None);
    })
    .await;
}

#[tokio::test]
async fn test_query_syntax_error() {
    common::with_tasks_db("query_test_query_syntax_error", |pool| async move {
        let result = query::query(
            &pool,
            "THIS IS NOT VALID AQL",
            None,
            None,
            false,
            ExecutionTarget::Reader,
        )
        .await;
        assert!(result.is_err(), "expected error for invalid AQL");
    })
    .await;
}
