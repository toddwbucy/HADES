//! Transport contracts against a separate server and a fresh database per test.
//! ARANGO_TESTS=1 makes missing socket/credentials fail instead of skipping.
mod common;

#[tokio::test]
async fn get_server_version() {
    common::with_tasks_db("transport_version", |pool| async move {
        let result = pool.reader().get("version").await.unwrap();
        assert_eq!(result["server"], "arango");
    })
    .await;
}

#[tokio::test]
async fn list_databases() {
    common::with_tasks_db("transport_databases", |pool| async move {
        let admin = hades_core::db::ArangoClient::with_socket(
            pool.writer().socket_path().unwrap().to_path_buf(),
            "_system",
            &std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into()),
            &std::env::var("ARANGO_PASSWORD").unwrap(),
        );
        let result = admin.get("database").await.unwrap();
        let names: Vec<_> = result["result"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(names.contains(&"_system"));
        assert!(names.contains(&pool.database()));
    })
    .await;
}

#[tokio::test]
async fn list_fixture_collections() {
    common::with_tasks_db("transport_collections", |pool| async move {
        let result = pool.reader().get("collection").await.unwrap();
        let names: Vec<_> = result["result"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert!(names.contains(&"persephone_tasks"));
    })
    .await;
}

#[tokio::test]
async fn get_nonexistent_document_returns_error() {
    common::with_tasks_db("transport_missing", |pool| async move {
        let error = pool
            .reader()
            .get("document/persephone_tasks/nonexistent_key_xyz")
            .await
            .unwrap_err();
        assert!(error.is_not_found(), "expected NotFound, got: {error}");
    })
    .await;
}

#[tokio::test]
async fn execute_aql_query() {
    common::with_tasks_db("transport_aql", |pool| async move {
        let result = pool.reader().post("cursor", &serde_json::json!({
            "query":"FOR t IN persephone_tasks SORT t._key LIMIT 3 RETURN t._key", "batchSize":10,
        })).await.unwrap();
        assert_eq!(
            result["result"],
            serde_json::json!(["task_0", "task_1", "task_2"])
        );
    })
    .await;
}

#[tokio::test]
async fn pool_health_check() {
    common::with_tasks_db("transport_health", |pool| async move {
        assert!(pool.is_shared());
        let status = pool.health_check().await;
        assert!(status.reader_ok);
        assert!(status.writer_ok);
        assert!(!status.version.is_empty());
    })
    .await;
}

#[tokio::test]
async fn pool_reader_writer_queries() {
    common::with_tasks_db("transport_pool", |pool| async move {
        let result = pool.reader().get("collection").await.unwrap();
        assert!(!result["result"].as_array().unwrap().is_empty());
        let result = pool.writer().get("version").await.unwrap();
        assert_eq!(result["server"], "arango");
    })
    .await;
}
