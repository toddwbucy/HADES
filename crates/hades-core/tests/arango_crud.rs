//! Isolated ArangoDB integration tests. Use a separate test server; ARANGO_TESTS=1
//! makes missing prerequisites fail. Each test creates and drops its own database.

mod common;
use hades_core::db::{ArangoPool, crud, query};

/// Helper to ensure a collection exists, dropping first if present.
async fn setup(pool: &ArangoPool, name: &str) {
    let _ = crud::drop_collection(pool, name, true).await;
    crud::create_collection(pool, name, None)
        .await
        .expect("failed to create test collection");
}

/// Helper to clean up a collection.
async fn teardown(pool: &ArangoPool, name: &str) {
    let _ = crud::drop_collection(pool, name, true).await;
}

// ---------------------------------------------------------------------------
// Collection operations (read-only, no temp collection needed)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_list_collections() {
    common::with_tasks_db("crud_test_list_collections", |pool| async move {
        let collections = crud::list_collections(&pool, true).await.unwrap();
        let names: Vec<&str> = collections.iter().map(|c| c.name.as_str()).collect();

        assert!(
            names.contains(&"persephone_tasks"),
            "collections: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with('_')),
            "system collections not excluded"
        );
    })
    .await;
}

#[tokio::test]
async fn test_list_collections_include_system() {
    common::with_tasks_db(
        "crud_test_list_collections_include_system",
        |pool| async move {
            let collections = crud::list_collections(&pool, false).await.unwrap();
            let has_system = collections.iter().any(|c| c.name.starts_with('_'));
            assert!(
                has_system,
                "expected system collections when exclude_system=false"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn test_drop_collection_ignore_missing() {
    common::with_tasks_db(
        "crud_test_drop_collection_ignore_missing",
        |pool| async move {
            // Use a unique name to avoid any collision
            let name = format!("nonexistent_{}", std::process::id());
            let resp = crud::drop_collection(&pool, &name, true).await.unwrap();
            assert_eq!(resp["dropped"], false);
        },
    )
    .await;
}

#[tokio::test]
async fn test_count_collection() {
    common::with_tasks_db("crud_test_count_collection", |pool| async move {
        let count = crud::count_collection(&pool, "persephone_tasks")
            .await
            .unwrap();
        assert!(count > 0, "expected at least one task, got {count}");
    })
    .await;
}

// ---------------------------------------------------------------------------
// Collection create/drop (own collection)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_and_drop_collection() {
    common::with_tasks_db("crud_test_create_and_drop_collection", |pool| async move {
        let col = format!("test_crud_create_drop_{}", std::process::id());

        let _ = crud::drop_collection(&pool, &col, true).await;

        let resp = crud::create_collection(&pool, &col, None).await.unwrap();
        assert_eq!(resp["name"].as_str(), Some(col.as_str()));

        let collections = crud::list_collections(&pool, false).await.unwrap();
        assert!(collections.iter().any(|c| c.name == col));

        crud::drop_collection(&pool, &col, false).await.unwrap();

        let collections = crud::list_collections(&pool, false).await.unwrap();
        assert!(!collections.iter().any(|c| c.name == col));
    })
    .await;
}

// ---------------------------------------------------------------------------
// Document CRUD (each test gets its own collection)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_document_crud_lifecycle() {
    common::with_tasks_db("crud_test_document_crud_lifecycle", |pool| async move {
        let col = "test_crud_lifecycle";
        setup(&pool, col).await;

        // Insert
        let docs = vec![serde_json::json!({
            "_key": "test_doc_1",
            "title": "Test Document",
            "value": 42
        })];
        let result = crud::insert_documents(&pool, col, &docs, false)
            .await
            .unwrap();
        assert_eq!(result.created, 1);

        // Get
        let doc = crud::get_document(&pool, col, "test_doc_1").await.unwrap();
        assert_eq!(doc["title"], "Test Document");
        assert_eq!(doc["value"], 42);

        // Update (merge-patch)
        let update = serde_json::json!({"value": 99, "extra": "field"});
        crud::update_document(&pool, col, "test_doc_1", &update)
            .await
            .unwrap();

        let doc = crud::get_document(&pool, col, "test_doc_1").await.unwrap();
        assert_eq!(doc["value"], 99);
        assert_eq!(doc["extra"], "field");
        assert_eq!(doc["title"], "Test Document"); // preserved by PATCH

        // Delete
        crud::delete_document(&pool, col, "test_doc_1")
            .await
            .unwrap();

        let result = crud::get_document(&pool, col, "test_doc_1").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().is_not_found());

        teardown(&pool, col).await;
    })
    .await;
}

#[tokio::test]
async fn test_replace_document() {
    common::with_tasks_db("crud_test_replace_document", |pool| async move {
        let col = "test_crud_replace";
        setup(&pool, col).await;

        let docs = vec![serde_json::json!({
            "_key": "replace_test",
            "title": "Original",
            "extra": "will_be_removed"
        })];
        crud::insert_documents(&pool, col, &docs, false)
            .await
            .unwrap();

        let replacement = serde_json::json!({"title": "Replaced"});
        crud::replace_document(&pool, col, "replace_test", &replacement)
            .await
            .unwrap();

        let doc = crud::get_document(&pool, col, "replace_test")
            .await
            .unwrap();
        assert_eq!(doc["title"], "Replaced");
        assert!(doc.get("extra").is_none() || doc["extra"].is_null());

        teardown(&pool, col).await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// Bulk import
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bulk_insert() {
    common::with_tasks_db("crud_test_bulk_insert", |pool| async move {
        let col = "test_crud_bulk";
        setup(&pool, col).await;

        let docs: Vec<serde_json::Value> = (0..25)
            .map(|i| serde_json::json!({"_key": format!("bulk_{i}"), "index": i}))
            .collect();

        let result = crud::bulk_insert(&pool, col, &docs, Some(10), false)
            .await
            .unwrap();
        assert_eq!(result.created, 25);

        let count = crud::count_collection(&pool, col).await.unwrap();
        assert_eq!(count, 25);

        teardown(&pool, col).await;
    })
    .await;
}

#[tokio::test]
async fn test_insert_with_overwrite() {
    common::with_tasks_db("crud_test_insert_with_overwrite", |pool| async move {
        let col = "test_crud_overwrite";
        setup(&pool, col).await;

        let docs = vec![serde_json::json!({"_key": "ow_test", "version": 1})];
        crud::insert_documents(&pool, col, &docs, false)
            .await
            .unwrap();

        let docs = vec![serde_json::json!({"_key": "ow_test", "version": 2})];
        let result = crud::insert_documents(&pool, col, &docs, true)
            .await
            .unwrap();
        assert!(result.created > 0 || result.updated > 0);

        let doc = crud::get_document(&pool, col, "ow_test").await.unwrap();
        assert_eq!(doc["version"], 2);

        teardown(&pool, col).await;
    })
    .await;
}

#[tokio::test]
async fn test_bulk_insert_zero_chunk_size() {
    common::with_tasks_db("crud_test_bulk_insert_zero_chunk_size", |pool| async move {
        let col = "test_crud_zero_chunk";
        setup(&pool, col).await;

        let docs = vec![serde_json::json!({"_key": "x"})];
        let err = crud::bulk_insert(&pool, col, &docs, Some(0), false)
            .await
            .unwrap_err();
        assert_eq!(err.committed.created, 0);
        assert!(err.error.to_string().contains("chunk_size must be > 0"));

        teardown(&pool, col).await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// remove_docs_by_fields — the delete shape behind --force document refresh
// ---------------------------------------------------------------------------

/// Regression guard for #169: every `--force` document refresh aborted with
/// ArangoDB error 1552 because the inlined delete shared one bind map across
/// two single-collection queries. The refresh path now runs on
/// `remove_docs_by_fields`, so this asserts the thing the bug prevented:
/// the delete actually deletes, returns an accurate count, and leaves
/// non-matching documents alone. The multi-field case covers the OR-filter
/// the #165 fix layers on top (legacy rows keyed only by `parent_key`).
#[tokio::test]
async fn test_remove_docs_by_fields() {
    common::with_tasks_db("crud_test_remove_docs_by_fields", |pool| async move {
        let name = format!("test_rm_fields_{}", std::process::id());
        setup(&pool, &name).await;

        let docs = vec![
            serde_json::json!({ "_key": "a", "doc_key": "target", "text": "native row" }),
            serde_json::json!({ "_key": "b", "doc_key": "target", "text": "native row 2" }),
            serde_json::json!({ "_key": "c", "parent_key": "target", "text": "python-era row" }),
            serde_json::json!({ "_key": "d", "doc_key": "other", "text": "must survive" }),
        ];
        crud::insert_documents(&pool, &name, &docs, false)
            .await
            .expect("insert fixtures");

        // Single-field: the #169 refresh shape. Removes the two native rows only.
        let removed = query::remove_docs_by_fields(&pool, &name, &["doc_key"], "target")
            .await
            .expect("single-field remove");
        assert_eq!(removed, 2, "both doc_key rows removed, count accurate");

        // Multi-field OR: the #165 shape. Catches the python-era row too.
        let removed =
            query::remove_docs_by_fields(&pool, &name, &["doc_key", "parent_key"], "target")
                .await
                .expect("multi-field remove");
        assert_eq!(removed, 1, "parent_key-only row removed by the OR arm");

        // The unrelated document survives both passes.
        let survivor = crud::get_document(&pool, &name, "d").await;
        assert!(survivor.is_ok(), "non-matching document must survive");

        // Idempotent on empty.
        let removed = query::remove_docs_by_fields(&pool, &name, &["doc_key"], "target")
            .await
            .expect("re-run");
        assert_eq!(removed, 0);

        teardown(&pool, &name).await;
    })
    .await;
}
