//! Each integration test owns its complete disposable database and seed data.
use hades_core::db::{ArangoPool, crud};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;

pub async fn with_tasks_db<F, Fut>(tag: &str, f: F)
where
    F: FnOnce(ArangoPool) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    with_temp_db(tag, Fixtures::Empty, |pool| async move {
        crud::create_collection(&pool, "persephone_tasks", None)
            .await
            .unwrap();
        let docs: Vec<_> = (0..5).map(|i| json!({
            "_key": format!("task_{i}"), "title": format!("Fixture task {i}"), "status": "todo",
        })).collect();
        crud::insert_documents(&pool, "persephone_tasks", &docs, false)
            .await
            .unwrap();
        f(pool).await;
    })
    .await;
}
