//! A throwaway database per test, dropped even when the test panics.
//!
//! The integration targets under `tests/` cannot share code with the CLI
//! crate's test harness, so this is the same shape kept in step with
//! `crates/hades-cli/src/commands/test_db.rs`: a per-process name carrying the
//! pid and a counter, delete-before-create, the user from `HADES_TEST_USER`,
//! and the database dropped whether or not the test panicked. It creates no
//! collections, since each target seeds what it needs.
//!
//! The rule it implements is CLAUDE.md's: a write test gets a database created
//! for the test, never a corpus somebody is querying. Until 2026-09-14 these
//! targets named `bident_burn`, a database that was then dropped, and so either
//! panicked on a 404 or skipped without a word depending on the flags.
//!
//! `ARANGO_SOCKET` names the socket (default `/run/arangodb3/arangodb.sock`,
//! which a user-level deployment does not have). `ARANGO_TESTS=1` turns a skip
//! into a failure, so a run that exercised nothing cannot report green.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use hades_core::db::{ArangoClient, ArangoPool};

static SEQ: AtomicU32 = AtomicU32::new(0);

pub fn strict() -> bool {
    std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true")
}

/// Create an empty throwaway database, run `f` against it, then drop it.
pub async fn with_temp_db<F, Fut>(tag: &str, f: F)
where
    F: FnOnce(ArangoPool) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let socket = std::path::PathBuf::from(
        std::env::var("ARANGO_SOCKET")
            .unwrap_or_else(|_| "/run/arangodb3/arangodb.sock".to_string()),
    );
    if !socket.exists() {
        if strict() {
            panic!(
                "ARANGO_TESTS is set but socket not found at {}",
                socket.display()
            );
        }
        eprintln!(
            "skipping: ArangoDB socket not found at {}",
            socket.display()
        );
        return;
    }
    let Ok(password) = std::env::var("ARANGO_PASSWORD") else {
        if strict() {
            panic!("ARANGO_TESTS is set but ARANGO_PASSWORD is not");
        }
        eprintln!("skipping: ARANGO_PASSWORD not set");
        return;
    };
    let user = std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".to_string());

    let db_name = format!(
        "hades_test_{tag}_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let sys = ArangoClient::with_socket(socket.clone(), "_system", &user, &password);
    let _ = sys.delete(&format!("database/{db_name}")).await;
    if let Err(e) = sys
        .post("database", &serde_json::json!({ "name": db_name }))
        .await
    {
        if strict() {
            panic!("ARANGO_TESTS is set but creating '{db_name}' as '{user}' failed: {e}");
        }
        eprintln!("skipping: cannot create test database '{db_name}' as '{user}': {e}");
        return;
    }

    let client = ArangoClient::with_socket(socket, &db_name, &user, &password);
    let pool = ArangoPool::new(client.clone(), client);

    let outcome = tokio::task::spawn(async move { f(pool).await }).await;

    let _ = sys.delete(&format!("database/{db_name}")).await;

    if let Err(join_err) = outcome {
        std::panic::resume_unwind(join_err.into_panic());
    }
}
