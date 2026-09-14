//! A throwaway database per test, dropped even when the test panics.
//!
//! **Why a database and not a collection.** The sweeps under test are
//! collection-global: `prune-orphans` deletes every orphan in the database and
//! `retire` removes every edge incident on its targets, not just fixture rows.
//! Run against a shared database, an exact-count assertion depends on whatever
//! else happens to be there, and a failing assertion deletes unrelated records
//! as a side effect. An isolated database makes the counts exact and the blast
//! radius nil.
//!
//! **Why it is here rather than in each test module.** Three write tests each
//! carried their own harness until 2026-09-14. Two of them named `bident_burn`,
//! a database that had been dropped, and so either panicked on a 404 or skipped
//! without a word, depending on the flags. The third, in `codebase_prune`, had
//! the right shape and this is that shape, with two fixes: the name carries a
//! per-process counter as well as the pid, because two tests in one binary run
//! in parallel with the same pid and the delete-before-create in one was
//! dropping the other's database mid-run; and the user is `HADES_TEST_USER`,
//! which `codebase_ingest`'s live tests already honoured while the other two
//! hardcoded `root`.
//!
//! The rule it implements is CLAUDE.md's: a write test gets a database created
//! for the test, never a corpus somebody is querying.

use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;

use hades_core::db::collections::CODEBASE;
use hades_core::db::crud;
use hades_core::db::{ArangoClient, ArangoPool};

/// Distinguishes databases created by tests running in one process.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// ArangoDB's collection type for edges.
const EDGE_COLLECTION_TYPE: u32 = 3;

/// `ARANGO_TESTS=1` turns a skip into a failure, so a run that exercised
/// nothing cannot report green.
pub(crate) fn strict() -> bool {
    std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true")
}

/// Create a throwaway database holding the codebase collections, run `f`
/// against it, then drop it whether or not `f` panicked.
///
/// Skips, with a line on stderr, when no ArangoDB socket is present or no
/// password is set. Under [`strict`] a skip is a panic, and so is a database
/// the server refused to create: the earlier harness turned every refusal into
/// a skip, so a wrong password made five tests pass without running.
pub(crate) async fn with_temp_db<F, Fut>(f: F)
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
        "hades_test_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let sys = ArangoClient::with_socket(socket.clone(), "_system", &user, &password);
    // A leftover from a killed run with the same pid and counter is ours to
    // remove; nothing else can carry this name.
    let _ = sys.delete(&format!("database/{db_name}")).await;
    if let Err(e) = sys.post("database", &json!({ "name": db_name })).await {
        if strict() {
            panic!("ARANGO_TESTS is set but creating '{db_name}' as '{user}' failed: {e}");
        }
        eprintln!("skipping: cannot create test database '{db_name}' as '{user}': {e}");
        return;
    }

    let client = ArangoClient::with_socket(socket, &db_name, &user, &password);
    let pool = ArangoPool::new(client.clone(), client);
    for col in [
        CODEBASE.files,
        CODEBASE.chunks,
        CODEBASE.embeddings,
        CODEBASE.symbols,
    ] {
        crud::create_collection(&pool, col, None)
            .await
            .expect("create fixture collection");
    }
    // The edge collections too, and not only because a real graph has them.
    // `retire`'s sweep binds all four, and ArangoDB's 1203 (collection not
    // found) on any of them is mapped to an empty result rather than an error,
    // so a database without them makes the sweep report nothing and the
    // regression guard for #158 assert on a null. That is how the guard was
    // found to have never run: the first database it ever executed in was this
    // one, and it lacked the collections.
    for col in [
        CODEBASE.defines_edges,
        CODEBASE.calls_edges,
        CODEBASE.implements_edges,
        CODEBASE.imports_edges,
    ] {
        crud::create_collection(&pool, col, Some(EDGE_COLLECTION_TYPE))
            .await
            .expect("create fixture edge collection");
    }

    // `JoinHandle` captures a panic instead of unwinding through us, so the
    // database is dropped either way and the failure is re-raised after.
    let outcome = tokio::task::spawn(async move { f(pool).await }).await;

    let _ = sys.delete(&format!("database/{db_name}")).await;

    if let Err(join_err) = outcome {
        std::panic::resume_unwind(join_err.into_panic());
    }
}
