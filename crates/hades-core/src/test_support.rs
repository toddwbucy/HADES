//! A throwaway database per test, dropped even when the test panics.
//!
//! Behind the `test-support` feature, which is off by default and enabled only
//! through a dev-dependency, so nothing here reaches a release build.
//!
//! **Why a database and not a collection.** The sweeps under test are
//! collection-global: `prune-orphans` deletes every orphan in the database and
//! `retire` removes every edge incident on its targets, not just fixture rows.
//! Run against a shared database, an exact-count assertion depends on whatever
//! else happens to be there, and a failing assertion deletes unrelated records
//! as a side effect. An isolated database makes the counts exact and the blast
//! radius nil.
//!
//! **Why it lives in the library.** Three write tests each carried their own
//! copy until 2026-09-14. Two named `bident_burn`, a database that had been
//! dropped, so they either panicked on a 404 or skipped without a word
//! depending on the flags. The first fix moved them onto one shape kept in two
//! places, `hades-cli`'s test module and `hades-core`'s `tests/common`, which is
//! the same two-halves-of-one-rule problem this repository has spent two days
//! removing: the copies had already drifted in signature, naming and fixtures,
//! and "kept in step" had no mechanism. `hades-cli` is a bin-only crate whose
//! test module nothing can import, so the single definition goes here, where
//! both its own `tests/` targets and the CLI's unit tests can reach it.
//!
//! The rule it implements is CLAUDE.md's: a write test gets a database created
//! for the test, never a corpus somebody is querying.
//!
//! ## Environment
//!
//! - `ARANGO_SOCKET` must explicitly name a separate test server's socket.
//!   No default system socket is used by write fixtures.
//! - `ARANGO_PASSWORD` is required.
//! - `HADES_TEST_USER` is the user, default `root`. **It needs `rw` on
//!   `_system`**, since this creates and drops databases. `root` has it, and
//!   `scripts/install/setup-arangodb-user.sh` grants it to the `hades` user.
//! - `ARANGO_TESTS=1` turns every skip into a panic, so a run that exercised
//!   nothing cannot report green.

use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;

use crate::db::collections::CODEBASE;
use crate::db::crud;
use crate::db::{ArangoClient, ArangoPool};

/// Distinguishes databases created by tests running in one process.
///
/// The pid alone is not enough: tests in one binary run in parallel and share
/// it, so the delete-before-create in one was dropping a sibling's database
/// mid-run.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// `ARANGO_TESTS=1` turns a skip into a failure.
fn strict() -> bool {
    std::env::var("ARANGO_TESTS").is_ok_and(|v| v == "1" || v == "true")
}

/// Which collections the fixture database starts with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fixtures {
    /// Nothing. The test creates what it needs.
    Empty,
    /// The universal code layer, from [`CODEBASE.all_collections`].
    ///
    /// Taken from that list rather than written out here, because it is the same
    /// list `ingest` creates from, and a ninth entry added there has to appear
    /// in the fixture too. `retire`'s sweep binds all four codebase edge
    /// collections and maps ArangoDB's 1203 (collection not found) to an empty
    /// result, so a fixture missing one makes the sweep silently report nothing.
    Codebase,
}

/// Create a throwaway database, run `f` against it, then drop it whether or not
/// `f` panicked.
///
/// `tag` distinguishes one target's databases from another's in a listing.
/// Skips, with a line on stderr, when the socket or password is absent; under
/// `ARANGO_TESTS=1` a skip is a panic, and so is a database the server refused
/// to create.
pub async fn with_temp_db<F, Fut>(tag: &str, fixtures: Fixtures, f: F)
where
    F: FnOnce(ArangoPool) -> Fut + Send + 'static,
    // `'static` is what `tokio::task::spawn` requires of the future below. It
    // compiles without it, since every caller passes an `async move` block that
    // already satisfies it, but stating it here puts a mismatch at the call
    // site rather than inside this function.
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let Some(socket) = std::env::var_os("ARANGO_SOCKET").filter(|v| !v.is_empty()) else {
        if strict() {
            panic!("ARANGO_TESTS requires ARANGO_SOCKET for an explicitly isolated test server");
        }
        eprintln!("skipping: ARANGO_SOCKET must name an explicitly isolated test server");
        return;
    };
    let socket = std::path::PathBuf::from(socket);
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
    // A leftover from a killed run carrying this exact name is ours; nothing
    // else can hold it.
    let _ = sys.delete(&format!("database/{db_name}")).await;
    if let Err(e) = sys.post("database", &json!({ "name": db_name })).await {
        // Not a skip. The previous harness swallowed every refusal, so a wrong
        // password made five tests report `ok` having run nothing. `_system`
        // rw is the grant this needs.
        if strict() {
            panic!(
                "ARANGO_TESTS is set but creating '{db_name}' as '{user}' failed \
                 (needs rw on _system): {e}"
            );
        }
        eprintln!("skipping: cannot create test database '{db_name}' as '{user}': {e}");
        return;
    }

    let client = ArangoClient::with_socket(socket, &db_name, &user, &password);
    let pool = ArangoPool::new(client.clone(), client);

    // `JoinHandle` captures a panic instead of unwinding through us, so the
    // database is dropped either way and the failure is re-raised after.
    //
    // **The fixtures are created inside the task, not before it.** Creating them
    // out here put one `expect` outside the only path that drops the database,
    // so a fixture that failed to create leaked the database it was meant to be
    // isolated in -- the failure this harness exists to prevent, in the setup of
    // the harness itself.
    let outcome = tokio::task::spawn(async move {
        if fixtures == Fixtures::Codebase {
            for (name, col_type) in CODEBASE.all_collections() {
                crud::create_collection(&pool, name, Some(col_type))
                    .await
                    .expect("create fixture collection");
            }
        }
        f(pool).await
    })
    .await;

    if let Err(e) = sys.delete(&format!("database/{db_name}")).await {
        // Reported rather than ignored: a leaked database is invisible
        // otherwise, and accumulates on a shared instance.
        eprintln!("warning: failed to drop test database '{db_name}': {e}");
    }

    if let Err(join_err) = outcome {
        std::panic::resume_unwind(join_err.into_panic());
    }
}
