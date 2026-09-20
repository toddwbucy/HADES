//! Process-wide admission for detached ingestion. Database rows are history,
//! not an atomic lock: all endpoints in this process must share this registry.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use tokio_util::sync::CancellationToken;

pub(crate) mod process;
pub(crate) mod records;

#[cfg(test)]
pub(crate) static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

static INSTANCE: LazyLock<String> = LazyLock::new(|| format!("{:032x}", rand::random::<u128>()));
pub(crate) fn instance() -> &'static str {
    &INSTANCE
}
pub(crate) fn owns(database: &str, job: &str) -> bool {
    ADMISSION.paths.lock().is_ok_and(|paths| {
        paths.values().any(|identity| {
            identity
                .as_ref()
                .is_some_and(|(db, id)| db == database && id == job)
        })
    })
}

const MAX_CONCURRENT: usize = 2;
static ADMISSION: LazyLock<Arc<Admission>> =
    LazyLock::new(|| Arc::new(Admission::new(MAX_CONCURRENT)));

struct Admission {
    limit: usize,
    paths: Mutex<HashMap<PathBuf, Option<(String, String)>>>,
    shutdown: CancellationToken,
}

impl Admission {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            paths: Mutex::new(HashMap::new()),
            shutdown: CancellationToken::new(),
        }
    }

    fn reserve(self: &Arc<Self>, canonical_path: &Path) -> Result<Reservation, String> {
        let mut paths = self
            .paths
            .lock()
            .map_err(|_| "ingest admission unavailable")?;
        if self.shutdown.is_cancelled() {
            return Err("ingest service is shutting down".into());
        }
        // Ancestors and descendants also overlap. Database selection does not
        // make concurrent analyzers over the same source tree independent.
        if paths
            .keys()
            .any(|path| path.starts_with(canonical_path) || canonical_path.starts_with(path))
        {
            return Err("an ingest already owns this tree or an overlapping tree".into());
        }
        if paths.len() >= self.limit {
            return Err(format!(
                "{} ingest jobs are already admitted across this service; wait for one to finish",
                self.limit
            ));
        }
        paths.insert(canonical_path.to_owned(), None);
        Ok(Reservation {
            admission: Arc::clone(self),
            path: canonical_path.to_owned(),
        })
    }

    fn close(&self) {
        // Serialize cancellation with admission. Even on poison, cancellation
        // reaches existing owners and the registry continues failing closed.
        let _guard = self.paths.lock();
        self.shutdown.cancel();
    }

    async fn drain(&self) -> anyhow::Result<()> {
        loop {
            if self
                .paths
                .lock()
                .map_err(|_| anyhow::anyhow!("ingest registry poisoned during shutdown"))?
                .is_empty()
            {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
}

/// Hold from before the first job-record await until child cleanup completes.
/// This type is intentionally neither Clone nor keyed by caller-supplied IDs.
pub(crate) struct Reservation {
    admission: Arc<Admission>,
    path: PathBuf,
}

impl Reservation {
    pub(crate) fn identify(&self, database: &str, job: &str) -> Result<(), String> {
        let mut paths = self
            .admission
            .paths
            .lock()
            .map_err(|_| "ingest admission unavailable")?;
        let identity = paths
            .get_mut(&self.path)
            .ok_or("ingest reservation missing")?;
        *identity = Some((database.to_owned(), job.to_owned()));
        Ok(())
    }

    pub(crate) fn shutdown(&self) -> CancellationToken {
        self.admission.shutdown.clone()
    }
}

/// Stop new reservations and request cleanup of every owned ingestion process.
pub fn begin_shutdown() {
    ADMISSION.close();
}

/// Wait for the daemon's shutdown signal without creating another OS handler.
pub async fn shutdown_requested() {
    ADMISSION.shutdown.cancelled().await;
}

/// Keep the runtime alive through child cleanup and completion persistence.
pub async fn shutdown_and_wait() -> anyhow::Result<()> {
    begin_shutdown();
    ADMISSION.drain().await
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // A poisoned registry stays unavailable; do not reopen admission after
        // an invariant failure by recovering a potentially inconsistent set.
        if let Ok(mut paths) = self.admission.paths.lock() {
            paths.remove(&self.path);
        }
    }
}

pub(crate) fn reserve(canonical_path: &Path) -> Result<Reservation, String> {
    ADMISSION.reserve(canonical_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn concurrent_admission_is_atomic_and_reusable() {
        let admission = Arc::new(Admission::new(2));
        let barrier = Arc::new(Barrier::new(24));
        let reservations = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..24)
                .map(|index| {
                    let admission = Arc::clone(&admission);
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        // Reservations survive until all racing calls finish.
                        admission.reserve(Path::new(&format!("/fixture/tree-{index}")))
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|handle| handle.join().unwrap().ok())
                .collect::<Vec<_>>()
        });
        assert_eq!(reservations.len(), 2);
        assert!(
            admission
                .reserve(Path::new("/different-database/tree"))
                .is_err()
        );
        drop(reservations);
        assert!(admission.reserve(Path::new("/fixture/new-tree")).is_ok());
    }

    #[test]
    fn overlapping_trees_conflict_without_prefix_false_positives() {
        let admission = Arc::new(Admission::new(4));
        let owned = admission.reserve(Path::new("/fixture/tree")).unwrap();
        for path in ["/fixture/tree", "/fixture", "/fixture/tree/subdir"] {
            assert!(admission.reserve(Path::new(path)).is_err(), "{path}");
        }
        let sibling = admission.reserve(Path::new("/fixture/tree-two")).unwrap();
        drop(owned);
        assert!(admission.reserve(Path::new("/fixture/tree/subdir")).is_ok());
        drop(sibling);
    }

    #[tokio::test]
    async fn cancellation_before_spawn_releases_reservation() {
        let admission = Arc::new(Admission::new(1));
        let (ready, started) = tokio::sync::oneshot::channel();
        let owner = Arc::clone(&admission);
        let task = tokio::spawn(async move {
            let _reservation = owner.reserve(Path::new("/fixture/tree")).unwrap();
            ready.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        started.await.unwrap();
        assert!(admission.reserve(Path::new("/fixture/tree")).is_err());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(admission.reserve(Path::new("/fixture/tree")).is_ok());
    }

    #[test]
    fn poisoned_admission_fails_closed() {
        let admission = Arc::new(Admission::new(2));
        let broken = Arc::clone(&admission);
        let _ = std::thread::spawn(move || {
            let _lock = broken.paths.lock().unwrap();
            panic!("private fixture invariant failure");
        })
        .join();
        assert!(admission.reserve(Path::new("/fixture/tree")).is_err());
    }

    #[tokio::test]
    async fn shutdown_refuses_admission_and_waits_for_existing_owner() {
        let admission = Arc::new(Admission::new(2));
        let owned = admission.reserve(Path::new("/fixture/tree")).unwrap();
        let shutdown = owned.shutdown();
        admission.close();
        assert!(shutdown.is_cancelled());
        assert!(admission.reserve(Path::new("/fixture/other")).is_err());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(30), admission.drain())
                .await
                .is_err()
        );
        drop(owned);
        tokio::time::timeout(std::time::Duration::from_secs(1), admission.drain())
            .await
            .unwrap()
            .unwrap();
    }
}
