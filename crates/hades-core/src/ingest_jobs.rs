//! Process-wide admission for detached ingestion. Database rows are history,
//! not an atomic lock: all endpoints in this process must share this registry.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

const MAX_CONCURRENT: usize = 2;
static ADMISSION: LazyLock<Arc<Admission>> =
    LazyLock::new(|| Arc::new(Admission::new(MAX_CONCURRENT)));

struct Admission {
    limit: usize,
    paths: Mutex<HashSet<PathBuf>>,
}

impl Admission {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            paths: Mutex::new(HashSet::new()),
        }
    }

    fn reserve(self: &Arc<Self>, canonical_path: &Path) -> Result<Reservation, String> {
        let mut paths = self
            .paths
            .lock()
            .map_err(|_| "ingest admission unavailable")?;
        // Ancestors and descendants also overlap. Database selection does not
        // make concurrent analyzers over the same source tree independent.
        if paths
            .iter()
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
        paths.insert(canonical_path.to_owned());
        Ok(Reservation {
            admission: Arc::clone(self),
            path: canonical_path.to_owned(),
        })
    }
}

/// Hold from before the first job-record await until child cleanup completes.
/// This type is intentionally neither Clone nor keyed by caller-supplied IDs.
pub(crate) struct Reservation {
    admission: Arc<Admission>,
    path: PathBuf,
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
}
