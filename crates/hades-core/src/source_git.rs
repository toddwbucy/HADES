//! One Git observation per repository per ingestion run (#171, #186).
//! This is an observation at admission/preparation, not an atomic tree snapshot.
use futures::{
    FutureExt,
    future::{BoxFuture, Shared},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
};
use tokio::sync::Mutex;

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("{stage}: {source}")]
    Io {
        stage: &'static str,
        source: Arc<std::io::Error>,
    },
    #[error("git executable not found on PATH")]
    MissingGit,
    #[error("cannot inspect source Git {0}")]
    Git(&'static str),
    #[error("source Git {0} is not valid UTF-8")]
    Encoding(&'static str),
    #[error("source Git worker failed: {0}")]
    Worker(Arc<tokio::task::JoinError>),
}
type Result<T> = std::result::Result<T, Error>;
fn io(stage: &'static str, source: std::io::Error) -> Error {
    Error::Io {
        stage,
        source: Arc::new(source),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceGit {
    pub commit: Option<String>,
    pub dirty: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Location {
    canonical: PathBuf,
    repository: Option<PathBuf>,
}

/// Only observations, never credentials or subprocess state, cross the sealed
/// admission-to-child handoff. Separate repositories keep separate identities.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    locations: Vec<(PathBuf, Location)>,
    repositories: Vec<(PathBuf, SourceGit)>,
    checkpoint: Option<PathBuf>,
}
type Pending<T> = Shared<BoxFuture<'static, Result<T>>>;
fn worker<T: Clone + Send + Sync + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Pending<T> {
    // Start once before sharing the future. Cancelling a waiter cannot cause
    // another git status: the running job and its result remain in this batch.
    tokio::task::spawn_blocking(work)
        .map(|r| r.map_err(|e| Error::Worker(Arc::new(e))).and_then(|r| r))
        .boxed()
        .shared()
}
fn ready<T: Clone + Send + Sync + 'static>(value: T) -> Pending<T> {
    futures::future::ready(Ok(value)).boxed().shared()
}

pub struct Batch {
    locations: Mutex<HashMap<PathBuf, Pending<Location>>>,
    repositories: Mutex<HashMap<PathBuf, Pending<SourceGit>>>,
    checkpoint: Option<PathBuf>,
}
impl Default for Batch {
    fn default() -> Self {
        Self::from_snapshot(Snapshot {
            checkpoint: std::env::current_dir()
                .ok()
                .map(|p| p.join(".hades-batch-state.json")),
            ..Snapshot::default()
        })
    }
}
impl Batch {
    pub fn from_snapshot(snapshot: Snapshot) -> Self {
        Self {
            locations: Mutex::new(
                snapshot
                    .locations
                    .into_iter()
                    .map(|(k, v)| (k, ready(v)))
                    .collect(),
            ),
            repositories: Mutex::new(
                snapshot
                    .repositories
                    .into_iter()
                    .map(|(k, v)| (k, ready(v)))
                    .collect(),
            ),
            checkpoint: snapshot.checkpoint,
        }
    }
    async fn location(&self, input: &Path) -> Result<Location> {
        let pending = {
            let mut locations = self.locations.lock().await;
            locations
                .entry(input.to_owned())
                .or_insert_with(|| {
                    let input = input.to_owned();
                    worker(move || locate(&input))
                })
                .clone()
        };
        let location = pending.clone().await?;
        self.locations
            .lock()
            .await
            .entry(location.canonical.clone())
            .or_insert(pending);
        Ok(location)
    }
    pub async fn canonical(&self, input: &Path) -> Result<PathBuf> {
        Ok(self.location(input).await?.canonical)
    }
    pub async fn resolve(&self, input: &Path) -> Result<Option<SourceGit>> {
        let Some(repository) = self.location(input).await?.repository else {
            return Ok(None);
        };
        let pending = {
            let mut repositories = self.repositories.lock().await;
            repositories
                .entry(repository.clone())
                .or_insert_with(|| {
                    let checkpoint = self.checkpoint.clone();
                    worker(move || observe(&repository, checkpoint.as_deref()))
                })
                .clone()
        };
        pending.await.map(Some)
    }
    /// Observe all named inputs before batch checkpoints or phase writes. A bad
    /// input stays a per-item failure; its cached error is reported by that item.
    pub async fn prepare(&self, inputs: &[PathBuf]) {
        for input in inputs {
            let _ = self.resolve(input).await;
        }
    }
    /// A mixed-repository envelope is populated only when its inputs agree.
    pub async fn common(&self, inputs: &[PathBuf]) -> Result<Option<SourceGit>> {
        let mut common = None;
        for input in inputs {
            let observed = self.resolve(input).await?;
            match &common {
                None => common = Some(observed),
                Some(previous) if previous != &observed => return Ok(None),
                _ => {}
            }
        }
        Ok(common.flatten())
    }

    pub async fn snapshot(&self) -> Result<Snapshot> {
        let locations: Vec<_> = self
            .locations
            .lock()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let repositories: Vec<_> = self
            .repositories
            .lock()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut snapshot = Snapshot {
            checkpoint: self.checkpoint.clone(),
            ..Snapshot::default()
        };
        for (path, pending) in locations {
            snapshot.locations.push((path, pending.await?));
        }
        for (path, pending) in repositories {
            snapshot.repositories.push((path, pending.await?));
        }
        Ok(snapshot)
    }
}
fn git(directory: &Path, args: &[&str], extra: Option<OsString>) -> Result<Output> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(directory)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("LC_ALL", "C");
    if let Some(extra) = extra {
        command.arg(extra);
    }
    command.output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::MissingGit
        } else {
            io("cannot run git to inspect source state", e)
        }
    })
}
fn locate(input: &Path) -> Result<Location> {
    let canonical = input
        .canonicalize()
        .map_err(|e| io("cannot resolve provenance input", e))?;
    let metadata = canonical
        .metadata()
        .map_err(|e| io("cannot inspect provenance input", e))?;
    let directory = if metadata.is_file() {
        canonical.parent().expect("canonical file has a parent")
    } else {
        &canonical
    };
    let top = git(directory, &["rev-parse", "--show-toplevel"], None)?;
    let repository = if top.status.success() {
        let top = String::from_utf8(top.stdout).map_err(|_| Error::Encoding("repository path"))?;
        Some(
            PathBuf::from(top.strip_suffix('\n').unwrap_or(&top))
                .canonicalize()
                .map_err(|e| io("cannot resolve repository top-level", e))?,
        )
    } else if String::from_utf8_lossy(&top.stderr).contains("not a git repository")
        || String::from_utf8_lossy(&top.stderr).contains("must be run in a work tree")
    {
        None
    } else {
        return Err(Error::Git("worktree"));
    };
    Ok(Location {
        canonical,
        repository,
    })
}
fn observe(repository: &Path, checkpoint: Option<&Path>) -> Result<SourceGit> {
    let head = git(repository, &["rev-parse", "--verify", "HEAD"], None)?;
    let commit = if head.status.success() {
        Some(
            String::from_utf8(head.stdout)
                .map_err(|_| Error::Encoding("HEAD"))?
                .trim()
                .to_owned(),
        )
    } else {
        if !git(repository, &["symbolic-ref", "-q", "HEAD"], None)?
            .status
            .success()
        {
            return Err(Error::Git("HEAD"));
        }
        None
    };
    // Exclude only this run's managed checkpoint, not similarly named files in
    // other directories. Checkpoint placement and --resume remain compatible.
    let excluded = checkpoint
        .and_then(|p| p.strip_prefix(repository).ok())
        .map(|relative| {
            let mut spec = OsString::from(":(top,literal,exclude)");
            spec.push(relative);
            spec
        });
    let status = git(
        repository,
        &[
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            ".",
        ],
        excluded,
    )?;
    if !status.status.success() {
        return Err(Error::Git("dirty state"));
    }
    Ok(SourceGit {
        commit,
        dirty: !status.stdout.is_empty(),
    })
}

/// One-shot synchronous observation for synchronous callers. Async ingestion
/// uses Batch; no Git subprocess runs on its async worker threads.
pub fn resolve(root: &Path) -> Result<Option<SourceGit>> {
    locate(root)?
        .repository
        .map(|repo| observe(&repo, None))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn committed_fixture(root: &Path) {
        let git = |args: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(["-c", "commit.gpgsign=false"])
                .args(args)
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
        };
        git(&["init", "-q"]);
        std::fs::write(root.join("source.txt"), root.to_string_lossy().as_bytes()).unwrap();
        git(&["add", "."]);
        git(&[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ]);
    }

    #[tokio::test]
    async fn observations_are_per_run_and_survive_snapshot_serialization() {
        let root = tempfile::tempdir().unwrap();
        committed_fixture(root.path());
        let batch = Batch::default();
        let original = batch.resolve(root.path()).await.unwrap().unwrap();
        assert!(!original.dirty);
        let snapshot = batch.snapshot().await.unwrap();
        // JSON is the exact representation carried by the sealed handoff.
        let snapshot: Snapshot =
            serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
        std::fs::write(root.path().join("source.txt"), "changed after observation").unwrap();
        let restored = Batch::from_snapshot(snapshot);
        assert_eq!(
            batch.resolve(root.path()).await.unwrap(),
            Some(original.clone())
        );
        assert_eq!(
            restored
                .resolve(&root.path().join("source.txt"))
                .await
                .unwrap(),
            Some(original)
        );
        assert!(
            Batch::default()
                .resolve(root.path())
                .await
                .unwrap()
                .unwrap()
                .dirty
        );
    }

    #[tokio::test]
    async fn only_the_managed_checkpoint_is_excluded_and_nested_repos_are_distinct() {
        let root = tempfile::tempdir().unwrap();
        committed_fixture(root.path());
        let checkpoint = root.path().join(".hades-batch-state.json");
        std::fs::write(&checkpoint, "private state").unwrap();
        let make_batch = || {
            Batch::from_snapshot(Snapshot {
                checkpoint: Some(checkpoint.clone()),
                ..Snapshot::default()
            })
        };
        assert!(
            !make_batch()
                .resolve(root.path())
                .await
                .unwrap()
                .unwrap()
                .dirty
        );
        let nested = root.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        committed_fixture(&nested);
        std::fs::write(nested.join(".hades-batch-state.json"), "unrelated state").unwrap();
        let batch = make_batch();
        let parent = batch
            .resolve(&root.path().join("source.txt"))
            .await
            .unwrap()
            .unwrap();
        let child = batch
            .resolve(&nested.join("source.txt"))
            .await
            .unwrap()
            .unwrap();
        assert!(parent.dirty && child.dirty);
        assert_ne!(parent.commit, child.commit);
        assert_eq!(
            batch
                .common(&[root.path().join("source.txt"), nested.join("source.txt")])
                .await
                .unwrap(),
            None
        );
        let non_git = tempfile::tempdir().unwrap();
        assert_eq!(batch.resolve(non_git.path()).await.unwrap(), None);
    }

    #[test]
    fn non_git_observation_forces_c_locale() {
        use std::os::unix::fs::PermissionsExt;
        const CHILD: &str = "HADES_GIT_LOCALE_TEST_CHILD";
        if let Some(root) = std::env::var_os(CHILD) {
            assert_eq!(resolve(Path::new(&root)).unwrap(), None);
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let git = root.path().join("git");
        std::fs::write(&git, "#!/bin/sh\nif [ \"$LC_ALL\" = C ]; then echo 'fatal: not a git repository' >&2; else echo 'fatal: kein Git-Repository' >&2; fi\nexit 128\n").unwrap();
        std::fs::set_permissions(&git, std::fs::Permissions::from_mode(0o700)).unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "source_git::tests::non_git_observation_forces_c_locale",
                "--nocapture",
            ])
            .env(CHILD, root.path())
            .env("PATH", root.path())
            .env("LANG", "de_DE.UTF-8")
            .env("LC_ALL", "de_DE.UTF-8")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
    }

    #[test]
    fn observes_committed_dirty_and_non_git_inputs() {
        let tree = tempfile::tempdir().unwrap();
        assert_eq!(resolve(tree.path()).unwrap(), None);
        let run = |args: &[&str]| {
            let output = Command::new("git")
                .arg("-C")
                .arg(tree.path())
                .args(args)
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
            output
        };
        run(&["init", "-q"]);
        std::fs::write(tree.path().join("a.txt"), "initial").unwrap();
        run(&["add", "."]);
        run(&[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ]);
        let expected = String::from_utf8(run(&["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim()
            .to_owned();
        assert_eq!(
            resolve(tree.path()).unwrap(),
            Some(SourceGit {
                commit: Some(expected.clone()),
                dirty: false
            })
        );
        std::fs::write(tree.path().join("untracked.txt"), "new").unwrap();
        assert_eq!(
            resolve(&tree.path().join("a.txt")).unwrap(),
            Some(SourceGit {
                commit: Some(expected),
                dirty: true
            })
        );
    }
}
