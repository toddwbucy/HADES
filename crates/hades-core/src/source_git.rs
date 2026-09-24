//! Git provenance observations for ingestion (#171), not an atomic tree snapshot.
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Mutex,
};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SourceGit {
    pub commit: Option<String>,
    pub dirty: bool,
}

/// Observations are shared only within one batch, including non-Git directories.
/// Canonical directory keys coalesce aliases; nested repositories remain distinct (#171).
#[derive(Default)]
pub struct Batch {
    observations: Mutex<HashMap<PathBuf, Option<SourceGit>>>,
}

impl Batch {
    pub fn resolve(&self, input: &Path) -> Result<Option<SourceGit>> {
        let input = input
            .canonicalize()
            .context("cannot resolve provenance input")?;
        let directory = if input.is_file() {
            input.parent().unwrap()
        } else {
            &input
        };
        // Hold the lock through resolution so concurrent named files cannot all
        // miss the cache and launch N git status processes for one directory (#171).
        let mut observations = self
            .observations
            .lock()
            .map_err(|_| anyhow!("source Git observation lock poisoned"))?;
        if let Some(observed) = observations.get(directory) {
            return Ok(observed.clone());
        }
        let observed = resolve(directory)?;
        observations.insert(directory.to_owned(), observed.clone());
        Ok(observed)
    }
}

/// Non-Git inputs serialize as null; failures inside a repository are errors.
/// Dirty covers the whole worktree, including untracked files and submodules.
pub fn resolve(root: &Path) -> Result<Option<SourceGit>> {
    let root = root
        .canonicalize()
        .context("cannot resolve provenance input")?;
    let directory = if root.is_file() {
        root.parent().unwrap()
    } else {
        &root
    };
    let git = |args: &[&str]| -> Result<Output> {
        Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env("GIT_OPTIONAL_LOCKS", "0")
            // The non-repository diagnostic is parsed below (#171).
            .env("LC_ALL", "C")
            .output()
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    anyhow!("git executable not found on PATH")
                } else {
                    anyhow!(error).context("cannot run git to inspect source state")
                }
            })
    };
    let inside = git(&["rev-parse", "--is-inside-work-tree"])?;
    if !inside.status.success() {
        if String::from_utf8_lossy(&inside.stderr).contains("not a git repository") {
            return Ok(None);
        }
        bail!("cannot inspect source Git worktree");
    }
    if inside.stdout != b"true\n" {
        return Ok(None);
    }
    let head = git(&["rev-parse", "--verify", "HEAD"])?;
    let commit = if head.status.success() {
        Some(String::from_utf8(head.stdout)?.trim().to_owned())
    } else {
        // An unborn branch has no commit, but still has an observable dirty state.
        let unborn = git(&["symbolic-ref", "-q", "HEAD"])?;
        if !unborn.status.success() {
            bail!("cannot resolve source Git HEAD");
        }
        None
    };
    let status = git(&[
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ])?;
    if !status.status.success() {
        bail!("cannot inspect source Git dirty state");
    }
    Ok(Some(SourceGit {
        commit,
        dirty: !status.stdout.is_empty(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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
