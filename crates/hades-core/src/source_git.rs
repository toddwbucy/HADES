//! Git provenance observations for ingestion (#171), not an atomic tree snapshot.
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    path::Path,
    process::{Command, Output},
};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SourceGit {
    pub commit: Option<String>,
    pub dirty: bool,
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
            .output()
            .context("cannot inspect source Git state")
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
