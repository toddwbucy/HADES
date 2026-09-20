"""Require an explicit, unmodified checkout of the historical audit revision."""
import os
from pathlib import Path
import subprocess

REVISION = "a71d73bfe988e17d487a2db1d35dadf3a18f0664"


def source_root(value=None):
    value = value or os.environ.get("HADES_AUDIT_SOURCE")
    if not value:
        raise RuntimeError("Set HADES_AUDIT_SOURCE to a separate historical worktree")
    root = Path(value).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Historical probes require {REVISION}; found {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    if status:
        raise RuntimeError("Historical checkout has tracked changes or untracked files")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--"],
                   check=True, stdout=subprocess.DEVNULL)
    return root


if __name__ == "__main__":
    print(source_root())
