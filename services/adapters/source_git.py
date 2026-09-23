"""Git provenance for adapter inputs (#171); observations, not tree snapshots."""
import os
import subprocess
from pathlib import Path


def resolve(root: Path):
    root = root.resolve(strict=True)
    directory = root.parent if root.is_file() else root
    env = {k: v for k, v in os.environ.items() if k not in
           {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    # The non-repository diagnostic is parsed below (#171).
    env["LC_ALL"] = "C"
    def git(*args):
        return subprocess.run(["git", "-C", str(directory), *args],
                              env=env, capture_output=True, check=False)
    inside = git("rev-parse", "--is-inside-work-tree")
    if inside.returncode:
        if b"not a git repository" in inside.stderr:
            return None
        raise ValueError("cannot inspect source Git worktree")
    if inside.stdout != b"true\n":
        return None
    head = git("rev-parse", "--verify", "HEAD")
    if head.returncode and git("symbolic-ref", "-q", "HEAD").returncode:
        raise ValueError("cannot resolve source Git HEAD")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none")
    if status.returncode:
        raise ValueError("cannot inspect source Git dirty state")
    return {"commit": None if head.returncode else head.stdout.decode().strip(),
            "dirty": bool(status.stdout)}
