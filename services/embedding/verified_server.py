"""Entry point for the verified provider service on a private Unix socket."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from .verified_jina import VerifiedJinaProvider
from .verified_late import VerifiedJinaLateProvider

# One process serves exactly one embedding space (#225). The late provider is a distinct
# space, not a superset: it rewrites the pooling identity, refuses conventional planning
# and embeds only queries, so the operator names the space rather than the service
# guessing it.
SPACES: dict[str, type[VerifiedJinaProvider]] = {
    "conventional": VerifiedJinaProvider,
    "late": VerifiedJinaLateProvider,
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Require every selector explicitly; no space or socket is chosen by default."""
    parser = argparse.ArgumentParser(prog="python -m embedding.verified_server",
                                     description=__doc__)
    parser.add_argument("--registry", required=True, type=Path,
                        help="model registry root holding the registered provider")
    parser.add_argument("--provider", required=True, help="registered provider ID")
    parser.add_argument("--space", required=True, choices=sorted(SPACES),
                        help="the one embedding space this process serves")
    parser.add_argument("--uds", required=True, type=Path,
                        help="absolute Unix socket path inside a private 0700 directory")
    return parser.parse_args(argv)


def check_socket_path(uds: Path) -> Path:
    """Refuse a socket location the provider catalog would refuse later.

    This mirrors the Rust catalog's socket checks (persephone/provider_registry/mod.rs,
    in `describe`) so the operator learns about an unsafe parent at service
    start, not at the first /v1/verify from HADES (#225). lstat is used throughout:
    a symlinked ancestor would let another path's owner redirect the socket.
    """
    if not uds.is_absolute():
        raise ValueError("socket path must be absolute")
    owner = os.geteuid()
    parent = uds.parent
    info = os.lstat(parent)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"socket parent must not be a symlink: {parent}")
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_mode & 0o7777 != 0o700:
        raise ValueError(f"socket parent must be a 0700 directory owned by this account: {parent}")
    for ancestor in parent.parents:
        info = os.lstat(ancestor)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"socket ancestors must not be symlinks: {ancestor}")
        if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner):
            raise ValueError("socket ancestors must be directories owned by this account "
                             f"or root: {ancestor}")
        shared = info.st_mode & 0o022
        if shared and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX):
            raise ValueError("socket ancestors must not be writable by other accounts "
                             f"outside root-owned sticky directories: {ancestor}")
    try:
        existing = os.lstat(uds)
    except FileNotFoundError:
        return uds
    if not stat.S_ISSOCK(existing.st_mode):
        raise ValueError("socket path is occupied by a non-socket file")
    # A socket left by a crashed run cannot be reused by bind; the private parent
    # guarantees it was ours, so removing it is safe.
    os.unlink(uds)
    return uds
