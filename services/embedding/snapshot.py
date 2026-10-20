"""Private read-only artifact copies with mutation detection at serving boundaries.

These snapshots protect against source/cache changes and accidental modification
of a private serving copy. They do not sandbox trusted model code or defend
against a hostile process controlling the service account or kernel.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .identity import ArtifactFile, ArtifactManifest, MAX_MANIFEST_FILES, _path

DEFAULT_MAX_BYTES = 32 * 1024**3


class ArtifactChangedError(ValueError):
    """An artifact could not be copied consistently or a serving snapshot changed."""


def _stamp(info: os.stat_result) -> tuple:
    """Track identity and mutation metadata, excluding read-induced access times."""
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def remove_private_tree(root: Path) -> None:
    """Remove an owned temporary tree whose directories were made read-only."""
    try:
        info = root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        root.unlink()
        return
    for directory, _, _ in os.walk(root, followlinks=False):
        os.chmod(directory, 0o700)
    shutil.rmtree(root)


def copy_assets(source: Path, destination: Path, paths: list[str],
                max_bytes: int = DEFAULT_MAX_BYTES) -> ArtifactManifest:
    """Copy selected regular assets and hash their copied bytes, never their names.

    Source file symlinks may resolve to HF cache blobs. Destination files are
    independent regular files, not hard links. Reject a source that changes
    during its read and enforce a total byte budget before publishing anything.
    The caller owns cleanup of the private destination on failure.
    """
    if not 0 < len(paths) <= MAX_MANIFEST_FILES or max_bytes <= 0:
        raise ValueError("invalid asset count or snapshot byte budget")
    files = []
    total = 0
    for logical in sorted(paths, key=lambda path: path.encode("utf-8")):
        _path(logical)
        resolved = (source / logical).resolve(strict=True)
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as incoming:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("model assets must resolve to regular files")
            if total + before.st_size > max_bytes:
                raise ValueError("model snapshot exceeds configured byte budget")
            target = destination / logical
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = hashlib.md5(usedforsecurity=False)
            count = 0
            with target.open("xb") as outgoing:
                while block := incoming.read(1024 * 1024):
                    count += len(block)
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("model snapshot exceeds configured byte budget")
                    outgoing.write(block)
                    digest.update(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if _stamp(before) != _stamp(os.fstat(incoming.fileno())) or count != before.st_size:
                raise ArtifactChangedError("source asset changed during snapshot preparation")
            files.append(ArtifactFile(logical, count, digest.hexdigest()))
    return ArtifactManifest(tuple(files))


def freeze_tree(root: Path) -> None:
    """Remove write permissions from an owned copy before capturing its metadata."""
    for directory, _, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("private snapshots cannot contain links or special files")
            os.chmod(path, 0o400)
        os.chmod(directory, 0o500)


class ServingSnapshot:
    """Own one verified load's private files until its model has been unloaded."""

    def __init__(self, root: Path, manifest: ArtifactManifest) -> None:
        """Capture a completed read-only tree; callers normally use prepare()."""
        self.root = root
        self.manifest = manifest
        self._closed = False
        self._baseline = self._inventory()
        expected = {file.path for file in manifest.files}
        actual = {path for path, stamp in self._baseline.items() if stat.S_ISREG(stamp[2])}
        if actual != expected:
            raise ArtifactChangedError("snapshot asset set differs from registered manifest")

    @classmethod
    def prepare(cls, source: Path, manifest: ArtifactManifest,
                parent: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> ServingSnapshot:
        """Copy and rehash a trusted registration before exposing a loading path."""
        root = Path(tempfile.mkdtemp(prefix="model_", dir=parent))
        try:
            copied = copy_assets(source, root, [file.path for file in manifest.files], max_bytes)
            copied.check_digest(manifest.digest())
            freeze_tree(root)
            return cls(root, copied)
        except BaseException:
            remove_private_tree(root)
            raise

    def _inventory(self) -> dict:
        """Inspect every entry without following links or reading model weights again."""
        entries = {}
        pending = [self.root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            if info.st_uid != os.geteuid() or info.st_mode & 0o222:
                raise ArtifactChangedError("serving snapshot ownership or permissions changed")
            logical = str(path.relative_to(self.root))
            entries[logical] = _stamp(info)
            if stat.S_ISDIR(info.st_mode):
                pending.extend(path.iterdir())
            elif not stat.S_ISREG(info.st_mode):
                raise ArtifactChangedError("serving snapshot contains a link or special file")
        return entries

    def check(self) -> None:
        """Reject mutation, replacement, insertion, or removal since this load's seal."""
        if self._closed:
            raise ArtifactChangedError("serving snapshot is closed")
        try:
            if self._inventory() != self._baseline:
                raise ArtifactChangedError("serving snapshot changed; reload from registration")
        except OSError as error:
            raise ArtifactChangedError("serving snapshot is no longer readable") from error

    def close(self) -> None:
        """Release the private copy after callers have stopped using its model."""
        if not self._closed:
            self._closed = True
            remove_private_tree(self.root)
