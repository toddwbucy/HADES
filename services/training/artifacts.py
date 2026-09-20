"""Publish complete training artifacts without truncating an existing version."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def publish_artifact(destination):
    """Yield a private staging stream; replace destination only on success.

    The destination directory must exist. File contents are fsynced, but the
    directory is not: this is atomic visibility, not a power-loss guarantee.
    """
    destination = Path(destination)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=".hades-artifact-",
                                         suffix=".tmp", dir=destination.parent,
                                         delete=False) as stream:
            staged = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
