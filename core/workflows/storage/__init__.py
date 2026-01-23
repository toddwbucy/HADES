"""
Storage Backends for Workflows

Provides different storage backends for workflow outputs including
local filesystem, S3, and RamFS for high-speed staging.
"""

try:
    from .storage_local import StorageManager as LocalStorage
except ImportError:
    LocalStorage = None  # type: ignore[misc]

try:
    from .storage_s3 import S3Storage
except ImportError:
    S3Storage = None

try:
    from .storage_ramfs import RamFSStorage
except ImportError:
    RamFSStorage = None

# Import base class
from .storage_base import StorageBase


def __getattr__(name: str):
    if name == "LocalStorage" and LocalStorage is None:
        raise ImportError(
            "LocalStorage backend is unavailable. Install optional dependencies or select a supported backend."
        )
    if name == "S3Storage" and S3Storage is None:
        raise ImportError("S3Storage backend is unavailable. Install boto3 to enable it.")
    if name == "RamFSStorage" and RamFSStorage is None:
        raise ImportError("RamFSStorage backend is unavailable. POSIX shared memory support missing.")
    raise AttributeError(name)

__all__ = [
    'LocalStorage',
    'RamFSStorage',
    'S3Storage',
    'StorageBase',
]
