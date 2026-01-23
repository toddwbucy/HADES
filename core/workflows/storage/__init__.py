"""
Storage Backends for Workflows.

Provides storage backends for workflow outputs including
local filesystem and ArangoDB utilities.
"""

from .storage_base import StorageBase
from .storage_local import LocalStorage

# Lazy imports for optional backends
_lazy_imports = {
    "ArangoStorageManager": (".storage_arango", "ArangoStorageManager"),
    "S3Storage": (".storage_s3", "S3Storage"),
    "RamFSStorage": (".storage_ramfs", "RamFSStorage"),
}


def __getattr__(name: str):
    """Lazy import optional storage backends."""
    if name in _lazy_imports:
        module_path, class_name = _lazy_imports[name]
        try:
            import importlib
            module = importlib.import_module(module_path, package=__name__)
            return getattr(module, class_name)
        except ImportError as e:
            if name == "ArangoStorageManager":
                raise ImportError(
                    "ArangoStorageManager requires python-arango. "
                    "Install with: pip install python-arango"
                ) from e
            elif name == "S3Storage":
                raise ImportError(
                    "S3Storage requires boto3. Install with: pip install boto3"
                ) from e
            elif name == "RamFSStorage":
                raise ImportError(
                    "RamFSStorage requires POSIX shared memory support."
                ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StorageBase",
    "LocalStorage",
    "ArangoStorageManager",
    "S3Storage",
    "RamFSStorage",
]
