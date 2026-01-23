"""Local Filesystem Storage Backend.

Implements StorageBase with filesystem semantics for workflow outputs.
"""

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from .storage_base import StorageBase

logger = logging.getLogger(__name__)


class LocalStorage(StorageBase):
    """
    Filesystem-based storage backend implementing StorageBase.

    Stores workflow outputs as JSON files in a directory structure.
    Supports atomic writes via temp file + rename pattern.
    """

    def __init__(self, base_path: str | Path, config: dict[str, Any] | None = None):
        """
        Initialize local filesystem storage.

        Args:
            base_path: Root directory for storage
            config: Optional configuration
        """
        super().__init__(config)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a storage key to a file path using percent-encoding."""
        # Percent-encode key to create safe, reversible filename
        # safe='' encodes all special characters including path separators
        safe_key = quote(key, safe="")
        return self.base_path / f"{safe_key}.json"

    def _path_to_key(self, file_path: Path) -> str:
        """Convert a file path back to the original storage key."""
        # Decode percent-encoded filename back to original key
        return unquote(file_path.stem)

    def store(self, key: str, data: Any, metadata: dict[str, Any] | None = None) -> bool:
        """
        Store data with the given key.

        Args:
            key: Storage key/identifier
            data: Data to store (must be JSON-serializable)
            metadata: Optional metadata (stored alongside data)

        Returns:
            True if successful, False otherwise
        """
        file_path = self._key_to_path(key)
        temp_path = file_path.with_suffix(".tmp")

        try:
            payload = {"data": data}
            if metadata is not None:
                payload["metadata"] = metadata

            # Atomic write: temp file + rename
            with open(temp_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

            temp_path.replace(file_path)
            logger.debug(f"Stored key '{key}' to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to store key '{key}': {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return False

    def retrieve(self, key: str) -> Any | None:
        """
        Retrieve data for the given key.

        Args:
            key: Storage key/identifier

        Returns:
            Stored data or None if not found
        """
        file_path = self._key_to_path(key)

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                payload = json.load(f)
            return payload.get("data")
        except Exception as e:
            logger.error(f"Failed to retrieve key '{key}': {e}")
            return None

    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage.

        Args:
            key: Storage key/identifier

        Returns:
            True if exists, False otherwise
        """
        return self._key_to_path(key).exists()

    def delete(self, key: str) -> bool:
        """
        Delete data for the given key.

        Args:
            key: Storage key/identifier

        Returns:
            True if successful, False otherwise
        """
        file_path = self._key_to_path(key)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            logger.debug(f"Deleted key '{key}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            return False

    def list_keys(self, prefix: str | None = None) -> list[str]:
        """
        List all keys in storage.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """
        keys = []
        for file_path in self.base_path.glob("*.json"):
            # Decode percent-encoded filename back to original key
            key = self._path_to_key(file_path)
            if prefix is None or key.startswith(prefix):
                keys.append(key)
        return sorted(keys)

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """
        Retrieve metadata for the given key.

        Args:
            key: Storage key/identifier

        Returns:
            Metadata dict or None if not found
        """
        file_path = self._key_to_path(key)

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                payload = json.load(f)
            return payload.get("metadata")
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for key '{key}': {e}")
            return None

    @property
    def storage_type(self) -> str:
        """Get the type of storage backend."""
        return "local_filesystem"

    @property
    def supports_metadata(self) -> bool:
        """Whether this backend supports metadata."""
        return True
