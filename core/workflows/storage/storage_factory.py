#!/usr/bin/env python3
"""Storage Factory with Registry Pattern.

Factory pattern for creating storage backends using a registry.
Supports configuration-based selection and auto-detection.

Usage:
    # Create storage backend
    storage = StorageFactory.create("local", base_path="/data")

    # List available backends
    StorageFactory.list_available()

    # Register custom backend
    @StorageFactory.register("custom")
    class CustomStorage(StorageBase):
        ...
"""

import logging
from typing import Any, ClassVar

from .storage_base import StorageBase

logger = logging.getLogger(__name__)


class StorageFactory:
    """Factory for creating storage backends using registry pattern.

    Manages storage backend instantiation with support for
    lazy loading and configuration-based selection.

    Example:
        # Create local storage
        storage = StorageFactory.create("local", base_path="/data")

        # Create ArangoDB storage
        storage = StorageFactory.create("arango", database="my_db")

        # List available backends
        available = StorageFactory.list_available()

        # Register custom backend
        @StorageFactory.register("mybackend")
        class MyStorage(StorageBase):
            ...
    """

    # Registry of available storage backends
    _registry: ClassVar[dict[str, type[StorageBase]]] = {}

    # Track which backends have been auto-registered
    _auto_registered: ClassVar[set[str]] = set()

    @classmethod
    def register(cls, name: str):
        """Decorator to register a storage backend class.

        Args:
            name: Name to register under (e.g., "local", "arango")

        Returns:
            Decorator function

        Example:
            @StorageFactory.register("mybackend")
            class MyStorage(StorageBase):
                ...
        """
        def decorator(storage_class: type[StorageBase]) -> type[StorageBase]:
            cls._registry[name] = storage_class
            logger.debug(f"Registered storage backend: {name}")
            return storage_class
        return decorator

    @classmethod
    def create(
        cls,
        storage_type: str,
        config: dict[str, Any] | None = None,
        **kwargs,
    ) -> StorageBase:
        """Create a storage backend instance.

        Args:
            storage_type: Type of storage ("local", "arango")
            config: Optional configuration dictionary
            **kwargs: Additional arguments for the backend

        Returns:
            Storage backend instance

        Raises:
            ValueError: If no backend registered for storage_type
        """
        # Try auto-registration if not already registered
        if storage_type not in cls._registry:
            cls._auto_register(storage_type)

        if storage_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"No storage backend registered for '{storage_type}'. "
                f"Available: {available}"
            )

        storage_class = cls._registry[storage_type]
        logger.info(f"Creating {storage_type} storage backend")

        # Merge config dict with kwargs
        if config is not None:
            merged_config = {**config, **kwargs}
        else:
            merged_config = kwargs

        return storage_class(config=merged_config)

    @classmethod
    def _auto_register(cls, storage_type: str) -> None:
        """Auto-register a storage backend on first use.

        Enables lazy loading of storage modules.

        Args:
            storage_type: Type of storage to register

        Note:
            Only backends implementing StorageBase are registered here.
            ArangoStorageManager is a connection manager, not a StorageBase
            implementation - use DatabaseFactory for database connections.
        """
        if storage_type in cls._auto_registered:
            return

        cls._auto_registered.add(storage_type)

        try:
            if storage_type == "local":
                from .storage_local import LocalStorage

                # Wrap LocalStorage to handle config-based initialization
                cls._registry["local"] = _LocalStorageWrapper
                logger.debug("Auto-registered local storage backend")

            # Note: ArangoStorageManager doesn't implement StorageBase
            # It's a connection manager. Use DatabaseFactory for ArangoDB connections.
            # Future: Could create an ArangoStorage(StorageBase) wrapper

            else:
                logger.warning(f"Unknown storage type for auto-registration: {storage_type}")

        except ImportError as e:
            logger.debug(f"Could not auto-register {storage_type}: {e}")

    @classmethod
    def _ensure_registered(cls) -> None:
        """Ensure all built-in backends are registered."""
        for storage_type in ["local"]:
            if storage_type not in cls._registry:
                cls._auto_register(storage_type)

    @classmethod
    def list_available(cls) -> dict[str, dict[str, Any]]:
        """List available storage backends.

        Returns:
            Dictionary mapping backend names to their info
        """
        cls._ensure_registered()

        available = {}
        for name, storage_class in cls._registry.items():
            try:
                info: dict[str, Any] = {
                    "class": storage_class.__name__,
                    "module": storage_class.__module__,
                }

                # Try to get properties without config
                try:
                    temp = storage_class()
                    info["storage_type"] = temp.storage_type
                    info["supports_metadata"] = temp.supports_metadata
                    info["supports_streaming"] = temp.supports_streaming
                except Exception:
                    pass

                available[name] = info

            except Exception as e:
                available[name] = {"error": str(e)}

        return available

    @classmethod
    def get_for_config(cls, config: dict[str, Any]) -> StorageBase:
        """Create storage backend from configuration dictionary.

        Convenience method that reads the storage type from config.

        Args:
            config: Configuration with 'type' key

        Returns:
            Storage backend instance

        Raises:
            ValueError: If 'type' not in config or invalid type
        """
        storage_type = config.get("type")
        if not storage_type:
            raise ValueError("Configuration must include 'type' key")

        # Remove 'type' from config before passing to create
        config_copy = {k: v for k, v in config.items() if k != "type"}
        return cls.create(storage_type, config=config_copy)


# =============================================================================
# Backend Wrapper Classes
# =============================================================================


class _LocalStorageWrapper(StorageBase):
    """Wrapper for LocalStorage that handles config-based initialization."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        """Initialize LocalStorage with config or kwargs.

        Args:
            config: Configuration dictionary (may contain 'base_path')
            **kwargs: Additional arguments (may contain 'base_path')
        """
        from .storage_local import LocalStorage

        # Merge config and kwargs
        merged = {**(config or {}), **kwargs}

        # Extract base_path
        base_path = merged.pop("base_path", None)
        if base_path is None:
            base_path = "/tmp/hades-storage"  # Default path

        # Create underlying LocalStorage
        self._storage = LocalStorage(base_path=base_path, config=merged if merged else None)

    def store(self, key: str, data: Any, metadata: dict[str, Any] | None = None) -> bool:
        return self._storage.store(key, data, metadata)

    def retrieve(self, key: str) -> Any | None:
        return self._storage.retrieve(key)

    def exists(self, key: str) -> bool:
        return self._storage.exists(key)

    def delete(self, key: str) -> bool:
        return self._storage.delete(key)

    def list_keys(self, prefix: str | None = None) -> list[str]:
        return self._storage.list_keys(prefix)

    @property
    def storage_type(self) -> str:
        return self._storage.storage_type

    @property
    def supports_metadata(self) -> bool:
        return self._storage.supports_metadata

    @property
    def supports_streaming(self) -> bool:
        return self._storage.supports_streaming

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying storage."""
        return getattr(self._storage, name)
