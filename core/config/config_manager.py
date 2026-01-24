"""
Centralized Configuration Management
===================================

Provides centralized configuration management for all system components
with caching, validation, and factory patterns for efficient access.
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .config_base import BaseConfig, ConfigError, ProcessingConfig, StorageConfig
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseConfig)


class ConfigScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"       # System-wide configuration
    MODULE = "module"       # Module-specific configuration
    COMPONENT = "component" # Component-specific configuration
    INSTANCE = "instance"   # Instance-specific configuration


@dataclass
class ConfigRegistration:
    """
    Configuration registration metadata.

    Tracks registered configuration types with their scope and access statistics.
    """
    name: str
    config_class: type[BaseConfig]
    scope: ConfigScope
    schema: dict[str, Any] | None = None
    factory_func: Callable | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    access_count: int = 0

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now(UTC)
        self.access_count += 1


class ConfigCache:
    """
    Thread-safe configuration cache with TTL support.

    Reduces configuration loading overhead through caching with
    TTL-based invalidation to prevent stale configurations.
    """

    def __init__(self, default_ttl: timedelta = timedelta(minutes=30)):
        """
        Initialize configuration cache.

        Args:
            default_ttl: Default time-to-live for cached configurations
        """
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, datetime] = {}
        self._ttl: dict[str, timedelta] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """
        Get cached configuration.

        Args:
            key: Cache key

        Returns:
            Cached configuration or None if expired/missing
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                return None

            return self._cache[key]

    def set(self, key: str, value: Any, ttl: timedelta | None = None) -> None:
        """
        Set cached configuration.

        Args:
            key: Cache key
            value: Configuration to cache
            ttl: Time-to-live (uses default if None)
        """
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = datetime.now(UTC)
            self._ttl[key] = ttl or self.default_ttl

    def invalidate(self, key: str) -> bool:
        """
        Invalidate cached configuration.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was cached
        """
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttl.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of expired entries removed
        """
        with self._lock:
            expired_keys = [key for key in self._cache.keys() if self._is_expired(key)]
            for key in expired_keys:
                self._remove_key(key)
            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        with self._lock:
            return {
                'total_entries': len(self._cache),
                'expired_entries': len([k for k in self._cache.keys() if self._is_expired(k)]),
                'cache_keys': list(self._cache.keys()),
                'default_ttl_seconds': self.default_ttl.total_seconds()
            }

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True

        timestamp = self._timestamps[key]
        ttl = self._ttl.get(key, self.default_ttl)
        return datetime.now(UTC) - timestamp > ttl

    def _remove_key(self, key: str) -> None:
        """Remove key from all cache structures."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttl.pop(key, None)


class ConfigManager:
    """
    Centralized configuration management with factory patterns and caching.

    Features:
    - Hierarchical scope management (global > module > component > instance)
    - Registration system with schema validation
    - Thread-safe caching with TTL support
    - Factory functions for complex initialization
    """

    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern for global configuration management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._registrations: dict[str, ConfigRegistration] = {}
        self._loader = ConfigLoader()
        self._cache = ConfigCache()
        self._scope_hierarchy = {
            ConfigScope.GLOBAL: 0,
            ConfigScope.MODULE: 1,
            ConfigScope.COMPONENT: 2,
            ConfigScope.INSTANCE: 3
        }

        # Register built-in configuration types
        self._register_builtin_configs()

        logger.info("Configuration manager initialized")

    def _register_builtin_configs(self) -> None:
        """Register built-in configuration types."""
        self.register('processing', ProcessingConfig, ConfigScope.COMPONENT)
        self.register('storage', StorageConfig, ConfigScope.COMPONENT)

    def register(self,
                name: str,
                config_class: type[T],
                scope: ConfigScope = ConfigScope.COMPONENT,
                schema: dict[str, Any] | None = None,
                factory_func: Callable | None = None) -> None:
        """
        Register a configuration type.

        Args:
            name: Configuration name
            config_class: Configuration class
            scope: Configuration scope (determines hierarchy level)
            schema: Optional JSON schema for validation
            factory_func: Optional factory function for complex initialization

        Raises:
            ConfigError: If registration fails
        """
        if name in self._registrations:
            existing = self._registrations[name]
            if existing.config_class != config_class:
                raise ConfigError(f"Configuration '{name}' already registered with different class")
            logger.warning(f"Re-registering configuration '{name}'")

        registration = ConfigRegistration(
            name=name,
            config_class=config_class,
            scope=scope,
            schema=schema,
            factory_func=factory_func
        )

        self._registrations[name] = registration

        # Register schema with loader if provided
        if schema:
            self._loader.register_schema(name, schema)

        logger.debug(f"Registered configuration '{name}' with scope {scope.value}")

    def get(self,
           name: str,
           instance_id: str | None = None,
           force_reload: bool = False,
           **overrides) -> BaseConfig:
        """
        Get configuration instance.

        Retrieves configuration from cache or loads from sources.
        Supports runtime overrides that are merged with base configuration.

        Args:
            name: Configuration name
            instance_id: Optional instance identifier for instance-scoped configs
            force_reload: Force reload from sources (bypasses cache)
            **overrides: Runtime configuration overrides

        Returns:
            Configuration instance

        Raises:
            ConfigError: If configuration cannot be loaded
        """
        if name not in self._registrations:
            raise ConfigError(f"Configuration '{name}' not registered")

        registration = self._registrations[name]
        registration.update_access()

        # Build cache key
        cache_key = name
        if instance_id:
            cache_key = f"{name}:{instance_id}"
        if overrides:
            # Use JSON for stable hashing of potentially nested overrides
            override_str = json.dumps(overrides, sort_keys=True, default=str)
            override_hash = hashlib.md5(override_str.encode()).hexdigest()[:8]
            cache_key = f"{cache_key}:{override_hash}"

        # Check cache
        if not force_reload:
            cached_config = self._cache.get(cache_key)
            if cached_config is not None:
                logger.debug(f"Using cached configuration: {cache_key}")
                return cached_config

        # Load configuration
        try:
            if registration.factory_func:
                # Use custom factory function
                config_instance = registration.factory_func(
                    name, instance_id, overrides
                )
            else:
                # Use hierarchical loader
                config_instance = self._loader.load_hierarchical(
                    config_name=name,
                    config_class=registration.config_class,
                    schema_name=name if registration.schema else None
                )

                # Apply runtime overrides
                if overrides:
                    override_config = registration.config_class.from_dict(overrides)
                    config_instance = config_instance.merge(override_config)

            # Cache the result
            self._cache.set(cache_key, config_instance)

            logger.debug(f"Loaded configuration: {cache_key}")
            return config_instance

        except Exception as e:
            raise ConfigError(f"Failed to load configuration '{name}': {e}")

    def get_or_create(self,
                     name: str,
                     config_class: type[T],
                     instance_id: str | None = None,
                     **kwargs) -> T:
        """
        Get existing configuration or create with default values.

        Args:
            name: Configuration name
            config_class: Configuration class
            instance_id: Optional instance identifier
            **kwargs: Default configuration values

        Returns:
            Configuration instance
        """
        try:
            return self.get(name, instance_id, **kwargs)
        except ConfigError:
            # Create with defaults
            if name not in self._registrations:
                self.register(name, config_class)

            config_instance = config_class.from_dict(kwargs)

            # Cache the result
            cache_key = name
            if instance_id:
                cache_key = f"{name}:{instance_id}"
            self._cache.set(cache_key, config_instance)

            return config_instance

    def invalidate(self, name: str, instance_id: str | None = None) -> None:
        """
        Invalidate cached configuration.

        Args:
            name: Configuration name
            instance_id: Optional instance identifier
        """
        cache_key = name
        if instance_id:
            cache_key = f"{name}:{instance_id}"

        invalidated = self._cache.invalidate(cache_key)
        if invalidated:
            logger.debug(f"Invalidated cached configuration: {cache_key}")

    def invalidate_all(self) -> None:
        """Invalidate all cached configurations."""
        self._cache.clear()
        logger.debug("Invalidated all cached configurations")

    def list_registrations(self) -> list[str]:
        """
        List all registered configuration names.

        Returns:
            List of configuration names
        """
        return list(self._registrations.keys())

    def get_registration_info(self, name: str) -> dict[str, Any] | None:
        """
        Get registration information.

        Args:
            name: Configuration name

        Returns:
            Registration information or None if not found
        """
        if name not in self._registrations:
            return None

        registration = self._registrations[name]
        return {
            'name': registration.name,
            'class': registration.config_class.__name__,
            'scope': registration.scope.value,
            'has_schema': registration.schema is not None,
            'has_factory': registration.factory_func is not None,
            'created_at': registration.created_at.isoformat(),
            'last_accessed': registration.last_accessed.isoformat() if registration.last_accessed else None,
            'access_count': registration.access_count
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get configuration manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'total_registrations': len(self._registrations),
            'registrations_by_scope': {
                scope.value: len([r for r in self._registrations.values() if r.scope == scope])
                for scope in ConfigScope
            },
            'cache_stats': self._cache.get_stats(),
            'most_accessed': [
                {
                    'name': reg.name,
                    'access_count': reg.access_count,
                    'last_accessed': reg.last_accessed.isoformat() if reg.last_accessed else None
                }
                for reg in sorted(self._registrations.values(),
                                key=lambda r: r.access_count, reverse=True)[:5]
            ]
        }

    def cleanup(self) -> None:
        """Cleanup expired cache entries and reset statistics."""
        expired_count = self._cache.cleanup_expired()
        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired cache entries")

    def export_config(self, name: str, output_path: Path) -> None:
        """
        Export configuration to file.

        Args:
            name: Configuration name
            output_path: Output file path

        Raises:
            ConfigError: If export fails
        """
        try:
            config = self.get(name)
            config.save_to_file(output_path)
            logger.info(f"Exported configuration '{name}' to {output_path}")
        except Exception as e:
            raise ConfigError(f"Failed to export configuration '{name}': {e}")


# Global configuration manager instance
config_manager = ConfigManager()
