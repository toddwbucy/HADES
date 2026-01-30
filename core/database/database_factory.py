#!/usr/bin/env python3
"""Database Factory with Registry Pattern.

Factory pattern for creating database connections using a registry.
Supports automatic connection type selection, lazy loading, and configuration.

Usage:
    # Create database connection
    client = DatabaseFactory.create("arango", database="my_db")

    # List available backends
    DatabaseFactory.list_available()

    # Register custom backend
    @DatabaseFactory.register("custom")
    class CustomDB(DatabaseBase):
        ...
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class DatabaseBase(ABC):
    """Abstract base class for database backends.

    All database backends must implement this interface to be
    compatible with the DatabaseFactory registry.
    """

    @property
    @abstractmethod
    def db_type(self) -> str:
        """Return the database type identifier."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the database is connected."""
        ...


class DatabaseFactory:
    """Factory for creating database connections using registry pattern.

    Manages the instantiation of different database types based on
    configuration, with support for lazy loading and auto-detection.

    Example:
        # Create ArangoDB client
        client = DatabaseFactory.create("arango", database="my_db")

        # List available backends
        available = DatabaseFactory.list_available()

        # Register new backend
        @DatabaseFactory.register("mydb")
        class MyDBClient(DatabaseBase):
            ...
    """

    # Registry of available database backends
    _registry: ClassVar[dict[str, type]] = {}

    # Track which backends have been auto-registered
    _auto_registered: ClassVar[set[str]] = set()

    @classmethod
    def register(cls, name: str):
        """Decorator to register a database backend class.

        Args:
            name: Name to register under (e.g., "arango", "postgres")

        Returns:
            Decorator function

        Example:
            @DatabaseFactory.register("mydb")
            class MyDBClient(DatabaseBase):
                ...
        """
        def decorator(db_class: type) -> type:
            cls._registry[name] = db_class
            logger.debug(f"Registered database backend: {name}")
            return db_class
        return decorator

    @classmethod
    def create(cls, db_type: str, **kwargs) -> Any:
        """Create a database connection instance.

        Args:
            db_type: Type of database ("arango", "postgres", "redis")
            **kwargs: Database-specific connection options

        Returns:
            Database connection instance

        Raises:
            ValueError: If no backend registered for db_type
        """
        # Try auto-registration if not already registered
        if db_type not in cls._registry:
            cls._auto_register(db_type)

        if db_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"No database backend registered for '{db_type}'. "
                f"Available: {available}"
            )

        db_class = cls._registry[db_type]
        logger.info(f"Creating {db_type} database connection")

        return db_class(**kwargs)

    @classmethod
    def _auto_register(cls, db_type: str) -> None:
        """Auto-register a database backend on first use.

        This enables lazy loading of database modules.

        Args:
            db_type: Type of database to register
        """
        if db_type in cls._auto_registered:
            return

        try:
            if db_type == "arango":
                # Create a wrapper that matches DatabaseBase interface
                cls._registry["arango"] = _ArangoWrapper
                cls._auto_registered.add(db_type)
                logger.debug("Auto-registered arango backend")

            elif db_type == "postgres":
                cls._registry["postgres"] = _PostgresWrapper
                cls._auto_registered.add(db_type)
                logger.debug("Auto-registered postgres backend")

            elif db_type == "redis":
                cls._registry["redis"] = _RedisWrapper
                cls._auto_registered.add(db_type)
                logger.debug("Auto-registered redis backend")

            else:
                # Mark unknown types to prevent repeated warnings
                cls._auto_registered.add(db_type)
                logger.warning(f"Unknown database type for auto-registration: {db_type}")

        except ImportError:
            logger.exception(f"Failed to auto-register {db_type}")

    @classmethod
    def list_available(cls) -> dict[str, dict[str, Any]]:
        """List available database backends.

        Returns:
            Dictionary mapping backend names to their info
        """
        # Ensure all built-in backends are registered
        for db_type in ["arango", "postgres", "redis"]:
            if db_type not in cls._registry:
                cls._auto_register(db_type)

        available = {}
        for name, db_class in cls._registry.items():
            try:
                available[name] = {
                    "class": db_class.__name__,
                    "module": db_class.__module__,
                }
            except Exception as e:
                available[name] = {"error": str(e)}

        return available

    # =========================================================================
    # Legacy API (preserved for backwards compatibility)
    # =========================================================================

    @classmethod
    def get_arango(
        cls,
        database: str = "academy_store",
        username: str = "root",
        password: str | None = None,
        host: str = "localhost",
        port: int = 8529,
        use_unix: bool | None = None,
        base_url: str | None = None,
        socket_path: str | None = None,
        read_socket: str | None = None,
        write_socket: str | None = None,
        use_proxies: bool | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        **_: Any,
    ) -> Any:
        """Return the optimized ArangoDB memory client.

        Legacy API preserved for backwards compatibility.
        Prefer using: DatabaseFactory.create("arango", ...)

        Returns:
            ArangoMemoryClient instance
        """
        from core.database.arango import ArangoMemoryClient, resolve_memory_config

        if password is None:
            password = os.environ.get("ARANGO_PASSWORD")
            if not password:
                raise ValueError("ArangoDB password required (set ARANGO_PASSWORD env var)")

        if use_unix is not None:
            use_proxies = True if use_unix else False

        if base_url is None and host:
            base_url = f"http://{host}:{port}"

        config = resolve_memory_config(
            database=database,
            username=username,
            password=password,
            base_url=base_url,
            socket_path=socket_path,
            read_socket=read_socket,
            write_socket=write_socket,
            use_proxies=use_proxies,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

        logger.info(
            "✓ Using Arango memory client via HTTP/2 (read_socket=%s, write_socket=%s)",
            config.read_socket,
            config.write_socket,
        )
        return ArangoMemoryClient(config)

    @classmethod
    def get_postgres(
        cls,
        database: str = "arxiv",
        username: str = "postgres",
        password: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        **kwargs,
    ) -> Any:
        """Get PostgreSQL connection.

        Legacy API preserved for backwards compatibility.
        Prefer using: DatabaseFactory.create("postgres", ...)
        """
        return cls.create(
            "postgres",
            database=database,
            username=username,
            password=password,
            host=host,
            port=port,
            **kwargs,
        )

    @classmethod
    def get_redis(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        **kwargs,
    ) -> Any:
        """Get Redis connection.

        Legacy API preserved for backwards compatibility.
        Prefer using: DatabaseFactory.create("redis", ...)
        """
        return cls.create(
            "redis",
            host=host,
            port=port,
            db=db,
            password=password,
            **kwargs,
        )

    @classmethod
    def get_arango_memory_service(cls, **kwargs) -> Any:
        """Return the optimized ArangoDB memory client.

        Legacy API preserved for backwards compatibility.
        Prefer using: DatabaseFactory.create("arango", ...)

        Returns:
            ArangoMemoryClient instance
        """
        return cls.get_arango(**kwargs)


# =============================================================================
# Backend Wrapper Classes
# =============================================================================


class _ArangoWrapper:
    """Wrapper for ArangoMemoryClient that provides factory-compatible interface."""

    def __init__(
        self,
        database: str = "academy_store",
        username: str = "root",
        password: str | None = None,
        host: str = "localhost",
        port: int = 8529,
        use_unix: bool | None = None,
        base_url: str | None = None,
        socket_path: str | None = None,
        read_socket: str | None = None,
        write_socket: str | None = None,
        use_proxies: bool | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
    ):
        from core.database.arango import ArangoMemoryClient, resolve_memory_config

        if password is None:
            password = os.environ.get("ARANGO_PASSWORD")
            if not password:
                raise ValueError("ArangoDB password required (set ARANGO_PASSWORD env var)")

        # Legacy use_unix maps to use_proxies
        if use_unix is not None:
            use_proxies = True if use_unix else False

        if base_url is None and host:
            base_url = f"http://{host}:{port}"

        config = resolve_memory_config(
            database=database,
            username=username,
            password=password,
            base_url=base_url,
            socket_path=socket_path,
            read_socket=read_socket,
            write_socket=write_socket,
            use_proxies=use_proxies,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

        self._client = ArangoMemoryClient(config)
        logger.info(
            "✓ Created Arango client (read_socket=%s, write_socket=%s)",
            config.read_socket,
            config.write_socket,
        )

    @property
    def db_type(self) -> str:
        return "arango"

    def close(self) -> None:
        self._client.close()

    def is_connected(self) -> bool:
        try:
            self._client.server_version()
            return True
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying client."""
        return getattr(self._client, name)


class _PostgresWrapper:
    """Wrapper for PostgreSQL connections."""

    def __init__(
        self,
        database: str = "arxiv",
        username: str = "postgres",
        password: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        **kwargs,
    ):
        if password is None:
            password = os.environ.get("PGPASSWORD")
            if not password:
                raise ValueError("PostgreSQL password required (set PGPASSWORD env var)")

        try:
            import psycopg
            # Use keyword arguments to handle special characters in credentials
            self._conn = psycopg.connect(
                host=host,
                port=port,
                dbname=database,
                user=username,
                password=password,
                **kwargs,
            )
            logger.info(f"✓ Connected to PostgreSQL at {host}:{port}/{database}")
        except ImportError:
            try:
                import psycopg2
                self._conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                    **kwargs,
                )
                logger.info(f"✓ Connected to PostgreSQL (psycopg2) at {host}:{port}/{database}")
            except ImportError as e:
                raise ImportError("Neither psycopg nor psycopg2 installed. Run: pip install psycopg") from e

    @property
    def db_type(self) -> str:
        return "postgres"

    def close(self) -> None:
        self._conn.close()

    def is_connected(self) -> bool:
        try:
            return not self._conn.closed
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying connection."""
        return getattr(self._conn, name)


class _RedisWrapper:
    """Wrapper for Redis connections."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        decode_responses: bool = True,
        **kwargs,
    ):
        try:
            import redis
            self._conn = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                **kwargs,
            )
            # Test connection
            self._conn.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port}/{db}")
        except ImportError as e:
            raise ImportError("redis not installed. Run: pip install redis") from e

    @property
    def db_type(self) -> str:
        return "redis"

    def close(self) -> None:
        self._conn.close()

    def is_connected(self) -> bool:
        try:
            self._conn.ping()
            return True
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying connection."""
        return getattr(self._conn, name)
