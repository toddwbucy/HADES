#!/usr/bin/env python3
"""
Database Factory

Factory pattern for creating database connections.
Supports automatic connection type selection and configuration.
"""

import logging
import os
from typing import Any

from core.database.arango import ArangoMemoryClient, resolve_memory_config

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory for creating database connections.

    Manages the instantiation of different database types based on
    configuration, with support for connection pooling and optimization.
    """

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
    ) -> ArangoMemoryClient:
        """Return the optimized ArangoDB memory client.

        The legacy python-arango client has been removed; this helper now wraps
        :func:`resolve_memory_config` and returns :class:`ArangoMemoryClient` so
        existing call-sites can seamlessly adopt the HTTP/2 pathway.
        """

        if password is None:
            password = os.environ.get("ARANGO_PASSWORD")
            if not password:
                raise ValueError("ArangoDB password required (set ARANGO_PASSWORD env var)")

        # Preserve the old ``use_unix`` flag by mapping it to proxy usage when
        # callers still provide it.
        if use_unix is not None:
            use_proxies = True if use_unix else False

        # Build a base URL from host/port when one was not explicitly supplied.
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
    def get_postgres(cls,
                    database: str = "arxiv",
                    username: str = "postgres",
                    password: str | None = None,
                    host: str = "localhost",
                    port: int = 5432,
                    **kwargs) -> Any:
        """
        Get PostgreSQL connection.

        Args:
            database: Database name
            username: Username
            password: Password (or from env)
            host: Database host
            port: Database port
            **kwargs: Additional connection options

        Returns:
            PostgreSQL connection object
        """
        # Get password from environment if not provided
        if password is None:
            password = os.environ.get('PGPASSWORD')
            if not password:
                raise ValueError("PostgreSQL password required (set PGPASSWORD env var)")

        try:
            import psycopg
            conn_string = f"host={host} port={port} dbname={database} user={username} password={password}"
            conn = psycopg.connect(conn_string, **kwargs)
            logger.info(f"✓ Connected to PostgreSQL at {host}:{port}/{database}")
            return conn
        except ImportError:
            # Try psycopg2 as fallback
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                    **kwargs
                )
                logger.info(f"✓ Connected to PostgreSQL (psycopg2) at {host}:{port}/{database}")
                return conn
            except ImportError:
                raise ImportError("Neither psycopg nor psycopg2 installed. Run: pip install psycopg")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @classmethod
    def get_arango_memory_service(
        cls,
        *,
        database: str = "arxiv_repository",
        username: str = "root",
        password: str | None = None,
        socket_path: str | None = None,
        read_socket: str | None = None,
        write_socket: str | None = None,
        use_proxies: bool | None = None,
        base_url: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
    ) -> ArangoMemoryClient:
        """Return the optimized ArangoDB memory client.

        Args:
            database: Target database name.
            username: Authentication user (default "root").
            password: Password (falls back to ``ARANGO_PASSWORD``).
            socket_path: Explicit Unix socket used for both reads and writes.
            read_socket: Optional read-only proxy socket.
            write_socket: Optional read-write proxy socket.
            use_proxies: Force proxy usage (defaults to environment autodetect).
            base_url: Base URL for HTTP/2 client (defaults to http://localhost).
            connect_timeout: Override connection timeout.
            read_timeout: Override read timeout.
            write_timeout: Override write timeout.

        Returns:
            Configured :class:`ArangoMemoryClient` instance.
        """

        config = resolve_memory_config(
            database=database,
            username=username,
            password=password,
            socket_path=socket_path,
            read_socket=read_socket,
            write_socket=write_socket,
            use_proxies=use_proxies,
            base_url=base_url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

        logger.info(
            "✓ Using Arango memory client (read_socket=%s, write_socket=%s)",
            config.read_socket,
            config.write_socket,
        )
        return ArangoMemoryClient(config)

    @classmethod
    def get_redis(cls,
                  host: str = "localhost",
                  port: int = 6379,
                  db: int = 0,
                  password: str | None = None,
                  **kwargs) -> Any:
        """
        Get Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Database number
            password: Password if required
            **kwargs: Additional connection options

        Returns:
            Redis connection object
        """
        try:
            import redis
            conn = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                **kwargs
            )
            # Test connection
            conn.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port}/{db}")
            return conn
        except ImportError:
            raise ImportError("redis not installed. Run: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @classmethod
    def create_pool(cls, db_type: str, pool_size: int = 10, **kwargs) -> Any:
        """
        Create a connection pool for the specified database type.

        Args:
            db_type: Type of database (arango, postgres, redis)
            pool_size: Size of connection pool
            **kwargs: Database-specific connection options

        Returns:
            Connection pool object
        """
        if db_type == "postgres":
            try:
                import psycopg_pool
                conn_string = cls._build_postgres_conn_string(**kwargs)
                pool = psycopg_pool.ConnectionPool(
                    conn_string,
                    min_size=1,
                    max_size=pool_size
                )
                logger.info(f"✓ Created PostgreSQL connection pool (size={pool_size})")
                return pool
            except ImportError:
                logger.warning("psycopg_pool not available, returning single connection")
                return cls.get_postgres(**kwargs)

        elif db_type == "redis":
            try:
                import redis
                pool = redis.ConnectionPool(
                    max_connections=pool_size,
                    **kwargs
                )
                conn = redis.Redis(connection_pool=pool)
                logger.info(f"✓ Created Redis connection pool (size={pool_size})")
                return conn
            except ImportError:
                raise ImportError("redis not installed")

        else:
            # ArangoDB handles pooling internally
            return cls.get_arango(**kwargs)

    @staticmethod
    def _build_postgres_conn_string(**kwargs) -> str:
        """Build PostgreSQL connection string from kwargs."""
        from typing import Any

        # Normalize keys to libpq names
        key_map = {"database": "dbname", "username": "user"}
        mapped: dict[str, Any] = {}

        for key, value in kwargs.items():
            if value is None:
                continue
            # Map to correct libpq key names
            mapped_key = key_map.get(key, key)
            mapped[mapped_key] = value

        # Password fallback from environment if not provided
        if "password" not in mapped:
            env_pw = os.environ.get("PGPASSWORD")
            if env_pw:
                mapped["password"] = env_pw

        return " ".join(f"{k}={v}" for k, v in mapped.items())
