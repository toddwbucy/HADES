"""HADES Database Service Client.

Client library for communicating with ArangoDB over Unix socket.
Provides health checks, service availability detection, and graceful fallback.

Usage:
    from core.services import DatabaseClient

    with DatabaseClient() as client:
        if client.is_service_available():
            result = client.query("FOR doc IN collection RETURN doc")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database.arango.optimized_client import ArangoHttp2Client

logger = logging.getLogger(__name__)


class DatabaseServiceError(Exception):
    """Error communicating with the database service."""

    pass


@dataclass
class DatabaseClientConfig:
    """Configuration for the database client.

    Attributes:
        ro_socket_path: Path to the read-only Unix socket
        rw_socket_path: Path to the read-write Unix socket
        fallback_to_http: Whether to fall back to HTTP if socket unavailable
        http_host: HTTP fallback host
        http_port: HTTP fallback port
        database: Database name to use
        username: Authentication username
        password: Authentication password
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
    """

    ro_socket_path: str | None = None
    rw_socket_path: str | None = None
    fallback_to_http: bool = True
    http_host: str = "localhost"
    http_port: int = 8529
    database: str = "_system"
    username: str | None = None
    password: str | None = None
    connect_timeout: float = 5.0
    read_timeout: float = 30.0

    def __post_init__(self):
        """Set default socket paths from environment variables."""
        if self.ro_socket_path is None:
            self.ro_socket_path = os.environ.get(
                "ARANGO_RO_SOCKET", "/run/hades/readonly/arangod.sock"
            )
        if self.rw_socket_path is None:
            self.rw_socket_path = os.environ.get(
                "ARANGO_RW_SOCKET", "/run/hades/readwrite/arangod.sock"
            )
        if self.password is None:
            self.password = os.environ.get("ARANGO_PASSWORD")
        if self.username is None:
            self.username = os.environ.get("ARANGO_USERNAME", "root")


class DatabaseClient:
    """Client for the HADES database service.

    Provides health checking, service availability detection, and graceful
    fallback from Unix socket to HTTP connection.

    Attributes:
        config: Client configuration
        read_only: Whether to use read-only connection

    Example:
        with DatabaseClient() as client:
            if client.is_service_available():
                health = client.get_health()
                print(f"Database latency: {health['latency_ms']}ms")
    """

    def __init__(
        self,
        config: DatabaseClientConfig | None = None,
        read_only: bool = True,
    ) -> None:
        """Initialize the database client.

        Args:
            config: Client configuration (uses defaults if not provided)
            read_only: Use read-only socket (default True)
        """
        self.config = config or DatabaseClientConfig()
        self.read_only = read_only

        self._client: ArangoHttp2Client | None = None
        self._service_available: bool | None = None
        self._using_socket: bool = True

    def _get_socket_path(self) -> str | None:
        """Get the appropriate socket path based on read_only setting."""
        if self.read_only:
            return self.config.ro_socket_path
        return self.config.rw_socket_path

    def _create_client(self, use_socket: bool = True) -> ArangoHttp2Client:
        """Create an ArangoDB client.

        Args:
            use_socket: Whether to use Unix socket (True) or HTTP (False)

        Returns:
            Configured ArangoHttp2Client instance
        """
        from core.database.arango.optimized_client import (
            ArangoHttp2Client,
            ArangoHttp2Config,
        )

        socket_path = self._get_socket_path() if use_socket else None

        if use_socket and socket_path:
            base_url = "http://localhost"
        else:
            base_url = f"http://{self.config.http_host}:{self.config.http_port}"

        arango_config = ArangoHttp2Config(
            database=self.config.database,
            socket_path=socket_path if use_socket else None,
            base_url=base_url,
            username=self.config.username,
            password=self.config.password,
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
        )

        return ArangoHttp2Client(arango_config)

    def _get_client(self) -> ArangoHttp2Client:
        """Get or create the ArangoDB client with fallback support.

        Returns:
            Active ArangoHttp2Client instance

        Raises:
            DatabaseServiceError: If no connection can be established
        """
        if self._client is not None:
            return self._client

        # Try socket connection first
        socket_path = self._get_socket_path()
        if socket_path and os.path.exists(socket_path):
            try:
                self._client = self._create_client(use_socket=True)
                self._using_socket = True
                logger.debug(f"Connected via Unix socket: {socket_path}")
                return self._client
            except Exception as e:
                logger.warning(f"Socket connection failed: {e}")
                if not self.config.fallback_to_http:
                    raise DatabaseServiceError(f"Socket connection failed: {e}") from e

        # Fall back to HTTP if enabled
        if self.config.fallback_to_http:
            try:
                self._client = self._create_client(use_socket=False)
                self._using_socket = False
                logger.info(
                    f"Connected via HTTP fallback: "
                    f"{self.config.http_host}:{self.config.http_port}"
                )
                return self._client
            except Exception as e:
                raise DatabaseServiceError(f"HTTP connection failed: {e}") from e

        raise DatabaseServiceError(
            f"Database unavailable: socket {socket_path} does not exist "
            "and HTTP fallback is disabled"
        )

    def is_service_available(self, force_check: bool = False) -> bool:
        """Check if the database service is available.

        Args:
            force_check: Force a fresh check instead of using cached result

        Returns:
            True if database is available and responding
        """
        if self._service_available is not None and not force_check:
            return self._service_available

        try:
            # Try to get server version as health check
            client = self._get_client()
            response = client._client.get("/_api/version")
            self._service_available = response.status_code == 200
        except Exception as e:
            logger.debug(f"Service availability check failed: {e}")
            self._service_available = False

        return self._service_available

    def get_health(self) -> dict[str, Any]:
        """Get database health information.

        Returns:
            Health response dict with:
            - available: Whether database is responding
            - latency_ms: Round-trip latency in milliseconds
            - server_version: ArangoDB server version
            - connection_type: 'socket' or 'http'
            - database: Current database name
            - engine: Storage engine (rocksdb, etc.)

        Raises:
            DatabaseServiceError: If health check fails
        """
        try:
            client = self._get_client()

            # Measure latency with version endpoint
            start = time.time()
            response = client._client.get("/_api/version")
            latency_ms = (time.time() - start) * 1000

            if response.status_code != 200:
                raise DatabaseServiceError(
                    f"Health check failed with status {response.status_code}"
                )

            data = response.json()

            # Get engine info
            engine_response = client._client.get("/_api/engine")
            engine_info = {}
            if engine_response.status_code == 200:
                engine_info = engine_response.json()

            return {
                "available": True,
                "latency_ms": round(latency_ms, 2),
                "server_version": data.get("version", "unknown"),
                "server_license": data.get("license", "unknown"),
                "connection_type": "socket" if self._using_socket else "http",
                "socket_path": self._get_socket_path() if self._using_socket else None,
                "database": self.config.database,
                "engine": engine_info.get("name", "unknown"),
            }

        except DatabaseServiceError:
            raise
        except Exception as e:
            raise DatabaseServiceError(f"Health check failed: {e}") from e

    def query(
        self,
        aql: str,
        bind_vars: dict[str, Any] | None = None,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """Execute an AQL query.

        Args:
            aql: AQL query string
            bind_vars: Query bind variables
            batch_size: Cursor batch size

        Returns:
            List of result documents

        Raises:
            DatabaseServiceError: If query fails
        """
        try:
            client = self._get_client()
            return client.query(aql, bind_vars=bind_vars, batch_size=batch_size)
        except Exception as e:
            raise DatabaseServiceError(f"Query failed: {e}") from e

    def get_document(self, collection: str, key: str) -> dict[str, Any]:
        """Fetch a single document by collection and key.

        Args:
            collection: Collection name
            key: Document key

        Returns:
            Document data

        Raises:
            DatabaseServiceError: If document fetch fails
        """
        try:
            client = self._get_client()
            return client.get_document(collection, key)
        except Exception as e:
            raise DatabaseServiceError(f"Document fetch failed: {e}") from e

    def get_client(self) -> ArangoHttp2Client:
        """Get the underlying ArangoDB client for direct access.

        Returns:
            The ArangoHttp2Client instance

        Note:
            Use this for operations not wrapped by DatabaseClient.
            The client is managed by DatabaseClient - don't close it directly.
        """
        return self._get_client()

    def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._service_available = None

    def __enter__(self) -> DatabaseClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


# Convenience function for quick health check
def check_database_health(
    ro_socket_path: str | None = None,
    fallback_to_http: bool = True,
) -> dict[str, Any]:
    """Check database health using a temporary client.

    Args:
        ro_socket_path: Path to read-only socket (uses default if None)
        fallback_to_http: Fall back to HTTP if socket unavailable

    Returns:
        Health information dict

    Raises:
        DatabaseServiceError: If health check fails
    """
    config = DatabaseClientConfig(
        ro_socket_path=ro_socket_path,
        fallback_to_http=fallback_to_http,
    )
    with DatabaseClient(config=config) as client:
        return client.get_health()
