"""Integration tests for database operations."""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
)
from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
from core.database.database_factory import DatabaseFactory


class TestDatabaseFactory:
    """Tests for DatabaseFactory."""

    def test_factory_has_get_arango_method(self) -> None:
        """DatabaseFactory should have get_arango method."""
        assert hasattr(DatabaseFactory, "get_arango")
        assert callable(DatabaseFactory.get_arango)

    def test_factory_has_get_arango_memory_service_method(self) -> None:
        """DatabaseFactory should have get_arango_memory_service method."""
        assert hasattr(DatabaseFactory, "get_arango_memory_service")
        assert callable(DatabaseFactory.get_arango_memory_service)


class TestArangoMemoryClientConfig:
    """Tests for ArangoMemoryClientConfig."""

    def test_config_accepts_all_parameters(self) -> None:
        """Config should accept all connection parameters."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket=None,
            write_socket=None,
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        assert config.database == "test_db"
        assert config.username == "root"
        assert config.password == "secret"
        assert config.base_url == "http://localhost:8529"

    def test_config_builds_read_config(self) -> None:
        """Config should build read HTTP/2 config."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/tmp/ro.sock",
            write_socket="/tmp/rw.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        http2_config = config.build_read_config()
        assert isinstance(http2_config, ArangoHttp2Config)
        assert http2_config.socket_path == "/tmp/ro.sock"

    def test_config_builds_write_config(self) -> None:
        """Config should build write HTTP/2 config."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/tmp/ro.sock",
            write_socket="/tmp/rw.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        http2_config = config.build_write_config()
        assert isinstance(http2_config, ArangoHttp2Config)
        assert http2_config.socket_path == "/tmp/rw.sock"


class TestArangoHttp2Config:
    """Tests for ArangoHttp2Config."""

    def test_config_with_socket_path(self) -> None:
        """Config should accept socket path for Unix socket transport."""
        config = ArangoHttp2Config(
            database="test",
            socket_path="/var/run/arangodb.sock",
            base_url="http://localhost",
        )

        assert config.socket_path == "/var/run/arangodb.sock"

    def test_config_without_socket_path(self) -> None:
        """Config should work without socket path (network transport)."""
        config = ArangoHttp2Config(
            database="test",
            socket_path=None,
            base_url="http://localhost:8529",
        )

        assert config.socket_path is None
        assert config.base_url == "http://localhost:8529"


class TestMockedDatabaseOperations:
    """Tests for database operations with mocked client."""

    @pytest.fixture
    def mock_http2_client(self) -> MagicMock:
        """Create mock HTTP/2 client."""
        mock = MagicMock(spec=ArangoHttp2Client)
        mock.query.return_value = []
        mock.request.return_value = {"result": []}
        mock.insert_documents.return_value = {"created": 1}
        mock.close.return_value = None
        return mock

    def test_execute_query_calls_client(self, mock_http2_client: MagicMock, clean_env) -> None:
        """execute_query should call underlying client."""
        from core.database.arango import memory_client as mc_module

        with patch.object(mc_module, "ArangoHttp2Client", return_value=mock_http2_client):
            config = ArangoMemoryClientConfig(
                database="test",
                username="root",
                password="test_password",
                base_url="http://localhost:8529",
                read_socket=None,
                write_socket=None,
                connect_timeout=5.0,
                read_timeout=30.0,
                write_timeout=30.0,
            )

            client = ArangoMemoryClient(config)
            try:
                # This tests the interface without actual DB
                assert hasattr(client, "execute_query")
            finally:
                client.close()

    def test_insert_in_transaction_method_exists(self, mock_http2_client: MagicMock, clean_env) -> None:
        """insert_in_transaction should exist on client."""
        from core.database.arango import memory_client as mc_module

        with patch.object(mc_module, "ArangoHttp2Client", return_value=mock_http2_client):
            config = ArangoMemoryClientConfig(
                database="test",
                username="root",
                password="test_password",
                base_url="http://localhost:8529",
                read_socket=None,
                write_socket=None,
                connect_timeout=5.0,
                read_timeout=30.0,
                write_timeout=30.0,
            )

            client = ArangoMemoryClient(config)
            try:
                # ArangoMemoryClient uses insert_in_transaction for document insertion
                assert hasattr(client, "insert_in_transaction")
            finally:
                client.close()


@pytest.mark.skipif(
    os.environ.get("ARANGO_PASSWORD") is None,
    reason="ArangoDB credentials not configured",
)
class TestLiveDatabaseConnection:
    """Tests that require a live ArangoDB connection.

    These tests are skipped unless ARANGO_PASSWORD is set.
    """

    @pytest.fixture
    def live_client(self) -> Generator[ArangoMemoryClient, None, None]:
        """Create a live database client.

        Note: Does NOT use clean_env fixture to preserve real credentials.
        """
        client = DatabaseFactory.get_arango_memory_service(
            database="_system",
            use_proxies=False,
        )
        try:
            yield client
        finally:
            client.close()

    def test_can_connect_to_database(self, live_client: ArangoMemoryClient) -> None:
        """Should be able to connect to ArangoDB."""
        # Simple connectivity test
        result = live_client.execute_query("RETURN 1")
        assert result == [1]

    def test_can_execute_basic_query(self, live_client: ArangoMemoryClient) -> None:
        """Should be able to execute basic AQL queries."""
        # Test array creation and manipulation
        result = live_client.execute_query("RETURN [1, 2, 3]")
        assert isinstance(result, list)
        assert result == [[1, 2, 3]]

    def test_can_execute_math_query(self, live_client: ArangoMemoryClient) -> None:
        """Should be able to execute math operations."""
        result = live_client.execute_query("RETURN 2 + 2")
        assert isinstance(result, list)
        assert result == [4]


class TestTransactionSupport:
    """Tests for transaction support."""

    def test_client_has_transaction_methods(self, clean_env) -> None:
        """Client should have transaction methods."""
        from core.database.arango import memory_client as mc_module

        mock_client = MagicMock()
        with patch.object(mc_module, "ArangoHttp2Client", return_value=mock_client):
            config = ArangoMemoryClientConfig(
                database="test",
                username="root",
                password="test_password",
                base_url="http://localhost:8529",
                read_socket=None,
                write_socket=None,
                connect_timeout=5.0,
                read_timeout=30.0,
                write_timeout=30.0,
            )

            client = ArangoMemoryClient(config)
            try:
                assert hasattr(client, "begin_transaction")
                assert hasattr(client, "commit_transaction")
                assert hasattr(client, "abort_transaction")
            finally:
                client.close()
