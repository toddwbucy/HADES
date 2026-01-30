"""Unit tests for DatabaseClient.

Tests for:
- Configuration initialization
- Service availability checking
- Health check functionality
- Context manager support
- Fallback behavior
"""

from unittest.mock import MagicMock, patch


class TestDatabaseClientConfig:
    """Tests for DatabaseClientConfig initialization."""

    def test_default_values(self):
        """Should use default values when not specified."""
        from core.services.database_client import DatabaseClientConfig

        # Clear environment variables for clean test
        with patch.dict("os.environ", {}, clear=True):
            config = DatabaseClientConfig()

        assert config.fallback_to_http is True
        assert config.http_host == "localhost"
        assert config.http_port == 8529
        assert config.database == "_system"
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 30.0

    def test_reads_from_environment(self):
        """Should read socket paths and password from environment."""
        from core.services.database_client import DatabaseClientConfig

        env_vars = {
            "ARANGO_RO_SOCKET": "/custom/ro.sock",
            "ARANGO_RW_SOCKET": "/custom/rw.sock",
            "ARANGO_PASSWORD": "secret123",
            "ARANGO_USERNAME": "admin",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = DatabaseClientConfig()

        assert config.ro_socket_path == "/custom/ro.sock"
        assert config.rw_socket_path == "/custom/rw.sock"
        assert config.password == "secret123"
        assert config.username == "admin"

    def test_explicit_values_override_environment(self):
        """Should use explicit values over environment variables."""
        from core.services.database_client import DatabaseClientConfig

        env_vars = {
            "ARANGO_RO_SOCKET": "/env/ro.sock",
            "ARANGO_PASSWORD": "env_password",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = DatabaseClientConfig(
                ro_socket_path="/explicit/ro.sock",
                password="explicit_password",
            )

        assert config.ro_socket_path == "/explicit/ro.sock"
        assert config.password == "explicit_password"


class TestDatabaseClientInit:
    """Tests for DatabaseClient initialization."""

    def test_default_config(self):
        """Should initialize with default configuration."""
        from core.services.database_client import DatabaseClient

        with patch.dict("os.environ", {}, clear=True):
            client = DatabaseClient()

        assert client.config is not None
        assert client.read_only is True
        assert client._client is None
        assert client._service_available is None

    def test_custom_config(self):
        """Should accept custom configuration."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseClientConfig,
        )

        config = DatabaseClientConfig(
            database="test_db",
            http_port=8530,
        )
        client = DatabaseClient(config=config, read_only=False)

        assert client.config.database == "test_db"
        assert client.config.http_port == 8530
        assert client.read_only is False

    def test_get_socket_path_read_only(self):
        """Should return read-only socket path when read_only=True."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseClientConfig,
        )

        config = DatabaseClientConfig(
            ro_socket_path="/ro.sock",
            rw_socket_path="/rw.sock",
        )
        client = DatabaseClient(config=config, read_only=True)

        assert client._get_socket_path() == "/ro.sock"

    def test_get_socket_path_read_write(self):
        """Should return read-write socket path when read_only=False."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseClientConfig,
        )

        config = DatabaseClientConfig(
            ro_socket_path="/ro.sock",
            rw_socket_path="/rw.sock",
        )
        client = DatabaseClient(config=config, read_only=False)

        assert client._get_socket_path() == "/rw.sock"


class TestDatabaseClientAvailability:
    """Tests for service availability checking."""

    def test_is_service_available_success(self):
        """Should return True when database responds."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http_client.get.return_value = mock_response

        mock_arango_client = MagicMock()
        mock_arango_client._client = mock_http_client
        client._client = mock_arango_client

        result = client.is_service_available()

        assert result is True
        assert client._service_available is True

    def test_is_service_available_failure(self):
        """Should return False when database doesn't respond."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        mock_http_client = MagicMock()
        mock_http_client.get.side_effect = Exception("Connection refused")

        mock_arango_client = MagicMock()
        mock_arango_client._client = mock_http_client
        client._client = mock_arango_client

        result = client.is_service_available()

        assert result is False
        assert client._service_available is False

    def test_is_service_available_cached(self):
        """Should return cached result without force_check."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()
        client._service_available = True

        # Should not make any calls
        result = client.is_service_available()

        assert result is True

    def test_is_service_available_force_check(self):
        """Should make fresh check when force_check=True."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()
        client._service_available = True

        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_http_client.get.return_value = mock_response

        mock_arango_client = MagicMock()
        mock_arango_client._client = mock_http_client
        client._client = mock_arango_client

        result = client.is_service_available(force_check=True)

        assert result is False
        mock_http_client.get.assert_called_once()


class TestDatabaseClientHealth:
    """Tests for health check functionality."""

    def test_get_health_success(self):
        """Should return health info on success."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()
        client._using_socket = True

        mock_http_client = MagicMock()

        # Version response
        version_response = MagicMock()
        version_response.status_code = 200
        version_response.json.return_value = {
            "version": "3.11.0",
            "license": "community",
        }

        # Engine response
        engine_response = MagicMock()
        engine_response.status_code = 200
        engine_response.json.return_value = {"name": "rocksdb"}

        mock_http_client.get.side_effect = [version_response, engine_response]

        mock_arango_client = MagicMock()
        mock_arango_client._client = mock_http_client
        client._client = mock_arango_client

        health = client.get_health()

        assert health["available"] is True
        assert health["server_version"] == "3.11.0"
        assert health["engine"] == "rocksdb"
        assert health["connection_type"] == "socket"
        assert "latency_ms" in health

    def test_get_health_failure(self):
        """Should raise DatabaseServiceError on failure."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseServiceError,
        )

        client = DatabaseClient()

        mock_http_client = MagicMock()
        mock_http_client.get.side_effect = Exception("Connection failed")

        mock_arango_client = MagicMock()
        mock_arango_client._client = mock_http_client
        client._client = mock_arango_client

        import pytest

        with pytest.raises(DatabaseServiceError, match="Health check failed"):
            client.get_health()


class TestDatabaseClientContextManager:
    """Tests for context manager support."""

    def test_context_manager_enter(self):
        """Should return self on enter."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        result = client.__enter__()

        assert result is client

    def test_context_manager_exit_closes_client(self):
        """Should close client on exit."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()
        mock_arango_client = MagicMock()
        client._client = mock_arango_client

        client.__exit__(None, None, None)

        mock_arango_client.close.assert_called_once()
        assert client._client is None

    def test_context_manager_with_statement(self):
        """Should work with 'with' statement."""
        from core.services.database_client import DatabaseClient

        with DatabaseClient() as client:
            assert client is not None
            # Client should be usable inside context


class TestDatabaseClientFallback:
    """Tests for fallback behavior."""

    def test_falls_back_to_http_when_socket_unavailable(self):
        """Should fall back to HTTP when socket doesn't exist."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseClientConfig,
        )

        config = DatabaseClientConfig(
            ro_socket_path="/nonexistent/socket.sock",
            fallback_to_http=True,
        )
        client = DatabaseClient(config=config)

        # Mock the HTTP client creation
        with patch(
            "core.services.database_client.DatabaseClient._create_client"
        ) as mock_create:
            mock_arango = MagicMock()
            mock_create.return_value = mock_arango

            client._get_client()

            # Should have been called with use_socket=False (fallback)
            assert mock_create.call_count >= 1
            last_call = mock_create.call_args_list[-1]
            assert last_call[1].get("use_socket") is False

    def test_raises_error_when_fallback_disabled(self):
        """Should raise error when socket unavailable and fallback disabled."""
        from core.services.database_client import (
            DatabaseClient,
            DatabaseClientConfig,
            DatabaseServiceError,
        )

        config = DatabaseClientConfig(
            ro_socket_path="/nonexistent/socket.sock",
            fallback_to_http=False,
        )
        client = DatabaseClient(config=config)

        import pytest

        with pytest.raises(DatabaseServiceError, match="does not exist"):
            client._get_client()


class TestDatabaseClientOperations:
    """Tests for database operations."""

    def test_query_delegates_to_client(self):
        """Should delegate query to underlying client."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        mock_arango_client = MagicMock()
        mock_arango_client.query.return_value = [{"_key": "1"}]
        client._client = mock_arango_client

        result = client.query("FOR doc IN test RETURN doc")

        mock_arango_client.query.assert_called_once()
        assert result == [{"_key": "1"}]

    def test_get_document_delegates_to_client(self):
        """Should delegate get_document to underlying client."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        mock_arango_client = MagicMock()
        mock_arango_client.get_document.return_value = {"_key": "1", "name": "test"}
        client._client = mock_arango_client

        result = client.get_document("test_collection", "1")

        mock_arango_client.get_document.assert_called_once_with("test_collection", "1")
        assert result["name"] == "test"

    def test_get_client_returns_underlying_client(self):
        """Should return the underlying ArangoHttp2Client."""
        from core.services.database_client import DatabaseClient

        client = DatabaseClient()

        mock_arango_client = MagicMock()
        client._client = mock_arango_client

        result = client.get_client()

        assert result is mock_arango_client


class TestConvenienceFunction:
    """Tests for the convenience function."""

    def test_check_database_health(self):
        """Should return health info using temporary client."""
        from core.services.database_client import check_database_health

        with patch("core.services.database_client.DatabaseClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.get_health.return_value = {"available": True}
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=None)
            MockClient.return_value = mock_instance

            result = check_database_health()

            assert result["available"] is True
            mock_instance.get_health.assert_called_once()
