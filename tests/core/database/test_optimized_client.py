"""Unit tests for core.database.arango.optimized_client module."""

from unittest.mock import MagicMock, patch

import pytest

from core.database.arango.optimized_client import (
    ArangoHttp2Client,
    ArangoHttp2Config,
    ArangoHttpError,
)


class TestArangoHttpError:
    """Tests for ArangoHttpError exception."""

    def test_error_is_runtime_error(self) -> None:
        """ArangoHttpError should inherit from RuntimeError."""
        assert issubclass(ArangoHttpError, RuntimeError)

    def test_error_stores_status_code(self) -> None:
        """ArangoHttpError should store status code."""
        error = ArangoHttpError(404, "Document not found")
        assert error.status_code == 404

    def test_error_stores_details(self) -> None:
        """ArangoHttpError should store details dict."""
        details = {"errorNum": 1202, "code": 404}
        error = ArangoHttpError(404, "Not found", details)
        assert error.details == details

    def test_error_default_empty_details(self) -> None:
        """ArangoHttpError should default to empty dict for details."""
        error = ArangoHttpError(500, "Internal error")
        assert error.details == {}

    def test_error_message_includes_status(self) -> None:
        """ArangoHttpError message should include status code."""
        error = ArangoHttpError(403, "Forbidden")
        assert "403" in str(error)
        assert "Forbidden" in str(error)


class TestArangoHttp2Config:
    """Tests for ArangoHttp2Config dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = ArangoHttp2Config()
        assert config.database == "_system"
        assert config.socket_path is None
        assert config.base_url == "http://localhost:8529"
        assert config.username is None
        assert config.password is None
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 30.0
        assert config.write_timeout == 30.0
        assert config.pool_limits is None

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = ArangoHttp2Config(
            database="test_db",
            socket_path="/var/run/arango.sock",
            base_url="http://arango:8529",
            username="root",
            password="secret",
            connect_timeout=10.0,
            read_timeout=60.0,
            write_timeout=60.0,
        )
        assert config.database == "test_db"
        assert config.socket_path == "/var/run/arango.sock"
        assert config.base_url == "http://arango:8529"
        assert config.username == "root"
        assert config.password == "secret"
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.write_timeout == 60.0


class TestArangoHttp2ClientInit:
    """Tests for client initialization."""

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_network_transport_by_default(
        self,
        mock_transport: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Client should use network transport when no socket_path."""
        config = ArangoHttp2Config(database="test")
        ArangoHttp2Client(config)

        # Transport created without uds parameter
        mock_transport.assert_called_once()
        call_kwargs = mock_transport.call_args.kwargs
        assert "uds" not in call_kwargs

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_unix_socket_transport(
        self,
        mock_transport: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Client should use Unix socket transport when socket_path provided."""
        config = ArangoHttp2Config(
            database="test",
            socket_path="/var/run/arango.sock",
        )
        ArangoHttp2Client(config)

        # Transport created with uds parameter
        mock_transport.assert_called_once()
        call_kwargs = mock_transport.call_args.kwargs
        assert call_kwargs["uds"] == "/var/run/arango.sock"

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_authentication(
        self,
        mock_transport: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Client should configure auth when username/password provided."""
        config = ArangoHttp2Config(
            database="test",
            username="root",
            password="secret",
        )
        ArangoHttp2Client(config)

        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["auth"] == ("root", "secret")

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_http2_enabled(
        self,
        mock_transport: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Client should enable HTTP/2."""
        config = ArangoHttp2Config(database="test")
        ArangoHttp2Client(config)

        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["http2"] is True


class TestArangoHttp2ClientContextManager:
    """Tests for context manager protocol."""

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_context_manager_enter_returns_self(
        self,
        mock_transport: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """__enter__ should return client instance."""
        config = ArangoHttp2Config(database="test")
        client = ArangoHttp2Client(config)
        assert client.__enter__() is client

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.database.arango.optimized_client.httpx.HTTPTransport")
    def test_context_manager_exit_closes(
        self,
        mock_transport: MagicMock,
        mock_client_class: MagicMock,
    ) -> None:
        """__exit__ should close the client."""
        mock_httpx_client = MagicMock()
        mock_client_class.return_value = mock_httpx_client

        config = ArangoHttp2Config(database="test")
        client = ArangoHttp2Client(config)
        client.__exit__(None, None, None)

        mock_httpx_client.close.assert_called_once()


class TestGetDocument:
    """Tests for get_document method."""

    @pytest.fixture
    def mock_client(self) -> tuple[ArangoHttp2Client, MagicMock]:
        """Create client with mocked httpx.Client."""
        with patch("core.database.arango.optimized_client.httpx.Client") as mock_client_class:
            with patch("core.database.arango.optimized_client.httpx.HTTPTransport"):
                mock_httpx_client = MagicMock()
                mock_client_class.return_value = mock_httpx_client

                config = ArangoHttp2Config(database="test_db")
                client = ArangoHttp2Client(config)
                return client, mock_httpx_client

    def test_get_document_builds_correct_path(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """get_document should build correct API path."""
        client, httpx_client = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.http_version = "HTTP/1.1"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"_key": "doc1", "value": 42}
        httpx_client.get.return_value = mock_response

        result = client.get_document("my_collection", "doc1")

        httpx_client.get.assert_called_once_with(
            "/_db/test_db/_api/document/my_collection/doc1"
        )
        assert result == {"_key": "doc1", "value": 42}


class TestInsertDocuments:
    """Tests for insert_documents method."""

    @pytest.fixture
    def mock_client(self) -> tuple[ArangoHttp2Client, MagicMock]:
        """Create client with mocked httpx.Client."""
        with patch("core.database.arango.optimized_client.httpx.Client") as mock_client_class:
            with patch("core.database.arango.optimized_client.httpx.HTTPTransport"):
                mock_httpx_client = MagicMock()
                mock_client_class.return_value = mock_httpx_client

                config = ArangoHttp2Config(database="test_db")
                client = ArangoHttp2Client(config)
                return client, mock_httpx_client

    def test_insert_empty_returns_zero_created(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """insert_documents with empty list should return zero created."""
        client, httpx_client = mock_client

        result = client.insert_documents("my_collection", [])
        assert result == {"created": 0}
        httpx_client.post.assert_not_called()

    def test_insert_sends_ndjson(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """insert_documents should send NDJSON payload."""
        client, httpx_client = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.http_version = "HTTP/1.1"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"created": 2}
        httpx_client.post.return_value = mock_response

        docs = [{"_key": "doc1", "val": 1}, {"_key": "doc2", "val": 2}]
        result = client.insert_documents("my_collection", docs)

        assert result == {"created": 2}
        httpx_client.post.assert_called_once()

        call_args = httpx_client.post.call_args
        assert "my_collection" in call_args.args[0]
        assert call_args.kwargs["headers"]["Content-Type"] == "application/x-ndjson"


class TestQuery:
    """Tests for query method."""

    @pytest.fixture
    def mock_client(self) -> tuple[ArangoHttp2Client, MagicMock]:
        """Create client with mocked httpx.Client."""
        with patch("core.database.arango.optimized_client.httpx.Client") as mock_client_class:
            with patch("core.database.arango.optimized_client.httpx.HTTPTransport"):
                mock_httpx_client = MagicMock()
                mock_client_class.return_value = mock_httpx_client

                config = ArangoHttp2Config(database="test_db")
                client = ArangoHttp2Client(config)
                return client, mock_httpx_client

    def test_query_sends_aql(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """query should send AQL to cursor endpoint."""
        client, httpx_client = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.http_version = "HTTP/1.1"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "result": [{"_key": "doc1"}],
            "hasMore": False,
        }
        httpx_client.post.return_value = mock_response

        result = client.query("FOR d IN docs RETURN d")

        assert result == [{"_key": "doc1"}]
        httpx_client.post.assert_called_once()

        call_args = httpx_client.post.call_args
        assert "/_db/test_db/_api/cursor" in call_args.args[0]
        assert call_args.kwargs["json"]["query"] == "FOR d IN docs RETURN d"

    def test_query_handles_pagination(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """query should handle paginated results."""
        client, httpx_client = mock_client

        # First response with hasMore=True
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.http_version = "HTTP/1.1"
        first_response.headers = {"content-type": "application/json"}
        first_response.json.return_value = {
            "result": [{"_key": "doc1"}],
            "hasMore": True,
            "id": "cursor123",
        }

        # Second response with hasMore=False
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.http_version = "HTTP/1.1"
        second_response.headers = {"content-type": "application/json"}
        second_response.json.return_value = {
            "result": [{"_key": "doc2"}],
            "hasMore": False,
        }

        httpx_client.post.return_value = first_response
        httpx_client.put.return_value = second_response

        result = client.query("FOR d IN docs RETURN d")

        assert result == [{"_key": "doc1"}, {"_key": "doc2"}]
        httpx_client.put.assert_called_once()


class TestHandleResponse:
    """Tests for _handle_response method."""

    @pytest.fixture
    def mock_client(self) -> tuple[ArangoHttp2Client, MagicMock]:
        """Create client with mocked httpx.Client."""
        with patch("core.database.arango.optimized_client.httpx.Client") as mock_client_class:
            with patch("core.database.arango.optimized_client.httpx.HTTPTransport"):
                mock_httpx_client = MagicMock()
                mock_client_class.return_value = mock_httpx_client

                config = ArangoHttp2Config(database="test_db")
                client = ArangoHttp2Client(config)
                return client, mock_httpx_client

    def test_raises_error_on_4xx(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """_handle_response should raise ArangoHttpError on 4xx."""
        client, httpx_client = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.http_version = "HTTP/1.1"
        mock_response.json.return_value = {"errorMessage": "Document not found"}
        mock_response.request.method = "GET"
        mock_response.request.url = "http://test"
        httpx_client.get.return_value = mock_response

        with pytest.raises(ArangoHttpError) as exc_info:
            client.get_document("test", "missing")

        assert exc_info.value.status_code == 404

    def test_returns_empty_dict_on_204(
        self,
        mock_client: tuple[ArangoHttp2Client, MagicMock],
    ) -> None:
        """_handle_response should return empty dict on 204."""
        client, httpx_client = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.http_version = "HTTP/1.1"
        httpx_client.get.return_value = mock_response

        result = client.get_document("test", "key")
        assert result == {}
