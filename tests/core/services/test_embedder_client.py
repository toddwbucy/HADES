"""Tests for the HADES Embedding Service Client."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.services.embedder_client import EmbedderClient, EmbedderServiceError, embed_texts


class TestEmbedderClientInit:
    """Tests for EmbedderClient initialization."""

    def test_default_values(self):
        """Test client initializes with default values."""
        client = EmbedderClient()
        assert client.socket_path == "/run/hades/embedder.sock"
        assert client.timeout == 30.0
        assert client.fallback_to_local is True

    def test_custom_values(self):
        """Test client accepts custom values."""
        client = EmbedderClient(
            socket_path="/custom/socket.sock",
            timeout=60.0,
            fallback_to_local=False,
        )
        assert client.socket_path == "/custom/socket.sock"
        assert client.timeout == 60.0
        assert client.fallback_to_local is False


class TestEmbedderClientServiceAvailability:
    """Tests for service availability checking."""

    def test_service_available_when_healthy(self):
        """Test service is marked available when health check succeeds."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ready"}

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            assert client.is_service_available() is True

    def test_service_unavailable_when_not_ready(self):
        """Test service is marked unavailable when status is not ready."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "loading"}

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            assert client.is_service_available() is False

    def test_service_unavailable_on_connection_error(self):
        """Test service is marked unavailable on connection error."""
        client = EmbedderClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.get.side_effect = ConnectionError("Connection refused")
            mock_get_client.return_value = mock_http_client

            assert client.is_service_available() is False

    def test_caches_availability_check(self):
        """Test availability check is cached."""
        client = EmbedderClient()
        client._service_available = True

        # Should return cached value without making request
        with patch.object(client, "_get_client") as mock_get_client:
            assert client.is_service_available() is True
            mock_get_client.assert_not_called()

    def test_force_check_bypasses_cache(self):
        """Test force_check bypasses cached availability."""
        client = EmbedderClient()
        client._service_available = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ready"}

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            assert client.is_service_available(force_check=True) is True
            mock_http_client.get.assert_called_once()


class TestEmbedderClientEmbedTexts:
    """Tests for embed_texts functionality."""

    def test_embed_via_service_when_available(self):
        """Test embedding via service when it's available."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "duration_ms": 50.0,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "is_service_available", return_value=True):
            with patch.object(client, "_get_client") as mock_get_client:
                mock_http_client = MagicMock()
                mock_http_client.post.return_value = mock_response
                mock_get_client.return_value = mock_http_client

                result = client.embed_texts(["text1", "text2"])

                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 3)
                np.testing.assert_array_almost_equal(
                    result, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
                )

    def test_fallback_to_local_when_service_unavailable(self):
        """Test falls back to local embedder when service is unavailable."""
        client = EmbedderClient(fallback_to_local=True)

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]])

        with patch.object(client, "is_service_available", return_value=False):
            with patch.object(client, "_get_local_embedder", return_value=mock_embedder):
                result = client.embed_texts(["test"])

                assert isinstance(result, np.ndarray)
                mock_embedder.embed_texts.assert_called_once()

    def test_raises_error_when_no_fallback(self):
        """Test raises error when service unavailable and fallback disabled."""
        client = EmbedderClient(fallback_to_local=False)

        with patch.object(client, "is_service_available", return_value=False):
            with pytest.raises(EmbedderServiceError, match="unavailable"):
                client.embed_texts(["test"])

    def test_empty_texts_returns_empty_array(self):
        """Test empty input returns empty array."""
        client = EmbedderClient()
        result = client.embed_texts([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_passes_task_to_service(self):
        """Test task parameter is passed to service."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1]], "duration_ms": 10}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "is_service_available", return_value=True):
            with patch.object(client, "_get_client") as mock_get_client:
                mock_http_client = MagicMock()
                mock_http_client.post.return_value = mock_response
                mock_get_client.return_value = mock_http_client

                client.embed_texts(["test"], task="retrieval.query")

                call_args = mock_http_client.post.call_args
                assert call_args[1]["json"]["task"] == "retrieval.query"


class TestEmbedderClientConvenienceMethods:
    """Tests for convenience methods."""

    def test_embed_query_uses_retrieval_query_task(self):
        """Test embed_query uses retrieval.query task."""
        client = EmbedderClient()

        with patch.object(client, "embed_texts") as mock_embed:
            mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])

            result = client.embed_query("test query")

            mock_embed.assert_called_once_with(["test query"], task="retrieval.query")
            assert result.shape == (3,)

    def test_embed_query_raises_on_empty_result(self):
        """Test embed_query raises error when embeddings are empty."""
        client = EmbedderClient()

        with patch.object(client, "embed_texts") as mock_embed:
            mock_embed.return_value = np.array([])

            with pytest.raises(EmbedderServiceError, match="empty result"):
                client.embed_query("test query")

    def test_embed_documents_uses_retrieval_passage_task(self):
        """Test embed_documents uses retrieval.passage task."""
        client = EmbedderClient()

        with patch.object(client, "embed_texts") as mock_embed:
            mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])

            client.embed_documents(["doc1", "doc2"])

            mock_embed.assert_called_once()
            call_args = mock_embed.call_args
            assert call_args[1]["task"] == "retrieval.passage"


class TestEmbedderClientShutdown:
    """Tests for shutdown functionality."""

    def test_shutdown_without_token(self):
        """Test shutdown request without token."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = client.shutdown_service()

            assert result is True
            mock_http_client.post.assert_called_once_with("/shutdown", json={})

    def test_shutdown_with_token(self):
        """Test shutdown request with token."""
        client = EmbedderClient()

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = client.shutdown_service(token="secret-token")

            assert result is True
            mock_http_client.post.assert_called_once_with(
                "/shutdown", json={"token": "secret-token"}
            )

    def test_shutdown_returns_false_on_error(self):
        """Test shutdown returns False on connection error."""
        client = EmbedderClient()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.post.side_effect = ConnectionError("Connection refused")
            mock_get_client.return_value = mock_http_client

            result = client.shutdown_service()

            assert result is False


class TestEmbedderClientContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_closes_client(self):
        """Test context manager closes client on exit."""
        with EmbedderClient() as client:
            pass

        # Client should be closed (no error on double close)
        client.close()

    def test_close_releases_resources(self):
        """Test close releases httpx client and local embedder."""
        client = EmbedderClient()
        client._client = MagicMock()
        client._local_embedder = MagicMock()

        client.close()

        assert client._client is None
        assert client._local_embedder is None


class TestEmbedTextsFunction:
    """Tests for the embed_texts convenience function."""

    def test_creates_temporary_client(self):
        """Test embed_texts function creates and uses a temporary client."""
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])

        with patch("core.services.embedder_client.EmbedderClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.embed_texts.return_value = mock_embeddings
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=None)
            MockClient.return_value = mock_client_instance

            result = embed_texts(["test"])

            MockClient.assert_called_once()
            mock_client_instance.embed_texts.assert_called_once()
            np.testing.assert_array_equal(result, mock_embeddings)
