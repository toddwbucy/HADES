"""Tests for core.tools.embed — standalone embedding tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.tools.embed import embed_text, embed_texts


class TestEmbedTexts:
    @patch("core.tools.embed.get_client")
    def test_returns_array(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.1] * 2048], dtype=np.float32)
        mock_get_client.return_value = mock_client

        result = embed_texts(["hello"])
        assert result.shape == (1, 2048)
        mock_client.close.assert_called_once()

    @patch("core.tools.embed.get_client")
    def test_passes_task_and_batch_size(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.0] * 2048], dtype=np.float32)
        mock_get_client.return_value = mock_client

        embed_texts(["text"], task="retrieval.query", batch_size=4)
        mock_client.embed_texts.assert_called_once_with(["text"], task="retrieval.query", batch_size=4)

    @patch("core.tools.embed.get_client")
    def test_closes_client_on_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.embed_texts.side_effect = RuntimeError("boom")
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="boom"):
            embed_texts(["fail"])
        mock_client.close.assert_called_once()


class TestEmbedText:
    @patch("core.tools.embed.get_client")
    def test_returns_1d(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.5] * 2048], dtype=np.float32)
        mock_get_client.return_value = mock_client

        result = embed_text("hello")
        assert result.shape == (2048,)


class TestReusableClient:
    def test_embed_texts_with_existing_client(self):
        """When client is provided, it should be used and NOT closed."""
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.1] * 2048], dtype=np.float32)

        result = embed_texts(["hello"], client=mock_client)

        assert result.shape == (1, 2048)
        mock_client.embed_texts.assert_called_once()
        # Client should NOT be closed when provided externally
        mock_client.close.assert_not_called()

    @patch("core.tools.embed.get_client")
    def test_embed_texts_without_client_creates_and_closes(self, mock_get_client):
        """When no client provided, it should create one and close it."""
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.1] * 2048], dtype=np.float32)
        mock_get_client.return_value = mock_client

        result = embed_texts(["hello"])

        assert result.shape == (1, 2048)
        mock_get_client.assert_called_once()  # Client was created
        mock_client.close.assert_called_once()  # And closed

    def test_reusable_client_multiple_calls(self):
        """Client can be reused for multiple embed_texts calls."""
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = np.array([[0.1] * 2048], dtype=np.float32)

        embed_texts(["a"], client=mock_client)
        embed_texts(["b"], client=mock_client)
        embed_texts(["c"], client=mock_client)

        assert mock_client.embed_texts.call_count == 3
        mock_client.close.assert_not_called()  # Never closed
