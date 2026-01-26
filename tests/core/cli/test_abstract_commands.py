"""Tests for abstract search CLI commands."""

from unittest.mock import MagicMock, patch

import numpy as np

from core.cli.commands.abstract import ingest_from_abstract, search_abstracts
from core.cli.output import ErrorCode


class TestAbstractSearch:
    """Tests for abstract search command."""

    @patch("core.cli.commands.abstract._search_abstract_embeddings")
    @patch("core.cli.commands.abstract._get_query_embedding")
    @patch("core.cli.commands.abstract.get_config")
    def test_search_returns_results(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test successful abstract search."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        mock_search.return_value = [
            {
                "arxiv_id": "2401.12345",
                "title": "Test Paper",
                "similarity": 0.95,
                "abstract": "Test abstract...",
                "categories": ["cs.AI"],
                "local": False,
                "local_chunks": None,
                "total_searched": 100000,
            }
        ]

        # Execute
        response = search_abstracts(
            query="test query",
            limit=10,
            start_time=0,
            category=None,
        )

        # Verify
        assert response.success is True
        assert response.command == "abstract.search"
        assert "results" in response.data
        assert len(response.data["results"]) == 1
        assert response.data["results"][0]["arxiv_id"] == "2401.12345"
        assert response.data["results"][0]["local"] is False

    @patch("core.cli.commands.abstract.get_config")
    def test_search_handles_config_error(self, mock_get_config):
        """Test search handles config errors."""
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = search_abstracts(
            query="test query",
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.cli.commands.abstract._search_abstract_embeddings")
    @patch("core.cli.commands.abstract._get_query_embedding")
    @patch("core.cli.commands.abstract.get_config")
    def test_search_with_category_filter(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test search with category filter."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)
        mock_search.return_value = []

        response = search_abstracts(
            query="test query",
            limit=10,
            start_time=0,
            category="cs.AI",
        )

        # Verify category was passed to search
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["category_filter"] == "cs.AI"


class TestAbstractIngest:
    """Tests for abstract ingest command."""

    @patch("core.cli.commands.ingest.ingest_papers")
    def test_ingest_delegates_to_ingest_papers(self, mock_ingest):
        """Test that abstract ingest delegates to existing ingest logic."""
        mock_ingest.return_value = MagicMock(
            success=True,
            command="ingest",
            data={"ingested": 1, "failed": 0},
        )

        response = ingest_from_abstract(
            arxiv_ids=["2401.12345"],
            force=False,
            start_time=0,
        )

        # Verify delegation
        mock_ingest.assert_called_once_with(
            arxiv_ids=["2401.12345"],
            pdf_paths=None,
            force=False,
            start_time=0,
        )

    @patch("core.cli.commands.ingest.ingest_papers")
    def test_ingest_passes_force_flag(self, mock_ingest):
        """Test that force flag is passed through."""
        mock_ingest.return_value = MagicMock(success=True)

        ingest_from_abstract(
            arxiv_ids=["2401.12345"],
            force=True,
            start_time=0,
        )

        call_args = mock_ingest.call_args
        assert call_args[1]["force"] is True


class TestSearchResultsFormat:
    """Tests for search result formatting."""

    @patch("core.cli.commands.abstract._search_abstract_embeddings")
    @patch("core.cli.commands.abstract._get_query_embedding")
    @patch("core.cli.commands.abstract.get_config")
    def test_results_include_local_status(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that results include local availability status."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        # One local, one not local
        mock_search.return_value = [
            {
                "arxiv_id": "2401.12345",
                "title": "Local Paper",
                "similarity": 0.95,
                "abstract": "...",
                "categories": ["cs.AI"],
                "local": True,
                "local_chunks": 47,
                "total_searched": 100000,
            },
            {
                "arxiv_id": "2401.67890",
                "title": "Remote Paper",
                "similarity": 0.90,
                "abstract": "...",
                "categories": ["cs.LG"],
                "local": False,
                "local_chunks": None,
                "total_searched": 100000,
            },
        ]

        response = search_abstracts(
            query="test",
            limit=10,
            start_time=0,
        )

        results = response.data["results"]
        assert results[0]["local"] is True
        assert results[0]["local_chunks"] == 47
        assert results[1]["local"] is False
        assert results[1]["local_chunks"] is None

    @patch("core.cli.commands.abstract._search_abstract_embeddings")
    @patch("core.cli.commands.abstract._get_query_embedding")
    @patch("core.cli.commands.abstract.get_config")
    def test_results_include_total_searched(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that response includes total embeddings searched."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        mock_search.return_value = [
            {
                "arxiv_id": "2401.12345",
                "title": "Test",
                "similarity": 0.95,
                "abstract": "...",
                "categories": [],
                "local": False,
                "local_chunks": None,
                "total_searched": 2826761,
            }
        ]

        response = search_abstracts(
            query="test",
            limit=10,
            start_time=0,
        )

        assert response.data["total_searched"] == 2826761
