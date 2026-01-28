"""Tests for abstract search CLI commands."""

from unittest.mock import MagicMock, patch

import numpy as np

from core.cli.commands.arxiv import (
    _compute_rocchio_centroid,
    find_similar,
    refine_search,
    search_abstracts,
    search_abstracts_bulk,
)
from core.cli.output import ErrorCode


class TestAbstractSearch:
    """Tests for abstract search command."""

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_search_returns_results(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test successful abstract search."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.12345",
                    "title": "Test Paper",
                    "similarity": 0.95,
                    "abstract": "Test abstract...",
                    "categories": ["cs.AI"],
                    "local": False,
                    "local_chunks": None,
                }
            ],
            100000,  # total_processed
        )

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

    @patch("core.cli.commands.arxiv.get_config")
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

    @patch("core.cli.commands.arxiv.get_config")
    def test_search_rejects_invalid_limit(self, mock_get_config):
        """Test search rejects limit <= 0."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        response = search_abstracts(
            query="test query",
            limit=0,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "limit must be >= 1" in response.error["message"]

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_search_with_category_filter(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test search with category filter."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)
        mock_search.return_value = ([], 0)

        search_abstracts(
            query="test query",
            limit=10,
            start_time=0,
            category="cs.AI",
        )

        # Verify category was passed to search
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["category_filter"] == "cs.AI"


class TestSearchResultsFormat:
    """Tests for search result formatting."""

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_results_include_local_status(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that results include local availability status."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        # One local, one not local
        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.12345",
                    "title": "Local Paper",
                    "similarity": 0.95,
                    "abstract": "...",
                    "categories": ["cs.AI"],
                    "local": True,
                    "local_chunks": 47,
                },
                {
                    "arxiv_id": "2401.67890",
                    "title": "Remote Paper",
                    "similarity": 0.90,
                    "abstract": "...",
                    "categories": ["cs.LG"],
                    "local": False,
                    "local_chunks": None,
                },
            ],
            100000,  # total_processed
        )

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

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_results_include_total_searched(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that response includes total embeddings searched."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)

        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.12345",
                    "title": "Test",
                    "similarity": 0.95,
                    "abstract": "...",
                    "categories": [],
                    "local": False,
                    "local_chunks": None,
                }
            ],
            2826761,  # total_processed
        )

        response = search_abstracts(
            query="test",
            limit=10,
            start_time=0,
        )

        assert response.data["total_searched"] == 2826761


class TestHybridSearch:
    """Tests for hybrid search functionality."""

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_hybrid_search_passes_query_to_search(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that hybrid mode passes query for keyword matching."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)
        mock_search.return_value = ([], 0)

        search_abstracts(
            query="transformer attention",
            limit=10,
            start_time=0,
            hybrid=True,
        )

        # Verify hybrid_query was passed
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["hybrid_query"] == "transformer attention"

    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_hybrid_search_mode_in_response(
        self, mock_get_config, mock_get_embedding, mock_search
    ):
        """Test that response indicates hybrid mode."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)
        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.12345",
                    "title": "Test",
                    "similarity": 0.95,
                    "abstract": "...",
                    "categories": [],
                    "local": False,
                    "local_chunks": None,
                    "keyword_score": 0.8,
                    "combined_score": 0.91,
                }
            ],
            100000,  # total_processed
        )

        response = search_abstracts(
            query="test",
            limit=10,
            start_time=0,
            hybrid=True,
        )

        assert response.data["mode"] == "hybrid"


class TestFindSimilar:
    """Tests for find similar papers command."""

    @patch("core.cli.commands.arxiv._get_paper_info")
    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_paper_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_similar_returns_results(
        self, mock_get_config, mock_get_embedding, mock_search, mock_get_info
    ):
        """Test successful similar search."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        # Return a mock embedding
        mock_get_embedding.return_value = np.zeros(2048)

        # Return similar papers (excluding the source)
        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.67890",
                    "title": "Similar Paper",
                    "similarity": 0.92,
                    "abstract": "...",
                    "categories": ["cs.AI"],
                    "local": False,
                    "local_chunks": None,
                }
            ],
            100000,  # total_processed
        )

        mock_get_info.return_value = {"title": "Source Paper"}

        response = find_similar(
            arxiv_id="2401.12345",
            limit=10,
            start_time=0,
        )

        assert response.success is True
        assert response.command == "abstract.similar"
        assert response.data["source_paper"]["arxiv_id"] == "2401.12345"
        assert len(response.data["results"]) == 1
        assert response.data["results"][0]["arxiv_id"] == "2401.67890"

    @patch("core.cli.commands.arxiv._get_paper_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_similar_paper_not_found(self, mock_get_config, mock_get_embedding):
        """Test similar search when paper not in database."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        # Paper not found
        mock_get_embedding.return_value = None

        response = find_similar(
            arxiv_id="9999.99999",
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.PAPER_NOT_FOUND.value

    @patch("core.cli.commands.arxiv._get_paper_info")
    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._get_paper_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_similar_excludes_source_paper(
        self, mock_get_config, mock_get_embedding, mock_search, mock_get_info
    ):
        """Test that similar search excludes the source paper."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_get_embedding.return_value = np.zeros(2048)
        mock_search.return_value = ([], 0)
        mock_get_info.return_value = {"title": "Source Paper"}

        find_similar(
            arxiv_id="2401.12345",
            limit=10,
            start_time=0,
        )

        # Verify exclude_arxiv_id was passed
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["exclude_arxiv_id"] == "2401.12345"

    @patch("core.cli.commands.arxiv.get_config")
    def test_similar_handles_config_error(self, mock_get_config):
        """Test similar handles config errors."""
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = find_similar(
            arxiv_id="2401.12345",
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value


class TestBulkSearch:
    """Tests for bulk search command."""

    @patch("core.cli.commands.arxiv._search_abstract_embeddings_bulk")
    @patch("core.cli.commands.arxiv._get_bulk_query_embeddings")
    @patch("core.cli.commands.arxiv.get_config")
    def test_bulk_search_returns_results(
        self, mock_get_config, mock_get_embeddings, mock_search
    ):
        """Test successful bulk search."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embeddings.return_value = np.zeros((2, 2048))

        mock_search.return_value = (
            {
                "query1": [
                    {
                        "arxiv_id": "2401.12345",
                        "title": "Paper 1",
                        "similarity": 0.95,
                        "abstract": "...",
                        "categories": ["cs.AI"],
                        "local": False,
                        "local_chunks": None,
                    }
                ],
                "query2": [
                    {
                        "arxiv_id": "2401.67890",
                        "title": "Paper 2",
                        "similarity": 0.90,
                        "abstract": "...",
                        "categories": ["cs.LG"],
                        "local": True,
                        "local_chunks": 42,
                    }
                ],
            },
            100000,
        )

        response = search_abstracts_bulk(
            queries=["query1", "query2"],
            limit=10,
            start_time=0,
        )

        assert response.success is True
        assert response.command == "abstract.search-bulk"
        assert response.data["query_count"] == 2
        assert "query1" in response.data["results_by_query"]
        assert "query2" in response.data["results_by_query"]
        assert response.data["total_searched"] == 100000

    @patch("core.cli.commands.arxiv.get_config")
    def test_bulk_search_rejects_empty_queries(self, mock_get_config):
        """Test bulk search rejects empty queries list."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        response = search_abstracts_bulk(
            queries=[],
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "empty" in response.error["message"]

    @patch("core.cli.commands.arxiv.get_config")
    def test_bulk_search_rejects_invalid_limit(self, mock_get_config):
        """Test bulk search rejects invalid limit."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        response = search_abstracts_bulk(
            queries=["test"],
            limit=0,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "limit" in response.error["message"]

    @patch("core.cli.commands.arxiv.get_config")
    def test_bulk_search_handles_config_error(self, mock_get_config):
        """Test bulk search handles config errors."""
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = search_abstracts_bulk(
            queries=["test"],
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.cli.commands.arxiv._search_abstract_embeddings_bulk")
    @patch("core.cli.commands.arxiv._get_bulk_query_embeddings")
    @patch("core.cli.commands.arxiv.get_config")
    def test_bulk_search_passes_category_filter(
        self, mock_get_config, mock_get_embeddings, mock_search
    ):
        """Test that category filter is passed through."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_embeddings.return_value = np.zeros((1, 2048))
        mock_search.return_value = ({"test": []}, 0)

        search_abstracts_bulk(
            queries=["test"],
            limit=10,
            start_time=0,
            category="cs.AI",
        )

        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["category_filter"] == "cs.AI"


class TestRefineSearch:
    """Tests for relevance feedback / refine search command."""

    @patch("core.cli.commands.arxiv._get_paper_info")
    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._compute_rocchio_centroid")
    @patch("core.cli.commands.arxiv._get_multiple_paper_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_refine_returns_results(
        self,
        mock_get_config,
        mock_get_query_emb,
        mock_get_paper_embs,
        mock_rocchio,
        mock_search,
        mock_get_info,
    ):
        """Test successful refine search."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_query_emb.return_value = np.ones(2048)
        mock_get_paper_embs.return_value = [np.ones(2048) * 0.5]
        mock_rocchio.return_value = np.ones(2048) * 0.8

        mock_search.return_value = (
            [
                {
                    "arxiv_id": "2401.99999",
                    "title": "Refined Result",
                    "similarity": 0.92,
                    "abstract": "...",
                    "categories": ["cs.AI"],
                    "local": False,
                    "local_chunks": None,
                }
            ],
            100000,
        )

        mock_get_info.return_value = {"title": "Positive Exemplar"}

        response = refine_search(
            query="transformer attention",
            positive_ids=["2401.12345"],
            limit=10,
            start_time=0,
        )

        assert response.success is True
        assert response.command == "abstract.refine"
        assert response.data["mode"] == "relevance_feedback"
        assert len(response.data["positive_exemplars"]) == 1
        assert len(response.data["results"]) == 1

    @patch("core.cli.commands.arxiv.get_config")
    def test_refine_rejects_empty_positive_ids(self, mock_get_config):
        """Test refine rejects empty positive exemplars."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        response = refine_search(
            query="test",
            positive_ids=[],
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "positive exemplar" in response.error["message"].lower()

    @patch("core.cli.commands.arxiv._get_multiple_paper_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_refine_handles_missing_papers(
        self, mock_get_config, mock_get_query_emb, mock_get_paper_embs
    ):
        """Test refine handles when no positive papers found."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_query_emb.return_value = np.ones(2048)
        mock_get_paper_embs.return_value = []  # No papers found

        response = refine_search(
            query="test",
            positive_ids=["9999.99999"],
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.PAPER_NOT_FOUND.value

    @patch("core.cli.commands.arxiv.get_config")
    def test_refine_handles_config_error(self, mock_get_config):
        """Test refine handles config errors."""
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = refine_search(
            query="test",
            positive_ids=["2401.12345"],
            limit=10,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.cli.commands.arxiv._get_paper_info")
    @patch("core.cli.commands.arxiv._search_abstract_embeddings")
    @patch("core.cli.commands.arxiv._compute_rocchio_centroid")
    @patch("core.cli.commands.arxiv._get_multiple_paper_embeddings")
    @patch("core.cli.commands.arxiv._get_query_embedding")
    @patch("core.cli.commands.arxiv.get_config")
    def test_refine_passes_weights_to_rocchio(
        self,
        mock_get_config,
        mock_get_query_emb,
        mock_get_paper_embs,
        mock_rocchio,
        mock_search,
        mock_get_info,
    ):
        """Test that custom weights are passed to Rocchio computation."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_get_query_emb.return_value = np.ones(2048)
        mock_get_paper_embs.return_value = [np.ones(2048)]
        mock_rocchio.return_value = np.ones(2048)
        mock_search.return_value = ([], 0)
        mock_get_info.return_value = {"title": "Test"}

        refine_search(
            query="test",
            positive_ids=["2401.12345"],
            limit=10,
            start_time=0,
            alpha=0.5,
            beta=1.0,
            gamma=0.25,
        )

        # Verify Rocchio was called with custom weights
        mock_rocchio.assert_called_once()
        call_kwargs = mock_rocchio.call_args[1]
        assert call_kwargs["alpha"] == 0.5
        assert call_kwargs["beta"] == 1.0
        assert call_kwargs["gamma"] == 0.25


class TestRocchioCentroid:
    """Tests for Rocchio centroid computation."""

    def test_rocchio_with_positive_only(self):
        """Test Rocchio with only positive exemplars."""
        query = np.array([1.0, 0.0, 0.0])
        positive = [np.array([0.0, 1.0, 0.0])]

        result = _compute_rocchio_centroid(query, positive, alpha=1.0, beta=1.0)

        # q' = 1.0 * [1,0,0] + 1.0 * [0,1,0] = [1,1,0]
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rocchio_with_negative(self):
        """Test Rocchio with positive and negative exemplars."""
        query = np.array([1.0, 0.0, 0.0])
        positive = [np.array([0.0, 1.0, 0.0])]
        negative = [np.array([0.0, 0.0, 1.0])]

        result = _compute_rocchio_centroid(
            query, positive, negative, alpha=1.0, beta=1.0, gamma=1.0
        )

        # q' = 1.0 * [1,0,0] + 1.0 * [0,1,0] - 1.0 * [0,0,1] = [1,1,-1]
        expected = np.array([1.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rocchio_averages_multiple_positives(self):
        """Test Rocchio averages multiple positive exemplars."""
        query = np.array([1.0, 0.0])
        positive = [
            np.array([0.0, 2.0]),
            np.array([0.0, 4.0]),
        ]

        result = _compute_rocchio_centroid(query, positive, alpha=1.0, beta=1.0)

        # mean(positive) = [0, 3]
        # q' = 1.0 * [1,0] + 1.0 * [0,3] = [1,3]
        expected = np.array([1.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rocchio_respects_weights(self):
        """Test Rocchio respects alpha and beta weights."""
        query = np.array([2.0, 0.0])
        positive = [np.array([0.0, 2.0])]

        result = _compute_rocchio_centroid(query, positive, alpha=0.5, beta=0.25)

        # q' = 0.5 * [2,0] + 0.25 * [0,2] = [1,0.5]
        expected = np.array([1.0, 0.5])
        np.testing.assert_array_almost_equal(result, expected)
