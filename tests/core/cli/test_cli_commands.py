"""Tests for CLI commands using mocked dependencies."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.cli.commands.arxiv import _metadata_to_dict, get_paper_info, search_arxiv
from core.cli.output import ErrorCode


class TestArxivSearch:
    """Tests for arxiv search command."""

    @patch("core.cli.commands.arxiv.ArXivAPIClient")
    def test_search_returns_results(self, mock_client_class):
        """Test successful arxiv search."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the API response
        mock_entry = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.arxiv_id = "2401.12345"
        mock_metadata.title = "Test Paper"
        mock_metadata.abstract = "This is a test abstract."
        mock_metadata.authors = ["Author One", "Author Two"]
        mock_metadata.categories = ["cs.AI"]
        mock_metadata.primary_category = "cs.AI"
        mock_metadata.published = datetime(2024, 1, 15)
        mock_metadata.updated = datetime(2024, 1, 16)
        mock_metadata.doi = None
        mock_metadata.journal_ref = None
        mock_metadata.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        mock_metadata.has_latex = False

        # Mock _make_request to return XML-like response
        mock_response = MagicMock()
        mock_response.content = b"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry></entry>
        </feed>"""
        mock_client._make_request.return_value = mock_response
        mock_client._parse_entry.return_value = mock_metadata

        # Execute
        response = search_arxiv("test query", max_results=10, categories=None, start_time=0)

        # Verify
        assert response.success is True
        assert response.command == "arxiv.search"
        assert "results" in response.data
        mock_client.close.assert_called_once()

    @patch("core.cli.commands.arxiv.ArXivAPIClient")
    def test_search_handles_exception(self, mock_client_class):
        """Test search handles exceptions gracefully."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client._make_request.side_effect = Exception("Network error")

        response = search_arxiv("test query", max_results=10, categories=None, start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.SEARCH_FAILED.value


class TestArxivInfo:
    """Tests for arxiv info command."""

    @patch("core.cli.commands.arxiv.ArXivAPIClient")
    def test_info_returns_metadata(self, mock_client_class):
        """Test successful paper info retrieval."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.validate_arxiv_id.return_value = True

        mock_metadata = MagicMock()
        mock_metadata.arxiv_id = "2401.12345"
        mock_metadata.title = "Test Paper"
        mock_metadata.abstract = "Test abstract"
        mock_metadata.authors = ["Author"]
        mock_metadata.categories = ["cs.AI"]
        mock_metadata.primary_category = "cs.AI"
        mock_metadata.published = datetime(2024, 1, 15)
        mock_metadata.updated = None
        mock_metadata.doi = None
        mock_metadata.journal_ref = None
        mock_metadata.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        mock_metadata.has_latex = False

        mock_client.get_paper_metadata.return_value = mock_metadata

        response = get_paper_info("2401.12345", start_time=0)

        assert response.success is True
        assert response.command == "arxiv.info"
        assert response.data["arxiv_id"] == "2401.12345"

    @patch("core.cli.commands.arxiv.ArXivAPIClient")
    def test_info_handles_invalid_id(self, mock_client_class):
        """Test info handles invalid arxiv ID."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.validate_arxiv_id.return_value = False

        response = get_paper_info("invalid-id", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.INVALID_ARXIV_ID.value

    @patch("core.cli.commands.arxiv.ArXivAPIClient")
    def test_info_handles_not_found(self, mock_client_class):
        """Test info handles paper not found."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.validate_arxiv_id.return_value = True
        mock_client.get_paper_metadata.return_value = None

        response = get_paper_info("2401.99999", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.PAPER_NOT_FOUND.value


class TestMetadataConversion:
    """Tests for metadata to dict conversion."""

    def test_metadata_to_dict_converts_all_fields(self):
        """Test that metadata is properly converted to dict."""
        mock_metadata = MagicMock()
        mock_metadata.arxiv_id = "2401.12345"
        mock_metadata.title = "Test Title"
        mock_metadata.abstract = "Test Abstract"
        mock_metadata.authors = ["Author A", "Author B"]
        mock_metadata.categories = ["cs.AI", "cs.LG"]
        mock_metadata.primary_category = "cs.AI"
        mock_metadata.published = datetime(2024, 1, 15, 10, 30)
        mock_metadata.updated = datetime(2024, 2, 1, 12, 0)
        mock_metadata.doi = "10.1234/test"
        mock_metadata.journal_ref = "Test Journal 2024"
        mock_metadata.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        mock_metadata.has_latex = True

        result = _metadata_to_dict(mock_metadata)

        assert result["arxiv_id"] == "2401.12345"
        assert result["title"] == "Test Title"
        assert result["abstract"] == "Test Abstract"
        assert result["authors"] == ["Author A", "Author B"]
        assert result["categories"] == ["cs.AI", "cs.LG"]
        assert result["primary_category"] == "cs.AI"
        assert "2024-01-15" in result["published"]
        assert "2024-02-01" in result["updated"]
        assert result["doi"] == "10.1234/test"
        assert result["journal_ref"] == "Test Journal 2024"
        assert result["pdf_url"] == "https://arxiv.org/pdf/2401.12345.pdf"
        assert result["has_latex"] is True
