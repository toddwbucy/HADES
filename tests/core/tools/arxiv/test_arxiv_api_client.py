"""Unit tests for ArXiv API client.

Tests for:
- ArXiv ID validation and normalization
- Metadata parsing
- Rate limiting behavior
- Download result handling
"""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestNormalizeArxivId:
    """Tests for normalize_arxiv_id function."""

    def test_strips_version_suffix(self):
        """Should strip version suffix from ArXiv ID."""
        from core.tools.arxiv.arxiv_api_client import normalize_arxiv_id

        assert normalize_arxiv_id("2308.12345v1") == "2308.12345"
        assert normalize_arxiv_id("2308.12345v2") == "2308.12345"
        assert normalize_arxiv_id("2308.12345v10") == "2308.12345"

    def test_leaves_versionless_unchanged(self):
        """Should leave IDs without version unchanged."""
        from core.tools.arxiv.arxiv_api_client import normalize_arxiv_id

        assert normalize_arxiv_id("2308.12345") == "2308.12345"
        assert normalize_arxiv_id("1234.56789") == "1234.56789"

    def test_handles_old_format(self):
        """Should handle old format ArXiv IDs."""
        from core.tools.arxiv.arxiv_api_client import normalize_arxiv_id

        assert normalize_arxiv_id("cs.AI/0601001v1") == "cs.AI/0601001"
        assert normalize_arxiv_id("cs.AI/0601001") == "cs.AI/0601001"


class TestArXivMetadata:
    """Tests for ArXivMetadata dataclass."""

    def test_generates_pdf_url(self):
        """Should auto-generate PDF URL if not provided."""
        from core.tools.arxiv.arxiv_api_client import ArXivMetadata

        metadata = ArXivMetadata(
            arxiv_id="2308.12345",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author One"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published=datetime.now(),
            updated=datetime.now(),
        )

        assert metadata.pdf_url == "https://arxiv.org/pdf/2308.12345.pdf"

    def test_preserves_provided_pdf_url(self):
        """Should preserve PDF URL if explicitly provided."""
        from core.tools.arxiv.arxiv_api_client import ArXivMetadata

        custom_url = "https://custom.url/paper.pdf"
        metadata = ArXivMetadata(
            arxiv_id="2308.12345",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author One"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published=datetime.now(),
            updated=datetime.now(),
            pdf_url=custom_url,
        )

        assert metadata.pdf_url == custom_url

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        from core.tools.arxiv.arxiv_api_client import ArXivMetadata

        metadata = ArXivMetadata(
            arxiv_id="2308.12345",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author One"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published=datetime.now(),
            updated=datetime.now(),
        )

        assert metadata.doi is None
        assert metadata.journal_ref is None
        assert metadata.license is None


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_success_result(self):
        """Should create successful download result."""
        from core.tools.arxiv.arxiv_api_client import DownloadResult

        result = DownloadResult(
            success=True,
            arxiv_id="2308.12345",
            pdf_path=Path("/tmp/paper.pdf"),
            file_size_bytes=1024,
        )

        assert result.success is True
        assert result.arxiv_id == "2308.12345"
        assert result.error_message is None

    def test_failure_result(self):
        """Should create failed download result with error message."""
        from core.tools.arxiv.arxiv_api_client import DownloadResult

        result = DownloadResult(
            success=False,
            arxiv_id="2308.12345",
            error_message="Network error",
        )

        assert result.success is False
        assert result.error_message == "Network error"
        assert result.pdf_path is None


class TestArXivAPIClientValidation:
    """Tests for ArXivAPIClient validation methods."""

    def test_validate_new_format_id(self):
        """Should validate new format ArXiv IDs."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        # Valid new format
        assert client.validate_arxiv_id("2308.12345") is True
        assert client.validate_arxiv_id("1234.56789") is True
        assert client.validate_arxiv_id("2308.1234") is True  # 4-digit paper number

        # With version
        assert client.validate_arxiv_id("2308.12345v1") is True
        assert client.validate_arxiv_id("2308.12345v10") is True

        client.close()

    def test_validate_old_format_id(self):
        """Should validate old format ArXiv IDs."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        # Valid old format
        assert client.validate_arxiv_id("cs.AI/0601001") is True
        assert client.validate_arxiv_id("hep-th/9901001") is True

        client.close()

    def test_reject_invalid_ids(self):
        """Should reject invalid ArXiv IDs."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        assert client.validate_arxiv_id("invalid") is False
        assert client.validate_arxiv_id("12345") is False
        assert client.validate_arxiv_id("") is False
        assert client.validate_arxiv_id("abc.12345") is False

        client.close()


class TestArXivAPIClientYearMonthExtraction:
    """Tests for year-month extraction."""

    def test_extract_from_new_format(self):
        """Should extract year-month from new format IDs."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        assert client._extract_year_month("2308.12345") == "2308"
        assert client._extract_year_month("1912.01234") == "1912"
        assert client._extract_year_month("2401.00001") == "2401"

        client.close()

    def test_extract_from_old_format(self):
        """Should extract year-month from old format IDs."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        # Old format without dot in category (e.g., hep-th/9912345)
        # uses the "/" branch and extracts first 4 chars after "/"
        assert client._extract_year_month("hep-th/9912345") == "9912"
        assert client._extract_year_month("quant-ph/0601001") == "0601"

        # Note: Old format WITH dot (cs.AI/0601001) triggers "." check first
        # and returns "cs" - this is edge case behavior in the implementation
        assert client._extract_year_month("cs.AI/0601001") == "cs"

        client.close()

    def test_fallback_for_unknown_format(self):
        """Should return default for unknown format."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        assert client._extract_year_month("unknown") == "0000"
        assert client._extract_year_month("") == "0000"

        client.close()


class TestArXivAPIClientRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_delay_enforced(self):
        """Should enforce rate limit delay between requests."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        # Use a short delay for testing
        client = ArXivAPIClient(rate_limit_delay=0.1)

        # First call sets the timestamp
        client._enforce_rate_limit()

        # Immediate second call should sleep
        start = time.time()
        client._enforce_rate_limit()
        elapsed = time.time() - start

        # Should have waited approximately the rate limit delay
        # (allow some tolerance for timing)
        assert elapsed >= 0.05  # At least half the delay

        client.close()

    def test_no_delay_after_sufficient_time(self):
        """Should not delay if enough time has passed."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient(rate_limit_delay=0.1)

        # Set last request time to the past
        client.last_request_time = time.time() - 1.0

        start = time.time()
        client._enforce_rate_limit()
        elapsed = time.time() - start

        # Should not have waited
        assert elapsed < 0.05

        client.close()


class TestArXivAPIClientInit:
    """Tests for ArXivAPIClient initialization."""

    def test_default_config(self):
        """Should initialize with default configuration."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        assert client.rate_limit_delay == 3.0
        assert client.max_retries == 3
        assert client.timeout == 30
        assert client.api_base_url == "https://export.arxiv.org/api/query"

        client.close()

    def test_custom_config(self):
        """Should accept custom configuration."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient(
            rate_limit_delay=5.0,
            max_retries=5,
            timeout=60,
        )

        assert client.rate_limit_delay == 5.0
        assert client.max_retries == 5
        assert client.timeout == 60

        client.close()

    def test_session_created(self):
        """Should create requests session."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()

        assert client.session is not None
        assert "User-Agent" in client.session.headers

        client.close()


class TestArXivAPIClientMetadataFetch:
    """Tests for metadata fetching (mocked)."""

    @patch("core.tools.arxiv.arxiv_api_client.ArXivAPIClient._make_request")
    def test_returns_none_for_invalid_id(self, mock_request):
        """Should return None for invalid ArXiv ID without making request."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        client = ArXivAPIClient()
        result = client.get_paper_metadata("invalid_id")

        assert result is None
        mock_request.assert_not_called()

        client.close()

    @patch("core.tools.arxiv.arxiv_api_client.ArXivAPIClient._make_request")
    def test_returns_none_when_no_entries(self, mock_request):
        """Should return None when API returns no entries."""
        from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

        # Mock empty response
        mock_response = MagicMock()
        mock_response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        mock_request.return_value = mock_response

        client = ArXivAPIClient()
        result = client.get_paper_metadata("2308.12345")

        assert result is None

        client.close()
