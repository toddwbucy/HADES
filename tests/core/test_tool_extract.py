"""Tests for core.tools.extract — standalone extraction tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tools.extract import extract_document


class TestExtractDocument:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_document(tmp_path / "nonexistent.pdf")

    @patch("core.extractors.get_extractor")
    def test_returns_expected_keys(self, mock_get_extractor, tmp_path):
        # Create a dummy file
        doc = tmp_path / "test.pdf"
        doc.write_text("hello")

        # Mock the extractor
        mock_result = MagicMock()
        mock_result.text = "extracted text"
        mock_result.tables = [{"id": 1}]
        mock_result.equations = []
        mock_result.images = []
        mock_result.metadata = {"pages": 1}

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        result = extract_document(doc)

        assert result["text"] == "extracted text"
        assert result["tables"] == [{"id": 1}]
        assert result["equations"] == []
        assert result["images"] == []
        assert result["metadata"] == {"pages": 1}
        assert "extraction_time" in result
        assert result["source_path"] == str(doc)

    @patch("core.extractors.get_extractor")
    def test_handles_none_fields(self, mock_get_extractor, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.text = None
        mock_result.tables = None
        mock_result.equations = None
        mock_result.images = None
        mock_result.metadata = None

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        result = extract_document(doc)
        assert result["text"] == ""
        assert result["tables"] == []
        assert result["equations"] == []
        assert result["images"] == []
        assert result["metadata"] == {}

    @patch("core.extractors.get_extractor")
    def test_passes_config_options(self, mock_get_extractor, tmp_path):
        doc = tmp_path / "test.pdf"
        doc.write_text("content")

        mock_result = MagicMock()
        mock_result.text = "text"
        mock_result.tables = []
        mock_result.equations = []
        mock_result.images = []
        mock_result.metadata = {}

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_result
        mock_get_extractor.return_value = mock_extractor

        extract_document(doc, ocr_enabled=True, extract_tables=False)

        # Check config was passed to get_extractor
        call_kwargs = mock_get_extractor.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.ocr_enabled is True
        assert config.extract_tables is False
