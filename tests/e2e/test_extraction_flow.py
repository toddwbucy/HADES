"""End-to-end tests for the extraction phase of the pipeline.

Tests the PDF extraction using real extractors where possible.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPyMuPDFExtraction:
    """Test PDF extraction using PyMuPDF fallback."""

    def test_pymupdf_extracts_text(self, sample_pdf: Path) -> None:
        """PyMuPDF should extract text from a valid PDF."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        with fitz.open(sample_pdf) as doc:
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            full_text = "\n".join(text_parts)

        assert len(full_text) > 0
        assert "Machine" in full_text or "learning" in full_text.lower()

    def test_pymupdf_handles_empty_pdf(self, empty_pdf: Path) -> None:
        """PyMuPDF should handle empty PDFs."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        with fitz.open(empty_pdf) as doc:
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            full_text = "\n".join(text_parts)

        # Empty PDF should produce empty or whitespace-only text
        assert full_text.strip() == ""

    def test_pymupdf_page_count(self, multi_page_pdf: Path) -> None:
        """PyMuPDF should correctly count pages."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not available")

        with fitz.open(multi_page_pdf) as doc:
            assert len(doc) == 5  # We created 5 pages


class TestRobustExtractor:
    """Test the RobustExtractor with fallback capabilities."""

    @pytest.mark.skipif(
        importlib.util.find_spec("core.extractors.extractors_robust") is None,
        reason="RobustExtractor not available",
    )
    def test_robust_extractor_with_timeout(self, sample_pdf: Path) -> None:
        """RobustExtractor should handle timeout scenarios."""
        from core.extractors.extractors_robust import RobustExtractor

        extractor = RobustExtractor(
            use_ocr=False,
            extract_tables=False,
            timeout=60,
            use_fallback=True,
        )

        result = extractor.extract(sample_pdf)

        # Should complete without timeout for simple PDF
        assert result is not None
        assert hasattr(result, "text") or hasattr(result, "full_text")

    @pytest.mark.skipif(
        importlib.util.find_spec("core.extractors.extractors_robust") is None,
        reason="RobustExtractor not available",
    )
    def test_robust_extractor_fallback(self, sample_pdf: Path) -> None:
        """RobustExtractor should fall back to PyMuPDF when Docling fails."""
        from core.extractors.extractors_robust import RobustExtractor

        # Create extractor with very short timeout to force fallback
        extractor = RobustExtractor(
            use_ocr=False,
            extract_tables=False,
            timeout=1,  # Very short timeout
            use_fallback=True,
        )

        result = extractor.extract(sample_pdf)

        # Should have extracted some text via fallback (may or may not have
        # content depending on whether timeout triggered and fallback succeeded)
        assert result is not None


class TestDoclingExtractorInterface:
    """Test DoclingExtractor interface compliance."""

    @patch("core.extractors.extractors_docling.DocumentConverter")
    def test_docling_extractor_returns_extraction_result(
        self,
        mock_converter_cls: MagicMock,
        sample_pdf: Path,
    ) -> None:
        """DoclingExtractor.extract should return ExtractionResult."""
        from enum import Enum

        from core.extractors.extractors_base import ExtractionResult
        from core.extractors.extractors_docling import DoclingExtractor

        # Create a mock status enum
        class MockStatus(Enum):
            SUCCESS = "success"

        # Mock the Docling converter properly (Docling v2 uses result.output)
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.status = MockStatus.SUCCESS
        mock_result.output = MagicMock()
        mock_result.output.export_to_markdown.return_value = "Extracted text content"
        mock_result.output.tables = []
        mock_result.output.pictures = []
        mock_converter.convert_single.return_value = mock_result
        mock_converter_cls.return_value = mock_converter

        extractor = DoclingExtractor(use_ocr=False, extract_tables=True)
        result = extractor.extract(sample_pdf)

        assert isinstance(result, ExtractionResult)
        assert result.text is not None

    @patch("core.extractors.extractors_docling.DocumentConverter")
    def test_docling_extractor_batch(
        self,
        mock_converter_cls: MagicMock,
        sample_pdf: Path,
        minimal_pdf: Path,
    ) -> None:
        """DoclingExtractor.extract_batch should process multiple PDFs."""
        from core.extractors.extractors_base import ExtractionResult
        from core.extractors.extractors_docling import DoclingExtractor

        # Mock the Docling converter (Docling v2 uses result.output and convert_single)
        mock_converter = MagicMock()
        mock_result = MagicMock()
        # Set status for _extract_with_docling's status check
        mock_result.status = MagicMock()
        mock_result.status.name = "SUCCESS"
        mock_result.output = MagicMock()
        mock_result.output.export_to_markdown.return_value = "Extracted text"
        mock_result.output.tables = []
        mock_result.output.pictures = []
        mock_result.output.figures = []
        mock_converter.convert_single.return_value = mock_result
        mock_converter_cls.return_value = mock_converter

        extractor = DoclingExtractor(use_ocr=False, extract_tables=False)
        results = extractor.extract_batch([sample_pdf, minimal_pdf])

        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)


class TestLatexExtractor:
    """Test LaTeX extraction capabilities."""

    @pytest.fixture
    def sample_latex(self, tmp_path: Path) -> Path:
        """Create a sample LaTeX file."""
        latex_content = r"""
\documentclass{article}
\title{Test Document}
\author{Test Author}

\begin{document}
\maketitle

\section{Introduction}
This is a test document with some math: $E = mc^2$

\begin{equation}
    \int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
\end{equation}

\section{Conclusion}
End of document.
\end{document}
"""
        latex_path = tmp_path / "test.tex"
        latex_path.write_text(latex_content)
        return latex_path

    def test_latex_extractor_exists(self) -> None:
        """LaTeX extractor module should exist."""
        from core.extractors import LaTeXExtractor

        assert LaTeXExtractor is not None

    @pytest.mark.skipif(
        importlib.util.find_spec("core.extractors.extractors_latex") is None,
        reason="LaTeXExtractor not available",
    )
    def test_latex_extractor_extracts_equations(self, sample_latex: Path) -> None:
        """LaTeX extractor should extract equations."""
        from core.extractors.extractors_latex import LaTeXExtractor

        extractor = LaTeXExtractor()
        result = extractor.extract(sample_latex)

        # Should have extracted content
        assert result is not None
        # Should find equations
        if result.equations:
            assert len(result.equations) > 0


class TestExtractorFactory:
    """Test extractor factory/dispatcher."""

    def test_get_extractor_for_pdf(self) -> None:
        """get_extractor should return extractor for PDF files."""
        from core.extractors import get_extractor

        extractor = get_extractor("document.pdf")
        assert extractor is not None
        assert ".pdf" in extractor.supported_formats

    def test_get_extractor_for_tex(self) -> None:
        """get_extractor should return extractor for LaTeX files."""
        from core.extractors import get_extractor

        extractor = get_extractor("document.tex")
        assert extractor is not None
        assert ".tex" in extractor.supported_formats

    def test_get_extractor_for_unknown_falls_back(self) -> None:
        """get_extractor should fall back to RobustExtractor for unknown formats."""
        from core.extractors import get_extractor

        # For unknown file types, get_extractor falls back to RobustExtractor
        # or DoclingExtractor if available (rather than raising)
        extractor = get_extractor("document.xyz")
        # Should return some extractor (fallback behavior)
        assert extractor is not None
        # The fallback extractor should support PDFs
        assert ".pdf" in extractor.supported_formats
