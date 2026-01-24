"""Integration tests for document extractors."""

from pathlib import Path

import pytest

from core.extractors import ExtractionResult, ExtractorBase, get_extractor


class TestGetExtractor:
    """Tests for the get_extractor factory function."""

    def test_get_extractor_for_pdf(self) -> None:
        """Should return appropriate extractor for PDF files."""
        extractor = get_extractor("document.pdf")
        assert extractor is not None
        assert isinstance(extractor, ExtractorBase)

    def test_get_extractor_for_tex(self) -> None:
        """Should return LaTeX extractor for .tex files."""
        try:
            extractor = get_extractor("document.tex")
            assert extractor is not None
        except ImportError:
            pytest.skip("LaTeXExtractor not available")

    def test_get_extractor_for_python(self) -> None:
        """Should return code extractor for .py files."""
        try:
            extractor = get_extractor("script.py")
            assert extractor is not None
        except ImportError:
            pytest.skip("CodeExtractor not available")

    def test_get_extractor_unknown_type_falls_back(self) -> None:
        """Should fall back to robust/docling for unknown types."""
        try:
            extractor = get_extractor("file.xyz")
            assert extractor is not None
        except ImportError:
            pytest.skip("No fallback extractor available")


class TestDoclingExtractor:
    """Integration tests for DoclingExtractor."""

    @pytest.fixture
    def docling_extractor(self):
        """Create DoclingExtractor instance if available."""
        try:
            from core.extractors import DoclingExtractor

            if DoclingExtractor is None:
                pytest.skip("DoclingExtractor not available")
            return DoclingExtractor()
        except ImportError:
            pytest.skip("DoclingExtractor not available")

    def test_extractor_has_required_methods(self, docling_extractor) -> None:
        """DoclingExtractor should implement ExtractorBase interface."""
        assert hasattr(docling_extractor, "extract")
        assert callable(docling_extractor.extract)

    def test_extract_returns_extraction_result(self, docling_extractor, sample_pdf_path: Path) -> None:
        """extract() should return ExtractionResult."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not available")

        result = docling_extractor.extract(sample_pdf_path)
        assert isinstance(result, ExtractionResult)
        # extractors.ExtractionResult uses 'text' not 'full_text'
        assert result.text is not None
        assert len(result.text) > 0

    def test_extract_populates_metadata(self, docling_extractor, sample_pdf_path: Path) -> None:
        """extract() should populate metadata."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not available")

        result = docling_extractor.extract(sample_pdf_path)
        assert result.metadata is not None

    def test_extract_handles_missing_file(self, docling_extractor) -> None:
        """extract() should handle missing files gracefully."""
        with pytest.raises((FileNotFoundError, OSError)):
            docling_extractor.extract(Path("/nonexistent/file.pdf"))


class TestLaTeXExtractor:
    """Integration tests for LaTeXExtractor."""

    @pytest.fixture
    def latex_extractor(self):
        """Create LaTeXExtractor instance if available."""
        try:
            from core.extractors import LaTeXExtractor

            if LaTeXExtractor is None:
                pytest.skip("LaTeXExtractor not available")
            return LaTeXExtractor()
        except ImportError:
            pytest.skip("LaTeXExtractor not available")

    def test_extractor_has_required_methods(self, latex_extractor) -> None:
        """LaTeXExtractor should implement ExtractorBase interface."""
        assert hasattr(latex_extractor, "extract")
        assert callable(latex_extractor.extract)

    def test_extract_latex_content(self, latex_extractor, sample_latex_path: Path) -> None:
        """extract() should process LaTeX files."""
        if not sample_latex_path.exists():
            pytest.skip("Sample LaTeX file not available")

        result = latex_extractor.extract(sample_latex_path)
        assert isinstance(result, ExtractionResult)
        # extractors.ExtractionResult uses 'text' not 'full_text'
        assert result.text is not None


class TestRobustExtractor:
    """Integration tests for RobustExtractor (fallback extractor)."""

    @pytest.fixture
    def robust_extractor(self):
        """Create RobustExtractor instance if available."""
        try:
            from core.extractors import RobustExtractor

            if RobustExtractor is None:
                pytest.skip("RobustExtractor not available")
            return RobustExtractor()
        except ImportError:
            pytest.skip("RobustExtractor not available")

    def test_extractor_has_required_methods(self, robust_extractor) -> None:
        """RobustExtractor should implement ExtractorBase interface."""
        assert hasattr(robust_extractor, "extract")
        assert callable(robust_extractor.extract)


class TestExtractorErrorHandling:
    """Tests for extractor error handling."""

    def test_corrupted_pdf_handling(self, tmp_path: Path) -> None:
        """Extractors should handle corrupted PDFs gracefully."""
        # Create a corrupted PDF (invalid content)
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"not a valid pdf content")

        try:
            extractor = get_extractor(corrupted_pdf)
            # Should either raise an error or return empty result
            try:
                result = extractor.extract(corrupted_pdf)
                # If it doesn't raise, it should return a valid structure
                assert isinstance(result, ExtractionResult)
            except Exception:
                # Expected - corrupted files should raise
                pass
        except ImportError:
            pytest.skip("No extractor available")

    def test_empty_file_handling(self, tmp_path: Path) -> None:
        """Extractors should handle empty files gracefully."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        try:
            extractor = get_extractor(empty_pdf)
            try:
                result = extractor.extract(empty_pdf)
                assert isinstance(result, ExtractionResult)
            except Exception:
                # Expected - empty files may raise
                pass
        except ImportError:
            pytest.skip("No extractor available")
