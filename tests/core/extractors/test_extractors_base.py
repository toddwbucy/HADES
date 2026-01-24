"""Unit tests for core.extractors.extractors_base module."""

from pathlib import Path

import pytest

from core.extractors.extractors_base import (
    ExtractionResult,
    ExtractorBase,
    ExtractorConfig,
)


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_minimal_creation(self) -> None:
        """ExtractionResult should only require text."""
        result = ExtractionResult(text="sample text")
        assert result.text == "sample text"
        assert result.metadata == {}
        assert result.chunks == []
        assert result.equations == []
        assert result.tables == []
        assert result.images == []
        assert result.code_blocks == []
        assert result.references == []
        assert result.error is None
        assert result.processing_time == 0.0

    def test_full_creation(self) -> None:
        """ExtractionResult should accept all fields."""
        result = ExtractionResult(
            text="main text",
            metadata={"author": "test"},
            chunks=[{"text": "chunk1"}],
            equations=[{"latex": "x^2"}],
            tables=[{"rows": 3}],
            images=[{"path": "img.png"}],
            code_blocks=[{"lang": "python"}],
            references=[{"title": "ref1"}],
            error="minor warning",
            processing_time=1.5,
        )
        assert result.text == "main text"
        assert result.metadata == {"author": "test"}
        assert len(result.chunks) == 1
        assert len(result.equations) == 1
        assert len(result.tables) == 1
        assert len(result.images) == 1
        assert len(result.code_blocks) == 1
        assert len(result.references) == 1
        assert result.error == "minor warning"
        assert result.processing_time == 1.5


class TestExtractorConfig:
    """Tests for ExtractorConfig dataclass."""

    def test_default_values(self) -> None:
        """ExtractorConfig should have sensible defaults."""
        config = ExtractorConfig()
        assert config.use_gpu is True
        assert config.batch_size == 1
        assert config.timeout_seconds == 300
        assert config.extract_equations is True
        assert config.extract_tables is True
        assert config.extract_images is True
        assert config.extract_code is True
        assert config.extract_references is True
        assert config.max_pages is None
        assert config.ocr_enabled is False

    def test_custom_values(self) -> None:
        """ExtractorConfig should accept custom values."""
        config = ExtractorConfig(
            use_gpu=False,
            batch_size=4,
            timeout_seconds=600,
            extract_equations=False,
            extract_tables=False,
            extract_images=False,
            extract_code=False,
            extract_references=False,
            max_pages=100,
            ocr_enabled=True,
        )
        assert config.use_gpu is False
        assert config.batch_size == 4
        assert config.timeout_seconds == 600
        assert config.extract_equations is False
        assert config.extract_tables is False
        assert config.extract_images is False
        assert config.extract_code is False
        assert config.extract_references is False
        assert config.max_pages == 100
        assert config.ocr_enabled is True


class ConcreteExtractor(ExtractorBase):
    """Concrete implementation for testing ExtractorBase."""

    def extract(
        self,
        file_path: str | Path,
        **kwargs,
    ) -> ExtractionResult:
        """Return mock extraction result."""
        return ExtractionResult(text=f"Extracted from {file_path}")

    def extract_batch(
        self,
        file_paths: list[str | Path],
        **kwargs,
    ) -> list[ExtractionResult]:
        """Return mock extraction results."""
        return [self.extract(path) for path in file_paths]

    @property
    def supported_formats(self) -> list[str]:
        """Return supported formats."""
        return [".pdf", ".txt"]


class TestExtractorBase:
    """Tests for ExtractorBase abstract class."""

    @pytest.fixture
    def extractor(self) -> ConcreteExtractor:
        """Create a concrete extractor instance."""
        return ConcreteExtractor()

    @pytest.fixture
    def extractor_with_config(self) -> ConcreteExtractor:
        """Create extractor with custom config."""
        config = ExtractorConfig(
            use_gpu=False,
            batch_size=8,
            timeout_seconds=120,
        )
        return ConcreteExtractor(config)

    def test_default_config(self, extractor: ConcreteExtractor) -> None:
        """Extractor should have default config when none provided."""
        assert extractor.config is not None
        assert extractor.config.use_gpu is True
        assert extractor.config.batch_size == 1

    def test_custom_config(self, extractor_with_config: ConcreteExtractor) -> None:
        """Extractor should use provided config."""
        assert extractor_with_config.config.use_gpu is False
        assert extractor_with_config.config.batch_size == 8
        assert extractor_with_config.config.timeout_seconds == 120

    def test_extract_returns_result(self, extractor: ConcreteExtractor) -> None:
        """extract should return ExtractionResult."""
        result = extractor.extract("/path/to/file.pdf")
        assert isinstance(result, ExtractionResult)
        assert "file.pdf" in result.text

    def test_extract_batch_returns_results(self, extractor: ConcreteExtractor) -> None:
        """extract_batch should return list of ExtractionResults."""
        paths = ["/path/a.pdf", "/path/b.pdf", "/path/c.pdf"]
        results = extractor.extract_batch(paths)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_supported_formats_property(self, extractor: ConcreteExtractor) -> None:
        """supported_formats property should return list of formats."""
        formats = extractor.supported_formats
        assert isinstance(formats, list)
        assert ".pdf" in formats
        assert ".txt" in formats

    def test_supports_gpu_default_false(self, extractor: ConcreteExtractor) -> None:
        """supports_gpu should default to False."""
        assert extractor.supports_gpu is False

    def test_supports_batch_default_true(self, extractor: ConcreteExtractor) -> None:
        """supports_batch should default to True."""
        assert extractor.supports_batch is True

    def test_supports_ocr_default_false(self, extractor: ConcreteExtractor) -> None:
        """supports_ocr should default to False."""
        assert extractor.supports_ocr is False

    def test_get_extractor_info(self, extractor_with_config: ConcreteExtractor) -> None:
        """get_extractor_info should return comprehensive info dict."""
        info = extractor_with_config.get_extractor_info()
        assert info["class"] == "ConcreteExtractor"
        assert info["supported_formats"] == [".pdf", ".txt"]
        assert info["supports_gpu"] is False
        assert info["supports_batch"] is True
        assert info["supports_ocr"] is False
        assert info["config"]["use_gpu"] is False
        assert info["config"]["batch_size"] == 8
        assert info["config"]["timeout_seconds"] == 120


class TestValidateFile:
    """Tests for validate_file method."""

    @pytest.fixture
    def extractor(self) -> ConcreteExtractor:
        """Create extractor for testing."""
        return ConcreteExtractor()

    def test_validate_file_nonexistent(self, extractor: ConcreteExtractor, tmp_path: Path) -> None:
        """validate_file should return False for nonexistent file."""
        fake_path = tmp_path / "nonexistent.pdf"
        assert extractor.validate_file(fake_path) is False

    def test_validate_file_directory(self, extractor: ConcreteExtractor, tmp_path: Path) -> None:
        """validate_file should return False for directory."""
        assert extractor.validate_file(tmp_path) is False

    def test_validate_file_empty(self, extractor: ConcreteExtractor, tmp_path: Path) -> None:
        """validate_file should return False for empty file."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.touch()
        assert extractor.validate_file(empty_file) is False

    def test_validate_file_valid(self, extractor: ConcreteExtractor, tmp_path: Path) -> None:
        """validate_file should return True for valid file."""
        valid_file = tmp_path / "valid.pdf"
        valid_file.write_text("content")
        assert extractor.validate_file(valid_file) is True


class GpuEnabledExtractor(ConcreteExtractor):
    """Extractor with GPU support."""

    @property
    def supports_gpu(self) -> bool:
        """Enable GPU support."""
        return True


class OcrEnabledExtractor(ConcreteExtractor):
    """Extractor with OCR support."""

    @property
    def supports_ocr(self) -> bool:
        """Enable OCR support."""
        return True


class TestExtractorCapabilities:
    """Tests for extractor capability flags."""

    def test_gpu_can_be_enabled(self) -> None:
        """Extractors can override supports_gpu."""
        extractor = GpuEnabledExtractor()
        assert extractor.supports_gpu is True

    def test_ocr_can_be_enabled(self) -> None:
        """Extractors can override supports_ocr."""
        extractor = OcrEnabledExtractor()
        assert extractor.supports_ocr is True

    def test_extractor_info_reflects_capabilities(self) -> None:
        """get_extractor_info should reflect actual capabilities."""
        extractor = GpuEnabledExtractor()
        info = extractor.get_extractor_info()
        assert info["supports_gpu"] is True
