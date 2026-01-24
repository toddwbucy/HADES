"""Unit tests for core.processors.document_processor module."""

from unittest.mock import MagicMock, patch

import pytest

from core.processors.document_processor import (
    DocumentProcessor,
    ExtractionResult,
    ProcessingConfig,
    ProcessingResult,
    _get_default_staging_dir,
)


class TestGetDefaultStagingDir:
    """Tests for _get_default_staging_dir function."""

    def test_returns_string(self) -> None:
        """_get_default_staging_dir should return a string path."""
        result = _get_default_staging_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_uses_dev_shm_if_available(
        self,
        mock_isdir: MagicMock,
        mock_exists: MagicMock,
    ) -> None:
        """/dev/shm should be used on Linux when available."""
        mock_exists.return_value = True
        mock_isdir.return_value = True

        result = _get_default_staging_dir()
        assert "/dev/shm" in result

    @patch("os.path.exists")
    def test_falls_back_to_tempdir(self, mock_exists: MagicMock) -> None:
        """Should fall back to tempdir when /dev/shm unavailable."""
        mock_exists.return_value = False

        result = _get_default_staging_dir()
        assert "document_staging" in result


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_values(self) -> None:
        """ProcessingConfig should have sensible defaults."""
        config = ProcessingConfig()
        # Extraction defaults
        assert config.use_gpu is True
        assert config.extract_tables is True
        assert config.extract_equations is True
        assert config.extract_images is True
        assert config.use_ocr is False
        # Embedding defaults
        assert config.embedding_model == "jina-v4"
        assert config.embedder_type == "jina"
        assert config.embedding_dim == 2048
        assert config.use_fp16 is True
        assert config.device is None
        # Chunking defaults
        assert config.chunk_size_tokens == 1000
        assert config.chunk_overlap_tokens == 200
        assert config.chunking_strategy == "late"
        assert config.max_chunk_size == 8192
        # Processing defaults
        assert config.batch_size == 1
        assert config.num_workers == 1
        assert config.timeout_seconds == 300
        # Performance defaults
        assert config.cache_embeddings is True
        assert config.use_ramfs_staging is True
        assert isinstance(config.staging_dir, str)

    def test_custom_values(self) -> None:
        """ProcessingConfig should accept custom values."""
        config = ProcessingConfig(
            use_gpu=False,
            extract_tables=False,
            embedding_model="custom-model",
            chunk_size_tokens=500,
            batch_size=4,
        )
        assert config.use_gpu is False
        assert config.extract_tables is False
        assert config.embedding_model == "custom-model"
        assert config.chunk_size_tokens == 500
        assert config.batch_size == 4


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_minimal_creation(self) -> None:
        """ExtractionResult should only require full_text."""
        result = ExtractionResult(full_text="sample text")
        assert result.full_text == "sample text"
        assert result.tables == []
        assert result.equations == []
        assert result.images == []
        assert result.figures == []
        assert result.metadata == {}
        assert result.latex_source is None
        assert result.has_latex is False
        assert result.extraction_time == 0.0
        assert result.extractor_version == ""

    def test_full_creation(self) -> None:
        """ExtractionResult should accept all fields."""
        result = ExtractionResult(
            full_text="text content",
            tables=[{"name": "table1"}],
            equations=[{"latex": "x^2"}],
            images=[{"path": "img.png"}],
            figures=[{"caption": "Figure 1"}],
            metadata={"author": "test"},
            latex_source="\\documentclass{article}",
            has_latex=True,
            extraction_time=1.5,
            extractor_version="1.0.0",
        )
        assert result.full_text == "text content"
        assert len(result.tables) == 1
        assert len(result.equations) == 1
        assert len(result.images) == 1
        assert len(result.figures) == 1
        assert result.metadata == {"author": "test"}
        assert result.latex_source == "\\documentclass{article}"
        assert result.has_latex is True
        assert result.extraction_time == 1.5
        assert result.extractor_version == "1.0.0"


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    @pytest.fixture
    def extraction_result(self) -> ExtractionResult:
        """Create sample extraction result."""
        return ExtractionResult(full_text="test text")

    def test_minimal_creation(self, extraction_result: ExtractionResult) -> None:
        """ProcessingResult should accept required fields."""
        result = ProcessingResult(
            extraction=extraction_result,
            chunks=[],
            processing_metadata={},
            total_processing_time=1.0,
            extraction_time=0.5,
            chunking_time=0.3,
            embedding_time=0.2,
        )
        assert result.extraction is extraction_result
        assert result.chunks == []
        assert result.total_processing_time == 1.0
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []

    def test_failure_state(self, extraction_result: ExtractionResult) -> None:
        """ProcessingResult should handle failure state."""
        result = ProcessingResult(
            extraction=extraction_result,
            chunks=[],
            processing_metadata={},
            total_processing_time=0.1,
            extraction_time=0.1,
            chunking_time=0.0,
            embedding_time=0.0,
            success=False,
            errors=["Extraction failed"],
        )
        assert result.success is False
        assert "Extraction failed" in result.errors

    def test_to_dict_structure(self, extraction_result: ExtractionResult) -> None:
        """to_dict should return proper structure."""
        result = ProcessingResult(
            extraction=extraction_result,
            chunks=[],
            processing_metadata={"source": "test"},
            total_processing_time=1.0,
            extraction_time=0.5,
            chunking_time=0.3,
            embedding_time=0.2,
            errors=["warning1"],
            warnings=["minor issue"],
        )
        d = result.to_dict()

        assert "extraction" in d
        assert d["extraction"]["full_text"] == "test text"

        assert "chunks" in d
        assert d["chunks"] == []

        assert "processing_metadata" in d
        assert d["processing_metadata"]["source"] == "test"

        assert "performance" in d
        assert d["performance"]["total_time"] == 1.0
        assert d["performance"]["extraction_time"] == 0.5
        assert d["performance"]["chunking_time"] == 0.3
        assert d["performance"]["embedding_time"] == 0.2

        assert d["success"] is True
        assert d["errors"] == ["warning1"]
        assert d["warnings"] == ["minor issue"]


class TestDocumentProcessorInit:
    """Tests for DocumentProcessor initialization."""

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_default_config(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should use default config when none provided."""
        mock_docling.return_value = MagicMock()
        processor = DocumentProcessor()
        assert processor.config is not None
        assert isinstance(processor.config, ProcessingConfig)

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_custom_config(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should use provided config."""
        mock_docling.return_value = MagicMock()
        config = ProcessingConfig(
            use_gpu=False,
            batch_size=8,
        )
        processor = DocumentProcessor(config)
        assert processor.config.use_gpu is False
        assert processor.config.batch_size == 8

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_creates_docling_extractor(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should create DoclingExtractor on init."""
        mock_docling.return_value = MagicMock()
        DocumentProcessor()
        mock_docling.assert_called_once()


class TestDocumentProcessorMethods:
    """Tests for DocumentProcessor methods."""

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_has_process_document_method(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should have process_document method."""
        mock_docling.return_value = MagicMock()
        processor = DocumentProcessor()
        assert hasattr(processor, "process_document")
        assert callable(processor.process_document)

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_has_process_batch_method(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should have process_batch method."""
        mock_docling.return_value = MagicMock()
        processor = DocumentProcessor()
        assert hasattr(processor, "process_batch")
        assert callable(processor.process_batch)

    @patch("core.processors.document_processor.DoclingExtractor")
    def test_has_cleanup_method(self, mock_docling: MagicMock) -> None:
        """DocumentProcessor should have cleanup method."""
        mock_docling.return_value = MagicMock()
        processor = DocumentProcessor()
        assert hasattr(processor, "cleanup")
        assert callable(processor.cleanup)
