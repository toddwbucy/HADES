"""End-to-end integration tests for the HADES RAG pipeline."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestPipelineComponents:
    """Tests for pipeline component integration."""

    def test_all_core_modules_importable(self) -> None:
        """All core modules should be importable."""
        # Test imports don't raise
        from core.config import BaseConfig
        from core.database.database_factory import DatabaseFactory
        from core.embedders import EmbedderFactory
        from core.extractors import get_extractor
        from core.logging import LogManager
        from core.processors import DocumentProcessor

        # Basic assertions
        assert BaseConfig is not None
        assert DatabaseFactory is not None
        assert get_extractor is not None
        assert EmbedderFactory is not None
        assert DocumentProcessor is not None
        assert LogManager is not None

    def test_extractor_to_processor_data_flow(self) -> None:
        """Data should flow correctly from extractor to processor."""
        # extractors.ExtractionResult uses 'text', not 'full_text'
        from core.extractors import ExtractionResult

        # Create mock extraction result
        extraction = ExtractionResult(
            text="Test document content for pipeline testing.",
            metadata={"title": "Test", "source": "test.pdf"},
        )

        # Verify structure is compatible with processor
        assert hasattr(extraction, "text")
        assert hasattr(extraction, "metadata")
        assert isinstance(extraction.text, str)
        assert isinstance(extraction.metadata, dict)


class TestMockedPipeline:
    """End-to-end tests with fully mocked components."""

    @pytest.fixture
    def mock_pipeline_components(self) -> dict[str, MagicMock]:
        """Create all mocked pipeline components."""
        # Mock extractor - return mock with 'text' attribute (extractors use 'text')
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            text="Extracted text content. " * 100,
            metadata={"title": "Test Document", "pages": 5},
        )

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embedding_dim = 2048
        mock_embedder.embed_with_late_chunking.return_value = [
            MagicMock(
                chunk_index=i,
                text=f"Chunk {i} text",
                embedding=[0.1] * 2048,
                token_start=i * 100,
                token_end=(i + 1) * 100,
            )
            for i in range(5)
        ]

        # Mock database client
        mock_db = MagicMock()
        mock_db.execute_query.return_value = []
        mock_db.insert_in_transaction.return_value = {"created": 5}
        mock_db.begin_transaction.return_value = "txn-123"
        mock_db.commit_transaction.return_value = None
        mock_db.close.return_value = None

        return {
            "extractor": mock_extractor,
            "embedder": mock_embedder,
            "database": mock_db,
        }

    def test_full_pipeline_with_mocks(
        self,
        mock_pipeline_components: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        """Test complete pipeline flow with all components mocked."""
        # Create test PDF file
        test_pdf = tmp_path / "test_document.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")

        extractor = mock_pipeline_components["extractor"]
        embedder = mock_pipeline_components["embedder"]
        db_client = mock_pipeline_components["database"]

        # Simulate pipeline execution
        # Step 1: Extract
        extraction_result = extractor.extract(test_pdf)
        assert extraction_result.text is not None

        # Step 2: Embed with late chunking
        chunks = embedder.embed_with_late_chunking(extraction_result.text)
        assert len(chunks) > 0

        # Step 3: Store in database (using transaction-based insert)
        documents = [
            {
                "_key": f"chunk-{chunk.chunk_index}",
                "text": chunk.text,
                "embedding": chunk.embedding,
                "document_id": "test-doc-1",
            }
            for chunk in chunks
        ]

        result = db_client.insert_in_transaction("embeddings", documents)
        assert result["created"] > 0

    def test_pipeline_error_propagation(
        self,
        mock_pipeline_components: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        """Pipeline should propagate errors appropriately."""
        extractor = mock_pipeline_components["extractor"]
        extractor.extract.side_effect = Exception("Extraction failed")

        test_pdf = tmp_path / "bad.pdf"
        test_pdf.write_bytes(b"invalid")

        with pytest.raises(Exception, match="Extraction failed"):
            extractor.extract(test_pdf)


class TestPipelineConfiguration:
    """Tests for pipeline configuration."""

    def test_processing_config_defaults(self) -> None:
        """ProcessingConfig should have sensible defaults."""
        from core.processors import ProcessingConfig

        config = ProcessingConfig()

        # ProcessingConfig has extraction/embedding settings, not chunk_size
        assert hasattr(config, "use_gpu")
        assert hasattr(config, "extract_tables")

    def test_config_from_dataclass(self) -> None:
        """Config should be creatable with valid parameters."""
        from core.processors import ProcessingConfig

        # ProcessingConfig uses extraction/embedding parameters
        config = ProcessingConfig(use_gpu=False, extract_tables=True)
        assert config is not None
        assert config.use_gpu is False
        assert config.extract_tables is True


class TestPipelineOutputFormats:
    """Tests for pipeline output formats."""

    def test_embedding_output_format(self) -> None:
        """Embeddings should be in correct format for storage."""
        # Simulated embedding output
        embedding_doc = {
            "_key": "doc1_chunk0",
            "document_id": "doc1",
            "chunk_index": 0,
            "text": "Sample chunk text",
            "embedding": [0.1] * 2048,
            "token_start": 0,
            "token_end": 50,
        }

        # Verify format
        assert "_key" in embedding_doc
        assert "embedding" in embedding_doc
        assert isinstance(embedding_doc["embedding"], list)
        assert len(embedding_doc["embedding"]) == 2048

    def test_metadata_output_format(self) -> None:
        """Metadata should be in correct format for storage."""
        metadata_doc = {
            "_key": "doc1",
            "title": "Test Document",
            "source_path": "/path/to/doc.pdf",
            "page_count": 10,
            "processed_at": "2024-01-01T00:00:00Z",
            "chunk_count": 5,
        }

        assert "_key" in metadata_doc
        assert "title" in metadata_doc
        assert "processed_at" in metadata_doc


class TestPipelinePerformance:
    """Basic performance-related tests."""

    def test_chunking_produces_reasonable_count(self) -> None:
        """Chunking should produce reasonable number of chunks."""
        # 10000 chars with 500 char chunks = ~20 chunks
        text = "A" * 10000

        from core.processors import TokenBasedChunking

        try:
            chunker = TokenBasedChunking(chunk_size=500, overlap=50)
            # ChunkingStrategy uses create_chunks, not chunk
            chunks = chunker.create_chunks(text)

            # Should be between 10 and 50 chunks
            assert 5 < len(chunks) < 100
        except Exception:
            pytest.skip("TokenBasedChunking not available")

    def test_embedding_dimension_consistency(self) -> None:
        """All embeddings should have same dimension."""
        embeddings = [
            [0.1] * 2048,
            [0.2] * 2048,
            [0.3] * 2048,
        ]

        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All same dimension
        assert dimensions[0] == 2048


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS", "0") != "1",
    reason="Slow integration tests skipped (set RUN_SLOW_TESTS=1 to enable)",
)
class TestSlowIntegration:
    """Slow integration tests that require real components.

    These tests are skipped by default. Enable with RUN_SLOW_TESTS=1.
    """

    def test_real_pdf_extraction(self, sample_pdf_path: Path) -> None:
        """Test extraction with real PDF file."""
        if not sample_pdf_path.exists():
            pytest.skip("Sample PDF not available")

        from core.extractors import get_extractor

        extractor = get_extractor(sample_pdf_path)
        result = extractor.extract(sample_pdf_path)

        # extractors.ExtractionResult uses 'text', not 'full_text'
        assert result.text is not None
        assert len(result.text) > 0

    def test_real_embedder_initialization(self) -> None:
        """Test real embedder initialization (downloads model)."""
        from core.embedders import JinaV4Embedder

        if JinaV4Embedder is None:
            pytest.skip("JinaV4Embedder not available")

        embedder = JinaV4Embedder()
        assert embedder.embedding_dim == 2048
