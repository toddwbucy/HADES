"""End-to-end tests for the complete document processing pipeline.

These tests verify the full flow: PDF extraction → chunking → embedding.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
)


class TestDocumentProcessingPipeline:
    """Test the complete document processing pipeline."""

    @pytest.fixture
    def processing_config(self) -> ProcessingConfig:
        """Create processing config for tests."""
        return ProcessingConfig(
            use_gpu=False,
            use_ocr=False,
            extract_tables=True,
            extract_equations=False,
            chunking_strategy="traditional",
            chunk_size_tokens=50,
            chunk_overlap_tokens=10,
            use_ramfs_staging=False,
        )

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_full_pipeline_with_mock_pdf(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
        processing_config: ProcessingConfig,
    ) -> None:
        """Test full pipeline processes a PDF and returns chunks with embeddings."""
        # Setup mock extractor
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Machine learning is a powerful technique. " * 20,
            metadata={"num_pages": 3},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor

        # Setup mock embedder
        mock_embedder_factory.create.return_value = mock_embedder

        # Process document
        processor = DocumentProcessor(processing_config)
        result = processor.process_document(sample_pdf)

        # Verify result structure
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert len(result.errors) == 0

        # Verify extraction happened
        assert mock_extractor.extract.called

        # Verify chunks were created
        assert len(result.chunks) > 0

        # Verify each chunk has an embedding
        for chunk in result.chunks:
            assert chunk.text is not None
            assert len(chunk.text) > 0
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 2048

        # Verify timing data
        assert result.total_processing_time > 0
        assert result.extraction_time >= 0
        assert result.embedding_time >= 0

        # Cleanup
        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_pipeline_handles_empty_document(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        empty_pdf: Path,
        mock_embedder: MagicMock,
        processing_config: ProcessingConfig,
    ) -> None:
        """Test pipeline handles empty documents gracefully."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="",
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(processing_config)
        result = processor.process_document(empty_pdf)

        # Should complete without crashing
        assert isinstance(result, ProcessingResult)
        # Empty document produces no chunks
        assert len(result.chunks) == 0
        # Should indicate failure or warning
        assert result.success is False or len(result.warnings) > 0

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_pipeline_preserves_document_metadata(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
        processing_config: ProcessingConfig,
    ) -> None:
        """Test that document metadata is preserved through the pipeline."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Sample content " * 50,
            metadata={
                "num_pages": 5,
                "author": "Test Author",
                "title": "Test Document",
            },
            tables=[{"name": "table1"}],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(processing_config)
        result = processor.process_document(sample_pdf, document_id="test_doc_123")

        # Verify processing metadata
        assert result.processing_metadata["document_id"] == "test_doc_123"
        assert result.processing_metadata["chunk_count"] == len(result.chunks)
        assert result.processing_metadata["has_tables"] is True

        # Verify source path
        assert str(sample_pdf) in result.processing_metadata["source_path"]

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_batch_processing(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        minimal_pdf: Path,
        mock_embedder: MagicMock,
        processing_config: ProcessingConfig,
    ) -> None:
        """Test batch processing of multiple documents."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Batch document content " * 30,
            metadata={"num_pages": 1},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(processing_config)

        # Process batch
        document_paths = [
            (sample_pdf, None),
            (minimal_pdf, None),
        ]
        document_ids = ["doc1", "doc2"]

        results = processor.process_batch(document_paths, document_ids)

        # Verify all documents processed
        assert len(results) == 2
        assert all(isinstance(r, ProcessingResult) for r in results)
        assert results[0].processing_metadata["document_id"] == "doc1"
        assert results[1].processing_metadata["document_id"] == "doc2"

        processor.cleanup()


class TestChunkingStrategies:
    """Test different chunking strategies in the pipeline."""

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_traditional_chunking(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test traditional (token-based) chunking strategy."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="traditional",
            chunk_size_tokens=30,
            chunk_overlap_tokens=5,
            use_ramfs_staging=False,
        )

        mock_extractor = MagicMock()
        # Create text with exactly 100 words for predictable chunking
        mock_extractor.extract.return_value = ExtractorResult(
            text=" ".join([f"word{i}" for i in range(100)]),
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        # With 100 words, chunk_size=30, overlap=5: expect ~4 chunks
        assert len(result.chunks) >= 3
        assert result.success is True

        # Verify chunk properties
        for chunk in result.chunks:
            words_in_chunk = len(chunk.text.split())
            assert words_in_chunk <= 30  # Should not exceed chunk size

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_late_chunking_strategy(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test late chunking strategy."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="late",
            chunk_size_tokens=50,
            use_ramfs_staging=False,
        )

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="This is sample text for late chunking. " * 50,
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        assert result.success is True
        assert len(result.chunks) > 0
        # Late chunking uses embedder's embed_with_late_chunking
        assert result.processing_metadata["chunking_strategy"] == "late"

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_semantic_chunking_strategy(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test semantic chunking strategy."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="semantic",
            chunk_size_tokens=100,
            chunk_overlap_tokens=20,
            use_ramfs_staging=False,
        )

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="""First paragraph about machine learning.

Second paragraph discussing neural networks and deep learning algorithms.

Third paragraph covering practical applications and use cases.""",
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        assert result.success is True
        assert result.processing_metadata["chunking_strategy"] == "semantic"

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_sliding_window_chunking(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test sliding window chunking strategy."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="sliding_window",
            chunk_size_tokens=50,
            chunk_overlap_tokens=25,
            use_ramfs_staging=False,
        )

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Sliding window test content. " * 40,
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        assert result.success is True
        # Sliding window creates overlapping chunks
        assert len(result.chunks) > 0

        processor.cleanup()


class TestErrorHandling:
    """Test error handling in the document pipeline."""

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_handles_extraction_error(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test pipeline handles extraction errors gracefully."""
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("Extraction failed")
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        config = ProcessingConfig(use_gpu=False, use_ramfs_staging=False)
        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        assert result.success is False
        assert len(result.errors) > 0
        assert "Extraction failed" in str(result.errors)

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_handles_nonexistent_file(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_embedder: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline handles missing files gracefully."""
        mock_docling_cls.return_value = MagicMock()
        mock_embedder_factory.create.return_value = mock_embedder

        config = ProcessingConfig(use_gpu=False, use_ramfs_staging=False)
        processor = DocumentProcessor(config)

        nonexistent = tmp_path / "does_not_exist.pdf"
        result = processor.process_document(nonexistent)

        # Should complete without crashing
        assert isinstance(result, ProcessingResult)
        # Should indicate some kind of issue
        assert result.success is False or len(result.errors) > 0

        processor.cleanup()

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_handles_embedding_error(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
    ) -> None:
        """Test pipeline handles embedding errors gracefully."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Content to embed",
            metadata={},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor

        # Create embedder that fails
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = RuntimeError("Embedding failed")
        mock_embedder.embed_with_late_chunking.side_effect = RuntimeError("Late chunking failed")
        mock_embedder_factory.create.return_value = mock_embedder

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="late",
            use_ramfs_staging=False,
        )
        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        assert result.success is False
        assert len(result.errors) > 0

        processor.cleanup()


class TestProcessingResultSerialization:
    """Test ProcessingResult serialization."""

    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_to_dict_is_json_serializable(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        sample_pdf: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Test that ProcessingResult.to_dict() is JSON serializable."""
        import json

        from core.extractors.extractors_base import ExtractionResult as ExtractorResult

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Sample text for serialization test. " * 20,
            metadata={"pages": 1},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="traditional",
            chunk_size_tokens=30,
            use_ramfs_staging=False,
        )
        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf)

        # Should not raise
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

        # Verify structure
        parsed = json.loads(json_str)
        assert "extraction" in parsed
        assert "chunks" in parsed
        assert "performance" in parsed
        assert "success" in parsed

        processor.cleanup()
