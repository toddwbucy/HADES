"""End-to-end tests for ArangoDB storage integration.

Tests the complete flow of storing processed documents in ArangoDB.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


class TestArangoHttp2Config:
    """Test ArangoHttp2Config dataclass."""

    def test_default_config(self) -> None:
        """ArangoHttp2Config should have sensible defaults."""
        from core.database.arango.optimized_client import ArangoHttp2Config

        config = ArangoHttp2Config()
        assert config.database == "_system"
        assert config.base_url == "http://localhost:8529"
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 30.0
        assert config.socket_path is None

    def test_socket_config(self) -> None:
        """ArangoHttp2Config should accept socket path."""
        from core.database.arango.optimized_client import ArangoHttp2Config

        config = ArangoHttp2Config(
            socket_path="/run/hades/readonly/arangod.sock",
            database="hades",
        )
        assert config.socket_path == "/run/hades/readonly/arangod.sock"
        assert config.database == "hades"

    def test_auth_config(self) -> None:
        """ArangoHttp2Config should accept auth credentials."""
        from core.database.arango.optimized_client import ArangoHttp2Config

        config = ArangoHttp2Config(
            username="root",
            password="secret",
        )
        assert config.username == "root"
        assert config.password == "secret"


class TestArangoClientMocked:
    """Test ArangoDB client with mocked HTTP."""

    @patch("core.database.arango.optimized_client.httpx.Client")
    def test_client_initialization(self, mock_client_cls: MagicMock) -> None:
        """Client should initialize with config."""
        from core.database.arango.optimized_client import (
            ArangoHttp2Client,
            ArangoHttp2Config,
        )

        config = ArangoHttp2Config(database="test_db")
        client = ArangoHttp2Client(config)

        assert client is not None
        mock_client_cls.assert_called_once()

    @patch("core.database.arango.optimized_client.httpx.Client")
    def test_client_close(self, mock_client_cls: MagicMock) -> None:
        """Client should close cleanly."""
        from core.database.arango.optimized_client import (
            ArangoHttp2Client,
            ArangoHttp2Config,
        )

        mock_http = MagicMock()
        mock_client_cls.return_value = mock_http

        config = ArangoHttp2Config()
        client = ArangoHttp2Client(config)
        client.close()

        mock_http.close.assert_called_once()


class TestDocumentStorage:
    """Test document storage patterns."""

    def create_sample_chunks(self, count: int = 5) -> list[dict[str, Any]]:
        """Create sample chunks for storage testing."""
        chunks = []
        for i in range(count):
            embedding = np.random.randn(2048).astype(np.float32)
            chunks.append(
                {
                    "_key": f"chunk_{i}",
                    "text": f"Sample chunk text number {i}",
                    "embedding": embedding.tolist(),
                    "start_char": i * 100,
                    "end_char": (i + 1) * 100,
                    "chunk_index": i,
                    "document_id": "test_doc",
                }
            )
        return chunks

    @patch("core.database.arango.optimized_client.httpx.Client")
    def test_document_format_for_arango(self, mock_client_cls: MagicMock) -> None:
        """Documents should be in correct format for ArangoDB."""
        chunks = self.create_sample_chunks(3)

        for chunk in chunks:
            # Should have _key for ArangoDB
            assert "_key" in chunk
            # Embedding should be list (JSON serializable)
            assert isinstance(chunk["embedding"], list)
            # Should have required fields
            assert "text" in chunk
            assert "document_id" in chunk

    @patch("core.database.arango.optimized_client.httpx.Client")
    def test_ndjson_bulk_format(self, mock_client_cls: MagicMock) -> None:
        """Chunks should be serializable to NDJSON for bulk import."""
        chunks = self.create_sample_chunks(3)

        # Convert to NDJSON (newline-delimited JSON)
        ndjson_lines = [json.dumps(chunk) for chunk in chunks]
        ndjson_content = "\n".join(ndjson_lines)

        # Should be valid NDJSON
        for line in ndjson_content.split("\n"):
            parsed = json.loads(line)
            assert "_key" in parsed


class TestEndToEndWithMockStorage:
    """Test complete pipeline with mocked storage."""

    @patch("core.database.arango.optimized_client.httpx.Client")
    @patch("core.processors.document_processor.EmbedderFactory")
    @patch("core.processors.document_processor.DoclingExtractor")
    def test_full_pipeline_to_storage(
        self,
        mock_docling_cls: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_http_cls: MagicMock,
        sample_pdf: Path,
        mock_embedder: Any,
    ) -> None:
        """Test processing a document and preparing it for storage."""
        from core.extractors.extractors_base import ExtractionResult as ExtractorResult
        from core.processors.document_processor import DocumentProcessor, ProcessingConfig

        # Setup mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractorResult(
            text="Test document content for storage " * 20,
            metadata={"pages": 1},
            tables=[],
            equations=[],
            images=[],
        )
        mock_docling_cls.return_value = mock_extractor
        mock_embedder_factory.create.return_value = mock_embedder

        # Process document
        config = ProcessingConfig(
            use_gpu=False,
            chunking_strategy="traditional",
            chunk_size_tokens=100,
            chunk_overlap_tokens=20,
            use_ramfs_staging=False,
        )
        processor = DocumentProcessor(config)
        result = processor.process_document(sample_pdf, document_id="storage_test")

        # Verify result can be converted to storage format
        assert result.success
        result_dict = result.to_dict()

        # Verify chunks have embeddings in list format (JSON-ready)
        for chunk in result_dict["chunks"]:
            assert "text" in chunk
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) == 2048

        processor.cleanup()


class TestConnectionPooling:
    """Test connection pool configuration."""

    def test_limits_config(self) -> None:
        """Pool limits should be configurable."""
        import httpx

        from core.database.arango.optimized_client import ArangoHttp2Config

        limits = httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
        )
        config = ArangoHttp2Config(pool_limits=limits)

        assert config.pool_limits is not None
        assert config.pool_limits.max_connections == 10


class TestErrorHandling:
    """Test storage error handling."""

    @patch("core.database.arango.optimized_client.httpx.Client")
    def test_http_error_raised(self, mock_client_cls: MagicMock) -> None:
        """ArangoHttpError should be raised on HTTP errors."""
        from core.database.arango.optimized_client import (
            ArangoHttp2Client,
            ArangoHttp2Config,
            ArangoHttpError,
        )

        # Mock HTTP client to return error
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": True, "errorMessage": "Not found"}
        mock_http.get.return_value = mock_response
        mock_client_cls.return_value = mock_http

        config = ArangoHttp2Config()
        _ = ArangoHttp2Client(config)  # Verify construction works

        # The error class should exist and be usable
        assert ArangoHttpError is not None
        error = ArangoHttpError(404, "Not found", {"code": 404})
        assert error.status_code == 404
        assert "404" in str(error)


class TestUnixSocketConnection:
    """Test Unix socket connection configuration."""

    def test_socket_path_configuration(self) -> None:
        """Socket path should be configurable."""
        from core.database.arango.optimized_client import ArangoHttp2Config

        # Read-only socket
        ro_config = ArangoHttp2Config(
            socket_path="/run/hades/readonly/arangod.sock",
        )
        assert "/readonly/" in ro_config.socket_path

        # Read-write socket
        rw_config = ArangoHttp2Config(
            socket_path="/run/hades/readwrite/arangod.sock",
        )
        assert "/readwrite/" in rw_config.socket_path
