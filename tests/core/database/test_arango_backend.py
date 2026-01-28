"""Tests for core.database.arango.backend — ArangoDB storage backend."""

from __future__ import annotations

import pytest

from core.database.arango.backend import _to_dict
from core.database.schemas import Chunk, ChunkEmbedding, DocumentMetadata


class TestToDict:
    """Test the _to_dict helper function."""

    def test_dict_passthrough(self):
        """Dicts should pass through unchanged."""
        d = {"key": "value", "nested": {"a": 1}}
        assert _to_dict(d) is d

    def test_document_metadata(self):
        """DocumentMetadata should convert via to_dict()."""
        meta = DocumentMetadata(
            doc_id="test",
            title="Test Title",
            source_metadata={"arxiv_id": "2501.12345"},
        )
        result = _to_dict(meta)
        assert isinstance(result, dict)
        assert result["doc_id"] == "test"
        assert result["title"] == "Test Title"
        assert result["source_metadata"]["arxiv_id"] == "2501.12345"

    def test_chunk(self):
        """Chunk should convert via to_dict()."""
        chunk = Chunk(
            doc_id="test",
            chunk_index=0,
            text="Hello world",
            start_char=0,
            end_char=11,
        )
        result = _to_dict(chunk)
        assert isinstance(result, dict)
        assert result["doc_id"] == "test"
        assert result["chunk_index"] == 0
        assert result["text"] == "Hello world"
        assert result["chunk_id"] == "test_chunk_0"

    def test_chunk_embedding(self):
        """ChunkEmbedding should convert via to_dict()."""
        emb = ChunkEmbedding(
            doc_id="test",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        result = _to_dict(emb)
        assert isinstance(result, dict)
        assert result["doc_id"] == "test"
        assert result["chunk_index"] == 0
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["embedding_id"] == "test_chunk_0_emb"

    def test_invalid_type_raises(self):
        """Non-dict objects without to_dict() should raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            _to_dict("not a dict or schema")

        with pytest.raises(TypeError, match="Cannot convert"):
            _to_dict(123)


class TestArangoBackendSchemaSupport:
    """Test that ArangoBackend handles schema objects correctly.

    These tests verify the schema conversion logic without requiring
    an actual ArangoDB connection.
    """

    def test_schema_objects_converted(self):
        """Verify schema objects would be converted to dicts for storage."""
        # Create schema objects
        meta = DocumentMetadata(doc_id="test", title="Test")
        chunk = Chunk(doc_id="test", chunk_index=0, text="text")
        emb = ChunkEmbedding(doc_id="test", chunk_index=0, embedding=[0.1])

        # Convert them
        meta_dict = _to_dict(meta)
        chunk_dict = _to_dict(chunk)
        emb_dict = _to_dict(emb)

        # Verify structure matches what ArangoBackend expects
        assert "doc_id" in meta_dict
        assert "title" in meta_dict
        assert "text" in chunk_dict
        assert "chunk_index" in chunk_dict
        assert "embedding" in emb_dict
