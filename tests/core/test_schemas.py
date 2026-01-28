"""Tests for core.database.schemas — document schema definitions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.database.schemas import Chunk, ChunkEmbedding, DocumentMetadata


class TestDocumentMetadata:
    def test_basic_creation(self):
        doc = DocumentMetadata(doc_id="test-doc", title="Test Document")
        assert doc.doc_id == "test-doc"
        assert doc.title == "Test Document"
        assert doc.authors == []
        assert doc.source_type == "unknown"
        assert doc.source_metadata == {}
        assert doc.tags == []

    def test_with_all_fields(self):
        now = datetime.now(timezone.utc)
        doc = DocumentMetadata(
            doc_id="arxiv-paper",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            created_at=now,
            source_type="arxiv",
            source_metadata={"arxiv_id": "1706.03762", "categories": ["cs.CL"]},
            tags=["transformer", "attention"],
            extra={"citation_count": 50000},
        )
        assert doc.authors == ["Vaswani et al."]
        assert doc.source_type == "arxiv"
        assert doc.source_metadata["arxiv_id"] == "1706.03762"
        assert "transformer" in doc.tags

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        doc = DocumentMetadata(
            doc_id="test",
            title="Test",
            created_at=now,
            source_metadata={"key": "value"},
        )
        d = doc.to_dict()
        assert d["doc_id"] == "test"
        assert d["title"] == "Test"
        assert d["created_at"] == now.isoformat()
        assert d["source_metadata"] == {"key": "value"}

    def test_from_dict(self):
        data = {
            "doc_id": "test",
            "title": "Test Title",
            "authors": ["Author 1"],
            "created_at": "2025-01-01T00:00:00+00:00",
            "source_type": "pdf",
            "source_metadata": {"path": "/data/test.pdf"},
        }
        doc = DocumentMetadata.from_dict(data)
        assert doc.doc_id == "test"
        assert doc.title == "Test Title"
        assert doc.authors == ["Author 1"]
        assert doc.source_type == "pdf"
        assert doc.created_at is not None

    def test_from_dict_minimal(self):
        data = {"doc_id": "minimal"}
        doc = DocumentMetadata.from_dict(data)
        assert doc.doc_id == "minimal"
        assert doc.title == ""
        assert doc.authors == []


class TestChunk:
    def test_basic_creation(self):
        chunk = Chunk(doc_id="doc1", chunk_index=0, text="Hello world")
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_index == 0
        assert chunk.text == "Hello world"
        assert chunk.start_char is None
        assert chunk.end_char is None

    def test_chunk_id_property(self):
        chunk = Chunk(doc_id="my-doc", chunk_index=5, text="text")
        assert chunk.chunk_id == "my-doc_chunk_5"

    def test_with_all_fields(self):
        chunk = Chunk(
            doc_id="doc1",
            chunk_index=2,
            text="Some text content",
            start_char=100,
            end_char=200,
            total_chunks=10,
            metadata={"section": "Introduction"},
        )
        assert chunk.start_char == 100
        assert chunk.end_char == 200
        assert chunk.total_chunks == 10
        assert chunk.metadata["section"] == "Introduction"

    def test_to_dict(self):
        chunk = Chunk(
            doc_id="doc1",
            chunk_index=0,
            text="text",
            start_char=0,
            end_char=4,
        )
        d = chunk.to_dict()
        assert d["doc_id"] == "doc1"
        assert d["chunk_index"] == 0
        assert d["chunk_id"] == "doc1_chunk_0"
        assert d["text"] == "text"
        assert d["start_char"] == 0

    def test_from_dict(self):
        data = {
            "doc_id": "doc1",
            "chunk_index": 3,
            "text": "content",
            "start_char": 50,
            "metadata": {"page": 2},
        }
        chunk = Chunk.from_dict(data)
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_index == 3
        assert chunk.text == "content"
        assert chunk.start_char == 50
        assert chunk.metadata["page"] == 2


class TestChunkEmbedding:
    def test_basic_creation(self):
        emb = ChunkEmbedding(
            doc_id="doc1",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
        )
        assert emb.doc_id == "doc1"
        assert emb.chunk_index == 0
        assert emb.embedding == [0.1, 0.2, 0.3]
        assert emb.model == "jina-embeddings-v4"
        assert emb.task == "retrieval.passage"

    def test_embedding_id_property(self):
        emb = ChunkEmbedding(doc_id="my-doc", chunk_index=5, embedding=[])
        assert emb.embedding_id == "my-doc_chunk_5_emb"

    def test_chunk_id_property(self):
        emb = ChunkEmbedding(doc_id="my-doc", chunk_index=5, embedding=[])
        assert emb.chunk_id == "my-doc_chunk_5"

    def test_with_custom_model(self):
        emb = ChunkEmbedding(
            doc_id="doc1",
            chunk_index=0,
            embedding=[0.1],
            model="openai-ada-002",
            task="text-matching",
        )
        assert emb.model == "openai-ada-002"
        assert emb.task == "text-matching"

    def test_to_dict(self):
        emb = ChunkEmbedding(
            doc_id="doc1",
            chunk_index=0,
            embedding=[0.1, 0.2],
        )
        d = emb.to_dict()
        assert d["doc_id"] == "doc1"
        assert d["chunk_index"] == 0
        assert d["embedding_id"] == "doc1_chunk_0_emb"
        assert d["chunk_id"] == "doc1_chunk_0"
        assert d["embedding"] == [0.1, 0.2]

    def test_from_dict(self):
        data = {
            "doc_id": "doc1",
            "chunk_index": 2,
            "embedding": [0.5, 0.6],
            "model": "custom-model",
            "task": "classification",
        }
        emb = ChunkEmbedding.from_dict(data)
        assert emb.doc_id == "doc1"
        assert emb.chunk_index == 2
        assert emb.embedding == [0.5, 0.6]
        assert emb.model == "custom-model"
        assert emb.task == "classification"
