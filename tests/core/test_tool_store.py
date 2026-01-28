"""Tests for core.tools.store — StorageBackend protocol."""

from __future__ import annotations

from typing import Any

import pytest

from core.tools.store import StorageBackend


class DummyBackend:
    """Minimal implementation for protocol conformance testing."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    def store_document(self, doc_id, metadata, chunks, embeddings, *, overwrite=False) -> dict[str, Any]:
        self._docs[doc_id] = {"metadata": metadata, "chunks": chunks}
        return {"doc_id": doc_id, "chunks": len(chunks)}

    def query_similar(self, query_embedding, *, limit=10, doc_filter=None):
        return []

    def purge_document(self, doc_id):
        self._docs.pop(doc_id, None)
        return {"metadata": 1, "chunks": 0, "embeddings": 0}

    def get_document(self, doc_id):
        return self._docs.get(doc_id)

    def list_documents(self, *, limit=20, filters=None):
        return list(self._docs.values())[:limit]

    def stats(self):
        return {"total_documents": len(self._docs)}


class TestStorageBackendProtocol:
    def test_dummy_is_storage_backend(self):
        backend = DummyBackend()
        assert isinstance(backend, StorageBackend)

    def test_store_and_get(self):
        backend = DummyBackend()
        backend.store_document("doc1", {"title": "Test"}, [{"text": "chunk"}], [])
        doc = backend.get_document("doc1")
        assert doc is not None
        assert doc["metadata"]["title"] == "Test"

    def test_purge(self):
        backend = DummyBackend()
        backend.store_document("doc1", {}, [], [])
        backend.purge_document("doc1")
        assert backend.get_document("doc1") is None

    def test_list_documents(self):
        backend = DummyBackend()
        backend.store_document("a", {}, [], [])
        backend.store_document("b", {}, [], [])
        docs = backend.list_documents(limit=1)
        assert len(docs) == 1

    def test_stats(self):
        backend = DummyBackend()
        backend.store_document("x", {}, [], [])
        assert backend.stats()["total_documents"] == 1

    def test_non_conforming_object_fails(self):
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), StorageBackend)
