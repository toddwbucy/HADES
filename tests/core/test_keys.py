"""Tests for core.database.keys — document key normalization."""

import pytest

from core.database.keys import (
    chunk_key,
    embedding_key,
    normalize_document_key,
    strip_version,
)


class TestNormalizeDocumentKey:
    def test_dots_replaced(self):
        assert normalize_document_key("2501.12345") == "2501_12345"

    def test_slashes_replaced(self):
        assert normalize_document_key("hep-th/9901001") == "hep-th_9901001"

    def test_version_stripped(self):
        assert normalize_document_key("2501.12345v2") == "2501_12345"

    def test_version_and_slash(self):
        assert normalize_document_key("hep-th/9901001v1") == "hep-th_9901001"

    def test_no_change_needed(self):
        assert normalize_document_key("my_local_doc") == "my_local_doc"

    def test_multi_digit_version(self):
        assert normalize_document_key("2501.12345v12") == "2501_12345"

    def test_empty_string(self):
        assert normalize_document_key("") == ""


class TestChunkKey:
    def test_basic(self):
        assert chunk_key("2501_12345", 0) == "2501_12345_chunk_0"

    def test_large_index(self):
        assert chunk_key("doc", 999) == "doc_chunk_999"


class TestEmbeddingKey:
    def test_basic(self):
        assert embedding_key("2501_12345_chunk_0") == "2501_12345_chunk_0_emb"


class TestStripVersion:
    def test_strips_v1(self):
        assert strip_version("2501.12345v1") == "2501.12345"

    def test_no_version(self):
        assert strip_version("2501.12345") == "2501.12345"

    def test_preserves_dots(self):
        # strip_version does NOT replace dots — that's normalize_document_key's job
        assert strip_version("2501.12345v2") == "2501.12345"

    def test_preserves_slashes(self):
        assert strip_version("hep-th/9901001v1") == "hep-th/9901001"
