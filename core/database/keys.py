"""Document key normalization for ArangoDB.

ArangoDB document keys have restricted characters. This module centralises
the rules so every call site produces identical keys for the same input.

Rules:
    - Replace '.' and '/' with '_'  (ArangoDB key-safe)
    - Strip trailing arxiv version suffix (e.g. v1, v2)
"""

from __future__ import annotations

import re

_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_document_key(raw_id: str) -> str:
    """Normalise a document identifier into an ArangoDB-safe key.

    Strips an optional trailing version suffix (``v1``, ``v2``, …) and
    replaces ``'.'`` and ``'/'`` with ``'_'``.

    Examples:
        >>> normalize_document_key("2501.12345v2")
        '2501_12345'
        >>> normalize_document_key("hep-th/9901001")
        'hep-th_9901001'
        >>> normalize_document_key("my_local_doc")
        'my_local_doc'
    """
    base = _VERSION_SUFFIX.sub("", raw_id)
    return base.replace(".", "_").replace("/", "_")


def chunk_key(document_key: str, chunk_index: int) -> str:
    """Build a chunk document key.

    Args:
        document_key: Already-normalised document key.
        chunk_index: Zero-based chunk index.

    Returns:
        Key like ``'2501_12345_chunk_0'``
    """
    return f"{document_key}_chunk_{chunk_index}"


def embedding_key(chunk_key_value: str) -> str:
    """Build an embedding document key from its chunk key.

    Args:
        chunk_key_value: The chunk's ``_key``.

    Returns:
        Key like ``'2501_12345_chunk_0_emb'``
    """
    return f"{chunk_key_value}_emb"


def file_key(rel_path: str) -> str:
    """Normalise a relative file path into an ArangoDB-safe key.

    Replaces ``'/'`` and ``'.'`` with ``'_'``.

    Examples:
        >>> file_key("core/persephone/models.py")
        'core_persephone_models_py'
        >>> file_key("setup.py")
        'setup_py'
    """
    return rel_path.replace("/", "_").replace(".", "_")


def strip_version(arxiv_id: str) -> str:
    """Strip the trailing version suffix from an arxiv ID.

    Unlike :func:`normalize_document_key` this does **not** replace
    ``'.'`` or ``'/'`` — it returns a clean arxiv ID, not a DB key.

    Examples:
        >>> strip_version("2501.12345v1")
        '2501.12345'
        >>> strip_version("2501.12345")
        '2501.12345'
    """
    return _VERSION_SUFFIX.sub("", arxiv_id)
