"""Abstract storage backend protocol.

The database is a pluggable backend — extract and embed are the core of HADES,
the database just stores and retrieves what they produce.

This module defines the interface that any storage backend must implement.
ArangoDB is the default (and currently only) backend.

Usage:
    from core.tools.store import StorageBackend
    from core.database.schemas import DocumentMetadata, Chunk, ChunkEmbedding

    # Using schema objects (preferred)
    def ingest(backend: StorageBackend, metadata, chunks, embeddings):
        backend.store_document(
            metadata.doc_id,
            metadata,
            chunks,
            embeddings,
        )

    # Using dicts (backward compatible)
    def ingest_dicts(backend: StorageBackend, doc_id, meta_dict, chunk_dicts, emb_dicts):
        backend.store_document(doc_id, meta_dict, chunk_dicts, emb_dicts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from core.database.schemas import Chunk, ChunkEmbedding, DocumentMetadata

# Type aliases for flexibility — accepts both schema objects and dicts
MetadataType = Union["DocumentMetadata", dict[str, Any]]
ChunkType = Union["Chunk", dict[str, Any]]
EmbeddingType = Union["ChunkEmbedding", dict[str, Any]]


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable storage backends.

    Implementations:
        core.database.arango.backend.ArangoBackend  (default)
        # Future: PostgreSQL/pgvector, Neo4j, Milvus

    Methods accept both schema dataclasses (preferred) and plain dicts
    for backward compatibility.
    """

    def store_document(
        self,
        doc_id: str,
        metadata: MetadataType,
        chunks: list[ChunkType],
        embeddings: list[EmbeddingType],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Store a document with its chunks and embeddings.

        Args:
            doc_id: Unique document identifier.
            metadata: Document-level metadata (DocumentMetadata or dict).
            chunks: List of chunks (Chunk objects or dicts).
            embeddings: List of embeddings (ChunkEmbedding objects or dicts).
            overwrite: Replace existing data for this doc_id.

        Returns:
            Summary dict with counts of stored items.
        """
        ...

    def query_similar(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to a query embedding.

        Args:
            query_embedding: Query vector.
            limit: Max results.
            doc_filter: Optional document ID to restrict search.

        Returns:
            List of result dicts with text, score, doc_id, chunk_index.
        """
        ...

    def purge_document(self, doc_id: str) -> dict[str, Any]:
        """Remove all data for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            Summary dict with counts of removed items.
        """
        ...

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve document metadata.

        Args:
            doc_id: Document identifier.

        Returns:
            Metadata dict, or None if not found.
        """
        ...

    def list_documents(
        self,
        *,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List stored documents.

        Args:
            limit: Max results.
            filters: Optional filter criteria.

        Returns:
            List of document metadata dicts.
        """
        ...

    def stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dict with counts, sizes, and backend-specific info.
        """
        ...
