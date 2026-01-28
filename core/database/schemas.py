"""Source-agnostic document schema definitions.

These dataclasses define the canonical representation for documents, chunks,
and embeddings across all storage backends. Source-specific fields (arxiv ID,
DOI, URL, etc.) go in `source_metadata`.

Usage:
    from core.database.schemas import DocumentMetadata, Chunk, ChunkEmbedding

    doc = DocumentMetadata(
        doc_id="my-document",
        title="Example Document",
        source_metadata={"arxiv_id": "2501.12345"},
    )

    chunk = Chunk(
        doc_id="my-document",
        chunk_index=0,
        text="First paragraph...",
    )

    embedding = ChunkEmbedding(
        doc_id="my-document",
        chunk_index=0,
        embedding=[0.1, 0.2, ...],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class DocumentMetadata:
    """Metadata for a document in the knowledge base.

    This is source-agnostic — arxiv, PDF, web page, etc. all use the same
    structure. Source-specific fields go in `source_metadata`.

    Attributes:
        doc_id: Unique document identifier (user-provided or derived from source).
        title: Document title.
        authors: List of author names (empty if unknown).
        created_at: When the document was added to the knowledge base.
        source_type: Source type hint (e.g., "arxiv", "pdf", "url").
        source_metadata: Source-specific fields (arxiv_id, doi, url, etc.).
        tags: Optional user-defined tags for organization.
        extra: Any additional metadata.
    """

    doc_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    source_type: str = "unknown"
    source_metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "authors": self.authors,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_type": self.source_type,
            "source_metadata": self.source_metadata,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", ""),
            authors=data.get("authors", []),
            created_at=created_at,
            source_type=data.get("source_type", "unknown"),
            source_metadata=data.get("source_metadata", {}),
            tags=data.get("tags", []),
            extra=data.get("extra", {}),
        )


@dataclass
class Chunk:
    """A text chunk from a document.

    Chunks are segments of document text, typically created via late chunking
    to preserve document-level context.

    Attributes:
        doc_id: Parent document identifier.
        chunk_index: Position in the document (0-indexed).
        text: The chunk text content.
        start_char: Character offset where chunk starts in source document.
        end_char: Character offset where chunk ends in source document.
        total_chunks: Total number of chunks in the parent document.
        metadata: Optional chunk-level metadata (section headers, page numbers, etc.).
    """

    doc_id: str
    chunk_index: int
    text: str
    start_char: int | None = None
    end_char: int | None = None
    total_chunks: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate a unique chunk identifier."""
        return f"{self.doc_id}_chunk_{self.chunk_index}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            chunk_index=data["chunk_index"],
            text=data["text"],
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            total_chunks=data.get("total_chunks"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChunkEmbedding:
    """Embedding vector for a chunk.

    Embeddings are stored separately from chunks to allow different embedding
    models or re-embedding without touching the text data.

    Attributes:
        doc_id: Parent document identifier.
        chunk_index: Chunk position (matches Chunk.chunk_index).
        embedding: The embedding vector (typically 2048-dim for Jina v4).
        model: Name/version of the embedding model used.
        task: Embedding task type (e.g., "retrieval.passage").
    """

    doc_id: str
    chunk_index: int
    embedding: list[float]
    model: str = "jina-embeddings-v4"
    task: str = "retrieval.passage"

    @property
    def embedding_id(self) -> str:
        """Generate a unique embedding identifier."""
        return f"{self.doc_id}_chunk_{self.chunk_index}_emb"

    @property
    def chunk_id(self) -> str:
        """Get the associated chunk ID."""
        return f"{self.doc_id}_chunk_{self.chunk_index}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "embedding_id": self.embedding_id,
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "model": self.model,
            "task": self.task,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkEmbedding:
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            chunk_index=data["chunk_index"],
            embedding=data["embedding"],
            model=data.get("model", "jina-embeddings-v4"),
            task=data.get("task", "retrieval.passage"),
        )
