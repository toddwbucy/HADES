"""ArangoDB implementation of the StorageBackend protocol.

Wraps ArangoHttp2Client with collection-profile support so callers
don't need to know physical collection names or key formats.

Usage:
    from core.database.arango.backend import ArangoBackend
    from core.database.schemas import DocumentMetadata, Chunk, ChunkEmbedding

    backend = ArangoBackend.from_config(cli_config)

    # Using schema objects (preferred)
    backend.store_document("my-doc", metadata_obj, chunk_objs, embedding_objs)

    # Using dicts (backward compatible)
    backend.store_document("2501_12345", meta_dict, chunk_dicts, emb_dicts)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from core.database.arango.optimized_client import (
    ArangoHttp2Client,
    ArangoHttp2Config,
    ArangoHttpError,
)
from core.database.collections import CollectionProfile, get_profile
from core.database.keys import chunk_key, embedding_key, normalize_document_key

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a schema object or dict to a dict.

    Handles DocumentMetadata, Chunk, and ChunkEmbedding objects,
    as well as plain dicts for backward compatibility.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Cannot convert {type(obj).__name__} to dict")


class ArangoBackend:
    """ArangoDB storage backend.

    Implements the StorageBackend protocol defined in core.tools.store.
    """

    def __init__(
        self,
        client: ArangoHttp2Client,
        profile: CollectionProfile | None = None,
    ) -> None:
        self._client = client
        self._profile = profile or get_profile("arxiv")
        self._vector_index_cache: bool | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        read_only: bool = False,
        profile_name: str = "arxiv",
    ) -> ArangoBackend:
        """Create from a CLIConfig instance.

        Args:
            config: CLIConfig from core.cli.config.
            read_only: Use the read-only socket.
            profile_name: Collection profile name.
        """
        from core.cli.config import get_arango_config

        arango_cfg = get_arango_config(config, read_only=read_only)
        client_cfg = ArangoHttp2Config(
            database=arango_cfg["database"],
            socket_path=arango_cfg.get("socket_path"),
            base_url=f"http://{arango_cfg['host']}:{arango_cfg['port']}",
            username=arango_cfg["username"],
            password=arango_cfg["password"],
        )
        client = ArangoHttp2Client(client_cfg)
        return cls(client, get_profile(profile_name))

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def store_document(
        self,
        doc_id: str,
        metadata: Any,
        chunks: list[Any],
        embeddings: list[Any],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Store a document with its chunks and embeddings.

        Args:
            doc_id: Unique document identifier.
            metadata: Document metadata (DocumentMetadata or dict).
            chunks: List of chunks (Chunk objects or dicts).
            embeddings: List of embeddings (ChunkEmbedding objects or dicts).
            overwrite: Replace existing data for this doc_id.

        Returns:
            Summary dict with counts of stored items.
        """
        key = normalize_document_key(doc_id)
        now = datetime.now(UTC).isoformat()
        col = self._profile

        # Convert schema objects to dicts if needed
        meta_dict = _to_dict(metadata)
        meta_doc = {"_key": key, "document_id": doc_id, **meta_dict, "created_at": now}

        chunk_docs = []
        embedding_docs = []

        # Validate embeddings match chunks to avoid silent data loss
        if embeddings and len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count ({len(embeddings)}) does not match chunks ({len(chunks)})")

        for i, chunk in enumerate(chunks):
            chunk_dict = _to_dict(chunk)
            ck = chunk_key(key, i)
            chunk_docs.append(
                {
                    **chunk_dict,  # Chunk fields first, then override with our values
                    "_key": ck,
                    "document_id": doc_id,
                    "paper_key": key,
                    "chunk_index": i,  # Must match _key derivation
                    "total_chunks": len(chunks),
                    "created_at": now,
                }
            )
            if i < len(embeddings):
                emb_dict = _to_dict(embeddings[i])
                embedding_docs.append(
                    {
                        **emb_dict,  # Embedding fields first, then override with our values
                        "_key": embedding_key(ck),
                        "chunk_key": ck,
                        "document_id": doc_id,
                        "paper_key": key,
                        "created_at": now,
                    }
                )

        if chunk_docs:
            self._client.insert_documents(col.chunks, chunk_docs, overwrite=overwrite)
        if embedding_docs:
            self._client.insert_documents(col.embeddings, embedding_docs, overwrite=overwrite)
        self._client.insert_documents(col.metadata, [meta_doc], overwrite=overwrite)

        return {
            "doc_id": doc_id,
            "metadata": 1,
            "chunks": len(chunk_docs),
            "embeddings": len(embedding_docs),
        }

    def query_similar(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        doc_filter: str | None = None,
        n_probe: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings using ANN (if available) or brute-force fallback.

        Args:
            query_embedding: Query vector as a list of floats.
            limit: Maximum results to return.
            doc_filter: Optional document ID to restrict search.
            n_probe: Override nProbe for ANN search (higher = better recall, slower).

        Returns:
            List of result dicts sorted by similarity score (descending).
        """
        if self._has_vector_index() and not doc_filter:
            # Fast path: server-side ANN via FAISS-backed inverted index
            return self._query_ann(query_embedding, limit=limit, n_probe=n_probe)
        else:
            # Slow path: brute-force cosine similarity in Python
            return self._query_brute_force(query_embedding, limit=limit, doc_filter=doc_filter)

    def _has_vector_index(self) -> bool:
        """Check if the embeddings collection has a vector index. Cached per-instance."""
        if self._vector_index_cache is not None:
            return self._vector_index_cache

        try:
            indexes = self._client.list_indexes(self._profile.embeddings)
            for idx in indexes:
                # Check for inverted index with vector features
                if idx.get("type") == "inverted":
                    for field in idx.get("fields", []):
                        if isinstance(field, dict) and "vector" in (field.get("features", []) or []):
                            self._vector_index_cache = True
                            return True
                        # ArangoDB may also return vector config directly on the field
                        if isinstance(field, dict) and field.get("vector"):
                            self._vector_index_cache = True
                            return True
        except ArangoHttpError:
            pass

        self._vector_index_cache = False
        return False

    def _query_ann(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        n_probe: int | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search using APPROX_NEAR_VECTORS — runs server-side via FAISS index."""
        col = self._profile

        options_clause = ""
        bind_vars: dict[str, Any] = {
            "query_vec": query_embedding,
            "limit": limit,
        }
        if n_probe is not None:
            options_clause = "OPTIONS { nProbe: @n_probe }"
            bind_vars["n_probe"] = n_probe

        aql = f"""
            FOR emb IN APPROX_NEAR_VECTORS(
                {col.embeddings}, "embedding", @query_vec, @limit {options_clause}
            )
                FILTER emb.chunk_key != null
                LET chunk = DOCUMENT(CONCAT("{col.chunks}/", emb.chunk_key))
                LET meta = DOCUMENT(CONCAT("{col.metadata}/", emb.paper_key))
                RETURN {{
                    paper_key: emb.paper_key,
                    text: chunk.text,
                    chunk_index: chunk.chunk_index,
                    total_chunks: chunk.total_chunks,
                    title: meta.title,
                    arxiv_id: meta.arxiv_id,
                    score: emb.$score
                }}
        """
        logger.debug("Using ANN search (vector index) with limit=%d", limit)
        return self._client.query(aql, bind_vars=bind_vars)

    def _query_brute_force(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity — fetches all embeddings to Python."""
        import numpy as np

        col = self._profile
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        filter_clause = ""
        bind_vars: dict[str, Any] = {}
        if doc_filter:
            filter_clause = "FILTER emb.paper_key == @paper_key"
            bind_vars["paper_key"] = normalize_document_key(doc_filter)

        aql = f"""
            FOR emb IN {col.embeddings}
                FILTER emb.chunk_key != null
                {filter_clause}
                LET chunk = DOCUMENT(CONCAT("{col.chunks}/", emb.chunk_key))
                LET meta = DOCUMENT(CONCAT("{col.metadata}/", emb.paper_key))
                RETURN {{
                    paper_key: emb.paper_key,
                    embedding: emb.embedding,
                    text: chunk.text,
                    chunk_index: chunk.chunk_index,
                    total_chunks: chunk.total_chunks,
                    title: meta.title,
                    arxiv_id: meta.arxiv_id
                }}
        """
        logger.debug("Using brute-force search (no vector index)")
        results = self._client.query(aql, bind_vars=bind_vars)

        scored = []
        for r in results:
            emb = r.get("embedding")
            if emb is None:
                continue
            emb_arr = np.array(emb, dtype=np.float32)
            emb_norm = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
            score = float(np.dot(query_norm, emb_norm))
            scored.append({**r, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Remove raw embedding from output
        for item in scored[:limit]:
            item.pop("embedding", None)
        return scored[:limit]

    def purge_document(self, doc_id: str) -> dict[str, Any]:
        col = self._profile
        key = normalize_document_key(doc_id)

        aql = f"""
            LET meta = (FOR d IN {col.metadata} FILTER d._key == @key REMOVE d IN {col.metadata} RETURN 1)
            LET chunks = (FOR d IN {col.chunks} FILTER d.paper_key == @key REMOVE d IN {col.chunks} RETURN 1)
            LET embs = (FOR d IN {col.embeddings} FILTER d.paper_key == @key REMOVE d IN {col.embeddings} RETURN 1)
            RETURN {{metadata: LENGTH(meta), chunks: LENGTH(chunks), embeddings: LENGTH(embs)}}
        """
        results = self._client.query(aql, bind_vars={"key": key})
        return results[0] if results else {"metadata": 0, "chunks": 0, "embeddings": 0}

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        key = normalize_document_key(doc_id)
        try:
            return self._client.get_document(self._profile.metadata, key)
        except ArangoHttpError as e:
            if e.status_code == 404:
                return None
            raise

    def list_documents(
        self,
        *,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = self._profile
        filter_clause = ""
        bind_vars: dict[str, Any] = {"limit": limit}

        if filters and "category" in filters:
            filter_clause = "FILTER @category IN doc.categories"
            bind_vars["category"] = filters["category"]

        aql = f"""
            FOR doc IN {col.metadata}
                {filter_clause}
                SORT doc.processing_timestamp DESC
                LIMIT @limit
                RETURN {{
                    document_id: doc.document_id,
                    arxiv_id: doc.arxiv_id,
                    title: doc.title,
                    num_chunks: doc.num_chunks,
                    source: doc.source
                }}
        """
        return self._client.query(aql, bind_vars=bind_vars)

    def stats(self) -> dict[str, Any]:
        col = self._profile
        paper_count = self._client.query(f"RETURN LENGTH({col.metadata})")
        chunk_count = self._client.query(f"RETURN LENGTH({col.chunks})")
        emb_count = self._client.query(f"RETURN LENGTH({col.embeddings})")
        return {
            "total_papers": paper_count[0] if paper_count else 0,
            "total_chunks": chunk_count[0] if chunk_count else 0,
            "total_embeddings": emb_count[0] if emb_count else 0,
            "profile": {
                "metadata": col.metadata,
                "chunks": col.chunks,
                "embeddings": col.embeddings,
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ArangoBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
