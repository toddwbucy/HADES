"""Standalone embedding tool.

Wraps Jina v4 (multimodal, 32k context, 2048-dim) via EmbedderClient with
local fallback.  Each function is self-contained — no database, no extraction.

Primary operations:
    embed_texts         — standard text embedding (batch)
    embed_text          — single-text convenience
    embed_with_late_chunking — full-document encode then segment (main ingest mode)
    embed_image         — multimodal image embedding

Usage:
    from core.tools.embed import embed_text, embed_texts, embed_with_late_chunking

    vec = embed_text("hello world")
    vecs = embed_texts(["a", "b", "c"])
    chunks = embed_with_late_chunking(open("paper.txt").read())
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_client(**kwargs: Any):
    """Create an EmbedderClient with sensible defaults."""
    from core.services.embedder_client import EmbedderClient

    defaults = {
        "socket_path": "/run/hades/embedder.sock",
        "timeout": 30.0,
        "fallback_to_local": True,
    }
    defaults.update(kwargs)
    return EmbedderClient(**defaults)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    *,
    task: str = "retrieval.passage",
    batch_size: int | None = None,
    **client_kwargs: Any,
) -> np.ndarray:
    """Embed multiple texts.

    Args:
        texts: Texts to embed.
        task: Jina task type (retrieval.passage, retrieval.query, text-matching).
        batch_size: Optional batch size override.
        **client_kwargs: Forwarded to EmbedderClient constructor.

    Returns:
        (N, 2048) float32 numpy array.
    """
    client = _get_client(**client_kwargs)
    try:
        return client.embed_texts(texts, task=task, batch_size=batch_size)
    finally:
        client.close()


def embed_text(
    text: str,
    *,
    task: str = "retrieval.passage",
    **client_kwargs: Any,
) -> np.ndarray:
    """Embed a single text string.

    Args:
        text: Text to embed.
        task: Jina task type.
        **client_kwargs: Forwarded to EmbedderClient constructor.

    Returns:
        1-D float32 numpy array (2048,).
    """
    result = embed_texts([text], task=task, **client_kwargs)
    return result[0]


def embed_with_late_chunking(
    full_text: str,
    *,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 200,
    device: str | None = None,
    use_fp16: bool = True,
) -> list[dict[str, Any]]:
    """Encode the full document first, then segment into context-aware chunks.

    Late chunking preserves document-level semantic context — each chunk's
    embedding is informed by the *entire* document, not just the chunk text.
    This is the primary mode for ingestion.

    Args:
        full_text: Complete document text.
        chunk_size_tokens: Target chunk size in tokens.
        chunk_overlap_tokens: Token overlap between adjacent chunks.
        device: PyTorch device (default: cuda if available).
        use_fp16: Use half-precision for lower memory usage.

    Returns:
        List of dicts, each with keys:
            text, embedding (list[float]), start_char, end_char,
            chunk_index, total_chunks, context_window_used
    """
    from core.embedders import EmbeddingConfig
    from core.embedders.embedders_jina import JinaV4Embedder

    config = EmbeddingConfig(
        device=device or "cuda",
        use_fp16=use_fp16,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    embedder = JinaV4Embedder(config)
    chunks = embedder.embed_with_late_chunking(full_text)

    return [
        {
            "text": c.text,
            "embedding": np.asarray(c.embedding).tolist(),
            "start_char": c.start_char,
            "end_char": c.end_char,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            "context_window_used": c.context_window_used,
        }
        for c in chunks
    ]


def embed_image(
    image_path: str,
    *,
    device: str | None = None,
    use_fp16: bool = True,
) -> np.ndarray:
    """Embed an image using Jina v4's multimodal capability.

    Args:
        image_path: Path to image file (PNG, JPG, TIFF, BMP).
        device: PyTorch device.
        use_fp16: Use half-precision.

    Returns:
        1-D float32 numpy array (2048,).
    """
    from core.embedders import EmbeddingConfig
    from core.embedders.embedders_jina import JinaV4Embedder

    config = EmbeddingConfig(
        device=device or "cuda",
        use_fp16=use_fp16,
    )
    embedder = JinaV4Embedder(config)
    return embedder.embed_image(image_path)
