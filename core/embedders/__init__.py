"""
Embedders Module

Provides embedding models for transforming text into vector representations.
Supports late chunking for superior context preservation in long documents.
"""

from .embedders_base import EmbedderBase, EmbeddingConfig
from .embedders_factory import EmbedderFactory

# Auto-register available embedders
from typing import Optional, Type

JinaV4Embedder: Optional[Type] = None
ChunkWithEmbedding: Optional[Type] = None

try:
    from .embedders_jina import ChunkWithEmbedding as _ChunkWithEmbedding
    from .embedders_jina import JinaV4Embedder as _JinaV4Embedder
    JinaV4Embedder = _JinaV4Embedder
    ChunkWithEmbedding = _ChunkWithEmbedding
    EmbedderFactory.register("jina", _JinaV4Embedder)
except ImportError:
    pass

# Backward compatibility exports
__all__ = [
    'EmbedderBase',
    'EmbeddingConfig',
    'EmbedderFactory',
    'JinaV4Embedder',
    'ChunkWithEmbedding',
]

# Convenience function for backward compatibility
def create_embedder(model_name: str = "jinaai/jina-embeddings-v4", **kwargs):
    """
    Create an embedder instance (backward compatibility).

    Args:
        model_name: Model name or path
        **kwargs: Additional configuration

    Returns:
        Embedder instance
    """
    return EmbedderFactory.create(model_name, **kwargs)
