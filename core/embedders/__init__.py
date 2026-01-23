"""
Embedders Module

Provides embedding models for transforming text into vector representations.
Supports late chunking for superior context preservation in long documents.
"""

from .embedders_base import EmbedderBase, EmbeddingConfig
from .embedders_factory import EmbedderFactory

# Auto-register available embedders
try:
    from .embedders_jina import JinaV4Embedder, ChunkWithEmbedding
    EmbedderFactory.register("jina", JinaV4Embedder)
except ImportError:
    JinaV4Embedder = None  # type: ignore[misc]
    ChunkWithEmbedding = None  # type: ignore[misc]

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
