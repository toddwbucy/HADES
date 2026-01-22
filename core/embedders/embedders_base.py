#!/usr/bin/env python3
"""
Base Embedder Interface

Defines the contract for all embedding implementations in HADES.
Embedders transform text content into vector representations while
preserving semantic relationships for similarity search.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedders."""
    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 8192
    use_fp16: bool = True
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None


class EmbedderBase(ABC):
    """
    Abstract base class for all embedders.

    Defines the interface that all embedding implementations must follow
    to ensure consistency across different models and approaches.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedder with configuration.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig(model_name="default")

    @abstractmethod
    def embed_texts(self,
                    texts: List[str],
                    task: str = "retrieval",
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed
            task: Task type (retrieval, classification, etc.)
            batch_size: Override default batch size

        Returns:
            Array of embeddings (N x D)
        """
        pass

    @abstractmethod
    def embed_single(self,
                     text: str,
                     task: str = "retrieval") -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Text to embed
            task: Task type

        Returns:
            Embedding vector (1D array)
        """
        pass

    def embed_queries(self,
                     queries: List[str],
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed search queries (convenience method).

        Args:
            queries: List of search queries
            batch_size: Override default batch size

        Returns:
            Array of query embeddings
        """
        return self.embed_texts(queries, task="retrieval", batch_size=batch_size)

    def embed_documents(self,
                       documents: List[str],
                       batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed documents for retrieval (convenience method).

        Args:
            documents: List of documents
            batch_size: Override default batch size

        Returns:
            Array of document embeddings
        """
        return self.embed_texts(documents, task="retrieval", batch_size=batch_size)

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported."""
        pass

    @property
    def supports_late_chunking(self) -> bool:
        """Whether this embedder supports late chunking."""
        return False

    @property
    def supports_multimodal(self) -> bool:
        """Whether this embedder supports multimodal inputs."""
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.max_sequence_length,
            "supports_late_chunking": self.supports_late_chunking,
            "supports_multimodal": self.supports_multimodal,
            "device": self.config.device,
            "use_fp16": self.config.use_fp16
        }