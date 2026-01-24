"""Unit tests for core.embedders.embedders_base module."""

import numpy as np
import pytest

from core.embedders.embedders_base import EmbedderBase, EmbeddingConfig


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_values(self) -> None:
        """EmbeddingConfig should have sensible defaults."""
        config = EmbeddingConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.max_seq_length == 8192
        assert config.use_fp16 is True
        assert config.chunk_size_tokens is None
        assert config.chunk_overlap_tokens is None

    def test_custom_values(self) -> None:
        """EmbeddingConfig should accept custom values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            device="cpu",
            batch_size=16,
            max_seq_length=4096,
            use_fp16=False,
            chunk_size_tokens=512,
            chunk_overlap_tokens=64,
        )
        assert config.model_name == "custom-model"
        assert config.device == "cpu"
        assert config.batch_size == 16
        assert config.max_seq_length == 4096
        assert config.use_fp16 is False
        assert config.chunk_size_tokens == 512
        assert config.chunk_overlap_tokens == 64


class ConcreteEmbedder(EmbedderBase):
    """Concrete implementation for testing EmbedderBase."""

    EMBEDDING_DIM = 768
    MAX_SEQ_LEN = 512

    def embed_texts(
        self,
        texts: list[str],
        task: str = "retrieval",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return fake embeddings."""
        return np.random.randn(len(texts), self.EMBEDDING_DIM).astype(np.float32)

    def embed_single(
        self,
        text: str,
        task: str = "retrieval",
    ) -> np.ndarray:
        """Return fake embedding."""
        return np.random.randn(self.EMBEDDING_DIM).astype(np.float32)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.EMBEDDING_DIM

    @property
    def max_sequence_length(self) -> int:
        """Get max sequence length."""
        return self.MAX_SEQ_LEN


class TestEmbedderBase:
    """Tests for EmbedderBase abstract class."""

    @pytest.fixture
    def embedder(self) -> ConcreteEmbedder:
        """Create a concrete embedder instance."""
        return ConcreteEmbedder()

    @pytest.fixture
    def embedder_with_config(self) -> ConcreteEmbedder:
        """Create embedder with custom config."""
        config = EmbeddingConfig(
            model_name="test-model",
            device="cpu",
            batch_size=8,
        )
        return ConcreteEmbedder(config)

    def test_default_config(self, embedder: ConcreteEmbedder) -> None:
        """Embedder should have default config when none provided."""
        assert embedder.config is not None
        assert embedder.config.model_name == "default"

    def test_custom_config(self, embedder_with_config: ConcreteEmbedder) -> None:
        """Embedder should use provided config."""
        assert embedder_with_config.config.model_name == "test-model"
        assert embedder_with_config.config.device == "cpu"
        assert embedder_with_config.config.batch_size == 8

    def test_embed_texts_returns_correct_shape(self, embedder: ConcreteEmbedder) -> None:
        """embed_texts should return array with correct shape."""
        texts = ["hello", "world", "test"]
        embeddings = embedder.embed_texts(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    def test_embed_single_returns_correct_shape(self, embedder: ConcreteEmbedder) -> None:
        """embed_single should return 1D array."""
        embedding = embedder.embed_single("test text")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_embed_queries_calls_embed_texts(self, embedder: ConcreteEmbedder) -> None:
        """embed_queries should use embed_texts internally."""
        queries = ["query 1", "query 2"]
        embeddings = embedder.embed_queries(queries)
        assert embeddings.shape == (2, 768)

    def test_embed_documents_calls_embed_texts(self, embedder: ConcreteEmbedder) -> None:
        """embed_documents should use embed_texts internally."""
        docs = ["doc 1", "doc 2", "doc 3"]
        embeddings = embedder.embed_documents(docs)
        assert embeddings.shape == (3, 768)

    def test_embedding_dimension_property(self, embedder: ConcreteEmbedder) -> None:
        """embedding_dimension property should return correct value."""
        assert embedder.embedding_dimension == 768

    def test_max_sequence_length_property(self, embedder: ConcreteEmbedder) -> None:
        """max_sequence_length property should return correct value."""
        assert embedder.max_sequence_length == 512

    def test_supports_late_chunking_default(self, embedder: ConcreteEmbedder) -> None:
        """supports_late_chunking should default to False."""
        assert embedder.supports_late_chunking is False

    def test_supports_multimodal_default(self, embedder: ConcreteEmbedder) -> None:
        """supports_multimodal should default to False."""
        assert embedder.supports_multimodal is False

    def test_get_model_info(self, embedder_with_config: ConcreteEmbedder) -> None:
        """get_model_info should return comprehensive info dict."""
        info = embedder_with_config.get_model_info()
        assert info["model_name"] == "test-model"
        assert info["embedding_dimension"] == 768
        assert info["max_sequence_length"] == 512
        assert info["supports_late_chunking"] is False
        assert info["supports_multimodal"] is False
        assert info["device"] == "cpu"
        assert info["use_fp16"] is True


class LateCunkingEmbedder(ConcreteEmbedder):
    """Embedder that supports late chunking."""

    @property
    def supports_late_chunking(self) -> bool:
        """Enable late chunking support."""
        return True


class TestEmbedderCapabilities:
    """Tests for embedder capability flags."""

    def test_late_chunking_can_be_enabled(self) -> None:
        """Embedders can override supports_late_chunking."""
        embedder = LateCunkingEmbedder()
        assert embedder.supports_late_chunking is True

    def test_model_info_reflects_capabilities(self) -> None:
        """get_model_info should reflect actual capabilities."""
        embedder = LateCunkingEmbedder()
        info = embedder.get_model_info()
        assert info["supports_late_chunking"] is True
