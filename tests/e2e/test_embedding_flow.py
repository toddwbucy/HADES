"""End-to-end tests for the embedding phase of the pipeline.

Tests embedder initialization, text embedding, and late chunking.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbedderFactory:
    """Test the embedder factory pattern."""

    def test_factory_creates_jina_embedder(self) -> None:
        """Factory should create JinaV4Embedder for jina model."""
        from core.embedders import EmbedderFactory, EmbeddingConfig

        config = EmbeddingConfig(
            model_name="jinaai/jina-embeddings-v4",
            device="cpu",
            batch_size=8,
        )

        # This may fail if Jina model isn't downloaded, so we catch that
        try:
            embedder = EmbedderFactory.create(
                model_name="jinaai/jina-embeddings-v4",
                config=config,
            )
            assert embedder is not None
        except (RuntimeError, OSError) as e:
            # Model not downloaded or CUDA not available
            if "CUDA" in str(e) or "model" in str(e).lower():
                pytest.skip("Jina model not available")
            raise

    def test_factory_registers_embedders(self) -> None:
        """Factory should have embedders registered."""
        from core.embedders import EmbedderFactory

        # Check that we can access the factory
        assert hasattr(EmbedderFactory, "create")
        assert hasattr(EmbedderFactory, "register")


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_values(self) -> None:
        """EmbeddingConfig should have sensible defaults."""
        from core.embedders import EmbeddingConfig

        # model_name is required, device defaults to "cuda"
        config = EmbeddingConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.device == "cuda"  # default
        assert config.batch_size == 32  # default
        assert config.max_seq_length == 8192  # default

    def test_custom_values(self) -> None:
        """EmbeddingConfig should accept custom values."""
        from core.embedders import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="custom-model",
            device="cpu",
            batch_size=16,
            use_fp16=False,
        )
        assert config.model_name == "custom-model"
        assert config.device == "cpu"
        assert config.batch_size == 16
        assert config.use_fp16 is False


class TestMockEmbedding:
    """Test embedding with mock embedder to verify flow."""

    def test_mock_embedder_produces_embeddings(self, mock_embedder: Any) -> None:
        """Mock embedder should produce valid embeddings."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = mock_embedder.embed_texts(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (2048,)
            # Should be normalized (unit vector)
            norm = np.linalg.norm(emb)
            assert 0.99 < norm < 1.01

    def test_mock_embedder_deterministic(self, mock_embedder: Any) -> None:
        """Mock embedder should be deterministic for same input."""
        text = "Test text for determinism"
        emb1 = mock_embedder.embed_texts([text])[0]
        emb2 = mock_embedder.embed_texts([text])[0]

        assert np.allclose(emb1, emb2)

    def test_mock_late_chunking(self, mock_embedder: Any) -> None:
        """Mock embedder should support late chunking."""
        text = " ".join(["word"] * 250)  # 250 words
        chunks = mock_embedder.embed_with_late_chunking(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text is not None
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 2048


class TestJinaEmbedderInterface:
    """Test JinaV4Embedder interface without loading model."""

    @patch("core.embedders.embedders_jina.AutoModel")
    @patch("core.embedders.embedders_jina.AutoTokenizer")
    def test_jina_embedder_initialization(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """JinaV4Embedder should initialize with config."""
        from core.embedders.embedders_jina import JinaV4Embedder

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 8192
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 2048
        mock_model_cls.from_pretrained.return_value = mock_model

        embedder = JinaV4Embedder(device="cpu")
        assert embedder is not None

    @patch("core.embedders.embedders_jina.AutoModel")
    @patch("core.embedders.embedders_jina.AutoTokenizer")
    def test_jina_embedder_embed_texts(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
    ) -> None:
        """JinaV4Embedder.embed_texts should return embeddings."""
        import torch

        from core.embedders.embedders_jina import JinaV4Embedder

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 8192
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 2048

        # Mock model forward pass
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 2048)
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        embedder = JinaV4Embedder(device="cpu")
        texts = ["Test text"]
        embeddings = embedder.embed_texts(texts)

        assert len(embeddings) >= 1


class TestChunkWithEmbedding:
    """Test ChunkWithEmbedding dataclass."""

    def test_chunk_creation(self) -> None:
        """ChunkWithEmbedding should store all fields."""
        from core.embedders.embedders_jina import ChunkWithEmbedding

        embedding = np.random.randn(2048).astype(np.float32)
        chunk = ChunkWithEmbedding(
            text="Sample chunk text",
            embedding=embedding,
            start_char=0,
            end_char=17,
            start_token=0,
            end_token=3,
            chunk_index=0,
            total_chunks=1,
            context_window_used=3,
        )

        assert chunk.text == "Sample chunk text"
        assert np.array_equal(chunk.embedding, embedding)
        assert chunk.start_char == 0
        assert chunk.end_char == 17
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1


class TestEmbeddingSimilarity:
    """Test that embeddings preserve semantic similarity."""

    def test_similar_texts_have_high_similarity(self, mock_embedder: Any) -> None:
        """Similar texts should have similar embeddings."""
        # Note: With mock embedder, similarity is based on hash,
        # so we just verify the mechanics work
        text1 = "Machine learning is great"
        text2 = "Machine learning is great"  # Identical

        emb1 = mock_embedder.embed_texts([text1])[0]
        emb2 = mock_embedder.embed_texts([text2])[0]

        similarity = np.dot(emb1, emb2)
        assert similarity > 0.99  # Should be essentially identical

    def test_different_texts_have_different_embeddings(self, mock_embedder: Any) -> None:
        """Different texts should have different embeddings."""
        text1 = "Machine learning for NLP"
        text2 = "Cooking recipes for dinner"

        emb1 = mock_embedder.embed_texts([text1])[0]
        emb2 = mock_embedder.embed_texts([text2])[0]

        # Different texts should have different embeddings
        assert not np.allclose(emb1, emb2)
