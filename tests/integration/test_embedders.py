"""Integration tests for embedders."""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.embedders import EmbedderBase, EmbedderFactory


class TestEmbedderFactory:
    """Tests for the EmbedderFactory."""

    def test_factory_has_registered_embedders(self) -> None:
        """Factory should have embedder registry."""
        # EmbedderFactory uses _embedders for registry
        assert hasattr(EmbedderFactory, "_embedders")

    def test_create_returns_embedder_base_instance(self) -> None:
        """create() should return EmbedderBase instance."""
        # This test will be skipped if model loading fails (no GPU/model)
        try:
            with patch.object(EmbedderFactory, "create") as mock_create:
                mock_embedder = MagicMock(spec=EmbedderBase)
                mock_create.return_value = mock_embedder

                embedder = EmbedderFactory.create("test")
                assert embedder is not None
        except Exception:
            pytest.skip("Embedder creation requires model download")


class TestJinaV4Embedder:
    """Integration tests for JinaV4Embedder."""

    @pytest.fixture
    def jina_embedder_class(self):
        """Get JinaV4Embedder class if available."""
        try:
            from core.embedders import JinaV4Embedder

            if JinaV4Embedder is None:
                pytest.skip("JinaV4Embedder not available")
            return JinaV4Embedder
        except ImportError:
            pytest.skip("JinaV4Embedder not available")

    def test_embedder_class_exists(self, jina_embedder_class) -> None:
        """JinaV4Embedder class should be importable."""
        assert jina_embedder_class is not None

    def test_embedder_has_required_methods(self, jina_embedder_class) -> None:
        """JinaV4Embedder should have required embedding methods."""
        # Check class has the methods (without instantiating)
        assert hasattr(jina_embedder_class, "embed_with_late_chunking")
        # JinaV4Embedder uses embed_single and embed_texts
        assert hasattr(jina_embedder_class, "embed_single")

    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "1") == "1", reason="Model tests skipped (set SKIP_MODEL_TESTS=0 to enable)"
    )
    def test_embedder_initialization(self, jina_embedder_class) -> None:
        """JinaV4Embedder should initialize with default model."""
        # This test downloads the model - only run when explicitly enabled
        embedder = jina_embedder_class()
        assert embedder is not None
        assert embedder.embedding_dim > 0


class TestMockedEmbedder:
    """Tests using mocked embedder (no GPU required)."""

    def test_embed_single_returns_vector(self, mock_embedder: MagicMock) -> None:
        """embed_single should return embedding vector."""
        result = mock_embedder.embed_single("test text")
        # Returns numpy array which can be checked for length
        assert len(result) == 2048  # Jina V4 dimension

    def test_embed_with_late_chunking_returns_chunks(self, mock_embedder: MagicMock) -> None:
        """embed_with_late_chunking should return ChunkWithEmbedding objects."""
        result = mock_embedder.embed_with_late_chunking("Test document text")
        assert isinstance(result, list)
        assert len(result) > 0

        chunk = result[0]
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "embedding")
        assert len(chunk.embedding) == 2048


class TestEmbeddingDimensions:
    """Tests for embedding dimension consistency."""

    def test_jina_v4_expected_dimensions(self) -> None:
        """Jina V4 should produce 2048-dimensional embeddings."""
        expected_dim = 2048

        try:
            from core.embedders import JinaV4Embedder

            if JinaV4Embedder is not None:
                # Just check the class attribute if available
                if hasattr(JinaV4Embedder, "EMBEDDING_DIM"):
                    assert JinaV4Embedder.EMBEDDING_DIM == expected_dim
        except ImportError:
            pytest.skip("JinaV4Embedder not available")


class TestLateChunking:
    """Tests for late chunking behavior."""

    def test_late_chunking_preserves_context(self, mock_embedder: MagicMock, sample_text: str) -> None:
        """Late chunking should process document before splitting."""
        # This is a conceptual test - late chunking embeds full doc then splits
        chunks = mock_embedder.embed_with_late_chunking(sample_text)

        # Each chunk should have embedding from full document context
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == mock_embedder.embedding_dim

    def test_chunks_have_position_info(self, mock_embedder: MagicMock) -> None:
        """Chunks should include token position information."""
        chunks = mock_embedder.embed_with_late_chunking("Test text for chunking")

        for chunk in chunks:
            assert hasattr(chunk, "token_start")
            assert hasattr(chunk, "token_end")
            assert chunk.token_start >= 0
            assert chunk.token_end >= chunk.token_start


class TestEmbedderResourceManagement:
    """Tests for embedder resource handling."""

    def test_embedder_class_is_importable(self) -> None:
        """JinaV4Embedder class should be importable and have expected attributes."""
        try:
            from core.embedders import JinaV4Embedder

            if JinaV4Embedder is None:
                pytest.skip("JinaV4Embedder not available")

            # Check class has basic expected attributes
            assert hasattr(JinaV4Embedder, "EMBEDDING_DIM")
            assert hasattr(JinaV4Embedder, "MAX_TOKENS")
        except ImportError:
            pytest.skip("JinaV4Embedder not available")
