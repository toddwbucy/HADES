"""Unit tests for core.embedders.embedders_factory module."""

from unittest.mock import MagicMock, patch

import pytest

from core.embedders.embedders_base import EmbeddingConfig
from core.embedders.embedders_factory import EmbedderFactory


class TestEmbedderFactoryRegistry:
    """Tests for embedder registration."""

    def test_embedders_registry_exists(self) -> None:
        """Factory should have _embedders registry."""
        assert hasattr(EmbedderFactory, "_embedders")
        assert isinstance(EmbedderFactory._embedders, dict)

    def test_register_embedder(self) -> None:
        """register should add embedder to registry."""
        # Save original registry
        original = EmbedderFactory._embedders.copy()

        try:
            mock_class = MagicMock()
            EmbedderFactory.register("test_embedder", mock_class)
            assert "test_embedder" in EmbedderFactory._embedders
            assert EmbedderFactory._embedders["test_embedder"] is mock_class
        finally:
            # Restore original registry
            EmbedderFactory._embedders = original

    def test_list_available_returns_dict(self) -> None:
        """list_available should return dictionary of embedders."""
        result = EmbedderFactory.list_available()
        assert isinstance(result, dict)


class TestDetermineEmbedderType:
    """Tests for _determine_embedder_type."""

    def test_jina_model_returns_jina(self) -> None:
        """Jina models should return 'jina' type."""
        assert EmbedderFactory._determine_embedder_type("jinaai/jina-embeddings-v4") == "jina"
        assert EmbedderFactory._determine_embedder_type("jina-embeddings-v2-base-en") == "jina"
        assert EmbedderFactory._determine_embedder_type("JINA-embeddings") == "jina"

    def test_sentence_transformers_returns_sentence(self) -> None:
        """Sentence transformer models should return 'sentence' type."""
        assert EmbedderFactory._determine_embedder_type("sentence-transformers/all-MiniLM-L6-v2") == "sentence"
        assert EmbedderFactory._determine_embedder_type("st-multi-qa-MiniLM-L6-cos-v1") == "sentence"

    def test_openai_returns_openai(self) -> None:
        """OpenAI models should return 'openai' type."""
        assert EmbedderFactory._determine_embedder_type("openai/text-embedding-3-small") == "openai"
        assert EmbedderFactory._determine_embedder_type("text-embedding-ada-002") == "openai"

    def test_cohere_returns_cohere(self) -> None:
        """Cohere models should return 'cohere' type."""
        assert EmbedderFactory._determine_embedder_type("cohere/embed-english-v3.0") == "cohere"

    def test_transformers_returns_jina(self) -> None:
        """Transformers models should return 'jina' type."""
        assert EmbedderFactory._determine_embedder_type("transformers/bert-base-uncased") == "jina"

    def test_unknown_defaults_to_jina(self) -> None:
        """Unknown models should default to 'jina'."""
        assert EmbedderFactory._determine_embedder_type("some-random-model") == "jina"
        assert EmbedderFactory._determine_embedder_type("my-custom-embedder") == "jina"


class TestEmbedderFactoryCreate:
    """Tests for factory create method."""

    def test_create_with_default_model(self) -> None:
        """create() should use jinaai/jina-embeddings-v4 as default."""
        # Save original registry
        original = EmbedderFactory._embedders.copy()

        try:
            # Register mock embedder
            mock_embedder_instance = MagicMock()
            mock_class = MagicMock(return_value=mock_embedder_instance)
            EmbedderFactory._embedders["jina"] = mock_class

            result = EmbedderFactory.create()

            assert result is mock_embedder_instance
            mock_class.assert_called_once()
            # Verify config was passed with default model name
            call_args = mock_class.call_args
            config = call_args[0][0]
            assert config.model_name == "jinaai/jina-embeddings-v4"
        finally:
            EmbedderFactory._embedders = original

    def test_create_with_custom_model_name(self) -> None:
        """create() should use provided model name."""
        original = EmbedderFactory._embedders.copy()

        try:
            mock_embedder_instance = MagicMock()
            mock_class = MagicMock(return_value=mock_embedder_instance)
            EmbedderFactory._embedders["jina"] = mock_class

            EmbedderFactory.create(model_name="my-custom-jina-model")

            call_args = mock_class.call_args
            config = call_args[0][0]
            assert config.model_name == "my-custom-jina-model"
        finally:
            EmbedderFactory._embedders = original

    def test_create_with_config(self) -> None:
        """create() should use provided config."""
        original = EmbedderFactory._embedders.copy()

        try:
            mock_embedder_instance = MagicMock()
            mock_class = MagicMock(return_value=mock_embedder_instance)
            EmbedderFactory._embedders["jina"] = mock_class

            config = EmbeddingConfig(
                model_name="jina-test",
                device="cpu",
                batch_size=16,
            )
            EmbedderFactory.create(config=config)

            call_args = mock_class.call_args
            passed_config = call_args[0][0]
            assert passed_config.model_name == "jina-test"
            assert passed_config.device == "cpu"
            assert passed_config.batch_size == 16
        finally:
            EmbedderFactory._embedders = original

    def test_create_with_kwargs_override(self) -> None:
        """create() should allow kwargs to override config."""
        original = EmbedderFactory._embedders.copy()

        try:
            mock_embedder_instance = MagicMock()
            mock_class = MagicMock(return_value=mock_embedder_instance)
            EmbedderFactory._embedders["jina"] = mock_class

            config = EmbeddingConfig(
                model_name="jina-test",
                device="cuda",
                batch_size=32,
            )
            EmbedderFactory.create(config=config, device="cpu", batch_size=8)

            call_args = mock_class.call_args
            passed_config = call_args[0][0]
            assert passed_config.device == "cpu"
            assert passed_config.batch_size == 8
        finally:
            EmbedderFactory._embedders = original

    def test_create_raises_for_unregistered_type(self) -> None:
        """create() should raise ValueError for unknown types."""
        original = EmbedderFactory._embedders.copy()

        try:
            # Clear registry and prevent auto-registration
            EmbedderFactory._embedders.clear()

            with patch.object(EmbedderFactory, "_auto_register"):
                with pytest.raises(ValueError, match="No embedder registered"):
                    EmbedderFactory.create(model_name="some-model")
        finally:
            EmbedderFactory._embedders = original


class TestAutoRegister:
    """Tests for _auto_register method."""

    def test_auto_register_unknown_type_logs_warning(self) -> None:
        """_auto_register should log warning for unknown types."""
        original = EmbedderFactory._embedders.copy()

        try:
            EmbedderFactory._embedders.clear()

            with patch(
                "core.embedders.embedders_factory.logger"
            ) as mock_logger:
                EmbedderFactory._auto_register("completely_unknown_type")
                mock_logger.warning.assert_called()
        finally:
            EmbedderFactory._embedders = original

    def test_auto_register_handles_import_error(self) -> None:
        """_auto_register should handle ImportError gracefully."""
        original = EmbedderFactory._embedders.copy()

        try:
            EmbedderFactory._embedders.clear()

            # Force import error
            with patch(
                "core.embedders.embedders_factory.logger"
            ) as mock_logger:
                # Unknown type should log warning
                EmbedderFactory._auto_register("nonexistent_type")
                mock_logger.warning.assert_called()
        finally:
            EmbedderFactory._embedders = original


class TestListAvailable:
    """Tests for list_available method."""

    def test_list_available_shows_registered_embedders(self) -> None:
        """list_available should show all registered embedders."""
        original = EmbedderFactory._embedders.copy()

        try:
            # Register test embedders
            mock_class_a = type("MockEmbedderA", (), {"__module__": "test.module"})
            mock_class_b = type("MockEmbedderB", (), {"__module__": "test.module"})

            EmbedderFactory._embedders["test_a"] = mock_class_a
            EmbedderFactory._embedders["test_b"] = mock_class_b

            available = EmbedderFactory.list_available()

            assert "test_a" in available
            assert "test_b" in available
            assert available["test_a"]["class"] == "MockEmbedderA"
            assert available["test_b"]["class"] == "MockEmbedderB"
        finally:
            EmbedderFactory._embedders = original
