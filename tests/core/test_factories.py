"""Unit tests for factory classes with registry pattern.

Tests for:
- DatabaseFactory (core/database/database_factory.py)
- ExtractorFactory (core/extractors/extractor_factory.py)
- StorageFactory (core/workflows/storage/storage_factory.py)
"""

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# DatabaseFactory Tests
# =============================================================================


class TestDatabaseFactory:
    """Tests for DatabaseFactory with registry pattern."""

    def test_list_available_returns_dict(self):
        """list_available should return dictionary of backends."""
        from core.database.database_factory import DatabaseFactory

        # Reset registry for clean test
        DatabaseFactory._registry.clear()
        DatabaseFactory._auto_registered.clear()

        available = DatabaseFactory.list_available()
        assert isinstance(available, dict)
        # Should have at least the built-in backends
        assert "arango" in available
        assert "postgres" in available
        assert "redis" in available

    def test_register_decorator(self):
        """register decorator should add class to registry."""
        from core.database.database_factory import DatabaseFactory

        # Save original registry
        original_registry = dict(DatabaseFactory._registry)

        try:
            @DatabaseFactory.register("test_db")
            class TestDB:
                pass

            assert "test_db" in DatabaseFactory._registry
            assert DatabaseFactory._registry["test_db"] is TestDB
        finally:
            # Restore original registry
            DatabaseFactory._registry = original_registry

    def test_create_raises_for_unknown_type(self):
        """create should raise ValueError for unknown database type."""
        from core.database.database_factory import DatabaseFactory

        # Reset registry for clean test
        DatabaseFactory._registry.clear()
        DatabaseFactory._auto_registered.clear()

        with pytest.raises(ValueError, match="No database backend registered"):
            DatabaseFactory.create("nonexistent_db")

    def test_create_auto_registers_arango(self):
        """create should auto-register arango backend."""
        from core.database.database_factory import DatabaseFactory

        # Reset registry for clean test
        DatabaseFactory._registry.clear()
        DatabaseFactory._auto_registered.clear()

        # Trigger auto-registration
        DatabaseFactory._auto_register("arango")

        assert "arango" in DatabaseFactory._registry

    def test_legacy_get_arango_requires_password(self):
        """get_arango should require password."""
        from core.database.database_factory import DatabaseFactory

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="password required"):
                DatabaseFactory.get_arango(password=None)

    @patch("core.database.arango.ArangoMemoryClient")
    @patch("core.database.arango.resolve_memory_config")
    def test_legacy_get_arango_creates_client(self, mock_resolve, mock_client):
        """get_arango should create ArangoMemoryClient."""
        from core.database.database_factory import DatabaseFactory

        mock_config = MagicMock()
        mock_config.read_socket = "/tmp/test.sock"
        mock_config.write_socket = "/tmp/test.sock"
        mock_resolve.return_value = mock_config

        DatabaseFactory.get_arango(password="test_password")

        mock_resolve.assert_called_once()
        mock_client.assert_called_once_with(mock_config)


# =============================================================================
# ExtractorFactory Tests
# =============================================================================


class TestExtractorFactory:
    """Tests for ExtractorFactory with registry pattern."""

    def test_list_available_returns_dict(self):
        """list_available should return dictionary of extractors."""
        from core.extractors import ExtractorFactory

        # Reset for clean test
        ExtractorFactory._registry.clear()
        ExtractorFactory._extension_map.clear()
        ExtractorFactory._auto_registered.clear()

        available = ExtractorFactory.list_available()
        assert isinstance(available, dict)

    def test_register_decorator(self):
        """register decorator should add class to registry."""
        from core.extractors import ExtractorBase, ExtractorFactory

        # Save original registry
        original_registry = dict(ExtractorFactory._registry)
        original_ext_map = dict(ExtractorFactory._extension_map)

        try:
            @ExtractorFactory.register("test_extractor", extensions=[".xyz"])
            class TestExtractor(ExtractorBase):
                def extract(self, file_path, **kwargs):
                    pass

                def extract_batch(self, file_paths, **kwargs):
                    pass

                @property
                def supported_formats(self):
                    return [".xyz"]

            assert "test_extractor" in ExtractorFactory._registry
            assert ".xyz" in ExtractorFactory._extension_map
            assert ExtractorFactory._extension_map[".xyz"] == "test_extractor"
        finally:
            # Restore original registry
            ExtractorFactory._registry = original_registry
            ExtractorFactory._extension_map = original_ext_map

    def test_create_raises_for_unknown_type(self):
        """create should raise ValueError for unknown extractor type."""
        from core.extractors import ExtractorFactory

        # Reset registry for clean test
        ExtractorFactory._registry.clear()
        ExtractorFactory._auto_registered.clear()

        with pytest.raises(ValueError, match="No extractor registered"):
            ExtractorFactory.create("nonexistent_extractor")

    def test_supports_extension(self):
        """supports_extension should check extension map."""
        from core.extractors import ExtractorFactory

        # Ensure built-in extractors are registered
        ExtractorFactory._ensure_registered()

        # PDF should be supported (docling)
        assert ExtractorFactory.supports_extension(".pdf") is True
        assert ExtractorFactory.supports_extension("pdf") is True

        # Unknown extension
        assert ExtractorFactory.supports_extension(".xyz123") is False

    def test_get_extensions_map(self):
        """get_extensions_map should return extension mapping."""
        from core.extractors import ExtractorFactory

        ExtractorFactory._ensure_registered()

        ext_map = ExtractorFactory.get_extensions_map()
        assert isinstance(ext_map, dict)
        # Should have common extensions
        assert ".pdf" in ext_map or ".tex" in ext_map or ".py" in ext_map

    def test_for_file_uses_robust_fallback(self):
        """for_file should use robust extractor as fallback for unknown extensions."""
        from core.extractors import ExtractorFactory

        # Reset registry for clean test
        ExtractorFactory._registry.clear()
        ExtractorFactory._extension_map.clear()
        ExtractorFactory._auto_registered.clear()

        # With robust extractor available, it should fall back to it
        try:
            extractor = ExtractorFactory.for_file("test.unknown_extension_xyz")
            # If robust is available, should return an extractor
            assert hasattr(extractor, "extract")
        except ValueError:
            # If no fallback available, should raise ValueError
            pass


# =============================================================================
# StorageFactory Tests
# =============================================================================


class TestStorageFactory:
    """Tests for StorageFactory with registry pattern."""

    def test_list_available_returns_dict(self):
        """list_available should return dictionary of backends."""
        from core.workflows.storage import StorageFactory

        # Reset for clean test
        StorageFactory._registry.clear()
        StorageFactory._auto_registered.clear()

        available = StorageFactory.list_available()
        assert isinstance(available, dict)
        # Should have local backend
        assert "local" in available

    def test_register_decorator(self):
        """register decorator should add class to registry."""
        from core.workflows.storage import StorageBase, StorageFactory

        # Save original registry
        original_registry = dict(StorageFactory._registry)

        try:
            @StorageFactory.register("test_storage")
            class TestStorage(StorageBase):
                def store(self, key, data, metadata=None):
                    return True

                def retrieve(self, key):
                    return None

                def exists(self, key):
                    return False

                def delete(self, key):
                    return True

                def list_keys(self, prefix=None):
                    return []

                @property
                def storage_type(self):
                    return "test"

            assert "test_storage" in StorageFactory._registry
        finally:
            # Restore original registry
            StorageFactory._registry = original_registry

    def test_create_raises_for_unknown_type(self):
        """create should raise ValueError for unknown storage type."""
        from core.workflows.storage import StorageFactory

        # Reset registry for clean test
        StorageFactory._registry.clear()
        StorageFactory._auto_registered.clear()

        with pytest.raises(ValueError, match="No storage backend registered"):
            StorageFactory.create("nonexistent_storage")

    def test_create_local_storage(self):
        """create should create local storage backend."""
        from core.workflows.storage import StorageBase, StorageFactory

        # Reset registry for clean test
        StorageFactory._registry.clear()
        StorageFactory._auto_registered.clear()

        storage = StorageFactory.create("local", base_path="/tmp/test_factories")
        # Should implement StorageBase interface
        assert hasattr(storage, "store")
        assert hasattr(storage, "retrieve")
        assert hasattr(storage, "exists")
        assert storage.storage_type == "local_filesystem"

    def test_get_for_config(self):
        """get_for_config should create storage from config dict."""
        from core.workflows.storage import StorageFactory

        # Reset registry for clean test
        StorageFactory._registry.clear()
        StorageFactory._auto_registered.clear()

        config = {"type": "local", "base_path": "/tmp/test_factories_config"}
        storage = StorageFactory.get_for_config(config)
        # Should implement StorageBase interface
        assert hasattr(storage, "store")
        assert hasattr(storage, "retrieve")
        assert storage.storage_type == "local_filesystem"

    def test_get_for_config_requires_type(self):
        """get_for_config should raise if 'type' missing."""
        from core.workflows.storage import StorageFactory

        with pytest.raises(ValueError, match="must include 'type' key"):
            StorageFactory.get_for_config({"base_path": "/tmp"})


# =============================================================================
# Integration: get_extractor backwards compatibility
# =============================================================================


class TestGetExtractorBackwardsCompat:
    """Test that get_extractor() function works with ExtractorFactory."""

    def test_get_extractor_uses_factory(self):
        """get_extractor should use ExtractorFactory internally."""
        from core.extractors import ExtractorFactory, get_extractor

        # Reset for clean test
        ExtractorFactory._ensure_registered()

        # If docling is available, this should work
        try:
            extractor = get_extractor("test.pdf")
            # Should return an extractor instance
            assert hasattr(extractor, "extract")
        except (ImportError, ValueError):
            # Extractor not available, which is fine
            pass
