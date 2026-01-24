"""Integration tests for storage backends."""

from pathlib import Path

import pytest

from core.workflows.storage.storage_base import StorageBase
from core.workflows.storage.storage_local import LocalStorage


class TestStorageBaseInterface:
    """Tests for StorageBase abstract interface."""

    def test_storage_base_is_abstract(self) -> None:
        """StorageBase should be abstract and not instantiable directly."""
        with pytest.raises(TypeError):
            StorageBase()  # type: ignore


class TestLocalStorage:
    """Integration tests for LocalStorage backend."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        """Create LocalStorage instance with temp directory."""
        return LocalStorage(tmp_path / "storage")

    def test_creates_base_directory(self, tmp_path: Path) -> None:
        """LocalStorage should create base directory on init."""
        storage_path = tmp_path / "new_storage"
        LocalStorage(storage_path)
        assert storage_path.exists()

    def test_storage_type_property(self, storage: LocalStorage) -> None:
        """storage_type property should return correct type."""
        assert storage.storage_type == "local_filesystem"

    def test_supports_metadata_property(self, storage: LocalStorage) -> None:
        """supports_metadata should be True for LocalStorage."""
        assert storage.supports_metadata is True

    def test_store_creates_json_file(self, storage: LocalStorage) -> None:
        """store should create a JSON file."""
        storage.store("test_key", {"value": 42})

        # Check file exists (key is percent-encoded)
        files = list(storage.base_path.glob("*.json"))
        assert len(files) == 1

    def test_store_and_retrieve_simple(self, storage: LocalStorage) -> None:
        """store and retrieve should work for simple data."""
        data = {"name": "test", "count": 123}
        assert storage.store("simple", data) is True
        result = storage.retrieve("simple")
        assert result == data

    def test_store_and_retrieve_complex(self, storage: LocalStorage) -> None:
        """store and retrieve should work for complex nested data."""
        data = {
            "items": [1, 2, 3],
            "nested": {"a": {"b": {"c": "deep"}}},
            "mixed": [{"key": "value"}, [1, 2]],
        }
        assert storage.store("complex", data) is True
        result = storage.retrieve("complex")
        assert result == data

    def test_store_with_metadata(self, storage: LocalStorage) -> None:
        """store should include metadata in file."""
        data = {"content": "test"}
        metadata = {"author": "tester", "version": 1}
        storage.store("with_meta", data, metadata)

        result = storage.retrieve("with_meta")
        assert result == data

        # Metadata should be retrievable
        meta = storage.get_metadata("with_meta")
        assert meta["author"] == "tester"

    def test_retrieve_nonexistent(self, storage: LocalStorage) -> None:
        """retrieve should return None for nonexistent key."""
        result = storage.retrieve("does_not_exist")
        assert result is None

    def test_exists(self, storage: LocalStorage) -> None:
        """exists should correctly detect stored keys."""
        assert storage.exists("missing") is False
        storage.store("present", "data")
        assert storage.exists("present") is True

    def test_delete(self, storage: LocalStorage) -> None:
        """delete should remove stored data."""
        storage.store("to_delete", "data")
        assert storage.exists("to_delete") is True
        assert storage.delete("to_delete") is True
        assert storage.exists("to_delete") is False

    def test_delete_nonexistent(self, storage: LocalStorage) -> None:
        """delete should return False for nonexistent key."""
        assert storage.delete("never_existed") is False

    def test_special_characters_in_key(self, storage: LocalStorage) -> None:
        """Keys with special characters should be handled correctly."""
        # Test with various special characters
        special_keys = [
            "key/with/slashes",
            "key:with:colons",
            "key with spaces",
            "key?with=query",
        ]
        for key in special_keys:
            data = {"key": key}
            assert storage.store(key, data) is True
            assert storage.retrieve(key) == data

    def test_list_keys(self, storage: LocalStorage) -> None:
        """list_keys should return all matching keys."""
        storage.store("prefix_a", "data_a")
        storage.store("prefix_b", "data_b")
        storage.store("other", "data_c")

        keys = storage.list_keys("prefix")
        assert len(keys) == 2
        assert "prefix_a" in keys
        assert "prefix_b" in keys

    def test_list_keys_no_prefix(self, storage: LocalStorage) -> None:
        """list_keys with no prefix should return all keys."""
        storage.store("a", "data")
        storage.store("b", "data")
        storage.store("c", "data")

        keys = storage.list_keys()
        assert len(keys) == 3

    def test_batch_store(self, storage: LocalStorage) -> None:
        """batch_store should store multiple items."""
        items = {
            "batch_1": {"v": 1},
            "batch_2": {"v": 2},
            "batch_3": {"v": 3},
        }
        results = storage.batch_store(items)

        assert all(results.values())
        assert storage.retrieve("batch_1") == {"v": 1}
        assert storage.retrieve("batch_2") == {"v": 2}
        assert storage.retrieve("batch_3") == {"v": 3}

    def test_batch_retrieve(self, storage: LocalStorage) -> None:
        """batch_retrieve should retrieve multiple items."""
        storage.store("a", {"val": 1})
        storage.store("b", {"val": 2})
        storage.store("c", {"val": 3})

        results = storage.batch_retrieve(["a", "b", "c", "missing"])
        assert len(results) == 3
        assert results["a"] == {"val": 1}
        assert "missing" not in results

    def test_clear_all(self, storage: LocalStorage) -> None:
        """clear should delete all items."""
        storage.store("a", "data")
        storage.store("b", "data")
        storage.store("c", "data")

        count = storage.clear()
        assert count == 3
        assert storage.list_keys() == []

    def test_clear_with_prefix(self, storage: LocalStorage) -> None:
        """clear with prefix should only delete matching items."""
        storage.store("prefix_a", "data")
        storage.store("prefix_b", "data")
        storage.store("other", "data")

        count = storage.clear("prefix")
        assert count == 2
        assert storage.exists("other")

    def test_atomic_write(self, storage: LocalStorage) -> None:
        """store should use atomic write pattern."""
        # Store data and verify no .tmp files remain
        storage.store("atomic_test", {"data": "important"})

        tmp_files = list(storage.base_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_get_storage_info(self, storage: LocalStorage) -> None:
        """get_storage_info should return storage info dict."""
        info = storage.get_storage_info()
        assert info["type"] == "local_filesystem"
        assert info["supports_metadata"] is True
        assert info["supports_streaming"] is False


class TestLocalStorageEdgeCases:
    """Edge case tests for LocalStorage."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        """Create LocalStorage instance."""
        return LocalStorage(tmp_path / "storage")

    def test_overwrite_existing(self, storage: LocalStorage) -> None:
        """Storing with same key should overwrite."""
        storage.store("key", {"version": 1})
        storage.store("key", {"version": 2})

        result = storage.retrieve("key")
        assert result == {"version": 2}

    def test_empty_string_data(self, storage: LocalStorage) -> None:
        """Should handle empty string data."""
        storage.store("empty", "")
        assert storage.retrieve("empty") == ""

    def test_null_data(self, storage: LocalStorage) -> None:
        """Should handle None/null data."""
        storage.store("null", None)
        assert storage.retrieve("null") is None

    def test_list_data(self, storage: LocalStorage) -> None:
        """Should handle list data directly."""
        data = [1, 2, 3, {"nested": True}]
        storage.store("list", data)
        assert storage.retrieve("list") == data

    def test_large_data(self, storage: LocalStorage) -> None:
        """Should handle reasonably large data."""
        data = {"items": list(range(10000))}
        storage.store("large", data)
        result = storage.retrieve("large")
        assert len(result["items"]) == 10000

    def test_unicode_data(self, storage: LocalStorage) -> None:
        """Should handle unicode data correctly."""
        data = {"text": "Hello 世界 🌍"}
        storage.store("unicode", data)
        assert storage.retrieve("unicode") == data
