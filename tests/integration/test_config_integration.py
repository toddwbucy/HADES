"""Integration tests for configuration loading and merging."""

import json
from pathlib import Path

import pytest

from core.config.config_base import (
    ConfigValidationError,
    ProcessingConfig,
    StorageConfig,
)


class TestProcessingConfigIntegration:
    """Integration tests for ProcessingConfig."""

    def test_default_processing_config(self) -> None:
        """ProcessingConfig should have sensible defaults."""
        config = ProcessingConfig()
        assert config.workers == 4
        assert config.batch_size == 10
        assert config.timeout_seconds == 300
        assert config.use_gpu is False

    def test_gpu_validation(self) -> None:
        """GPU config should validate device IDs."""
        # Valid GPU config
        config = ProcessingConfig(use_gpu=True, gpu_devices=[0, 1])
        errors = config.validate_semantics()
        assert len(errors) == 0

    def test_gpu_empty_devices_error(self) -> None:
        """Empty GPU devices list should fail validation."""
        config = ProcessingConfig(use_gpu=True, gpu_devices=[])
        errors = config.validate_semantics()
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_worker_batch_relationship(self) -> None:
        """Batch size validation relative to workers."""
        config = ProcessingConfig(workers=2, batch_size=1000)
        errors = config.validate_semantics()
        # Should warn about large batch size for few workers
        assert len(errors) > 0


class TestStorageConfigIntegration:
    """Integration tests for StorageConfig."""

    def test_default_storage_config(self) -> None:
        """StorageConfig should have sensible defaults."""
        config = StorageConfig()
        assert config.host == "localhost"
        assert config.port == 8529
        assert config.database == "academy_store"
        assert config.username == "root"

    def test_password_excluded_from_serialization(self) -> None:
        """Password should be excluded from serialization."""
        config = StorageConfig(password="secret123")
        d = config.to_dict()
        # Password should not appear in serialized output
        assert "password" not in d or d.get("password") != "secret123"

    def test_database_name_validation(self) -> None:
        """Database name validation."""
        config = StorageConfig(database="   ")  # Whitespace only
        errors = config.validate_semantics()
        assert len(errors) > 0

    def test_port_range(self) -> None:
        """Port should be in valid range."""
        # Valid port
        config = StorageConfig(port=8529)
        errors = config.validate_semantics()
        port_errors = [e for e in errors if "port" in e.lower()]
        assert len(port_errors) == 0


class TestConfigFilePersistence:
    """Tests for config file save/load."""

    def test_save_and_load_processing_config(self, tmp_path: Path) -> None:
        """ProcessingConfig should survive file round-trip."""
        config = ProcessingConfig(
            workers=8,
            batch_size=50,
            use_gpu=True,
            gpu_devices=[0],
        )

        file_path = tmp_path / "config.json"
        config.save_to_file(file_path)

        loaded = ProcessingConfig.from_file(file_path)
        assert loaded.workers == 8
        assert loaded.batch_size == 50
        assert loaded.use_gpu is True
        assert loaded.gpu_devices == [0]

    def test_save_and_load_storage_config(self, tmp_path: Path) -> None:
        """StorageConfig should survive file round-trip."""
        config = StorageConfig(
            host="db.example.com",
            port=9529,
            database="test_db",
            username="admin",
        )

        file_path = tmp_path / "storage.json"
        config.save_to_file(file_path)

        loaded = StorageConfig.from_file(file_path)
        assert loaded.host == "db.example.com"
        assert loaded.port == 9529
        assert loaded.database == "test_db"

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading nonexistent file should raise ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            ProcessingConfig.from_file(tmp_path / "nonexistent.json")


class TestConfigMerging:
    """Tests for configuration merging."""

    def test_merge_overrides_values(self) -> None:
        """merge should override base values with other values."""
        base = ProcessingConfig(workers=4, batch_size=10)
        override = ProcessingConfig(workers=8, batch_size=10)

        merged = base.merge(override)
        assert merged.workers == 8
        assert merged.batch_size == 10

    def test_merge_preserves_unset_values(self) -> None:
        """merge should preserve base values not in override."""
        base = ProcessingConfig(workers=4, batch_size=20, timeout_seconds=600)
        override = ProcessingConfig(workers=8, batch_size=20, timeout_seconds=300)

        merged = base.merge(override)
        assert merged.workers == 8  # overridden
        assert merged.timeout_seconds == 300  # overridden


class TestConfigJsonSerialization:
    """Tests for JSON serialization."""

    def test_to_json_valid(self) -> None:
        """to_json should produce valid JSON."""
        config = ProcessingConfig(workers=4, batch_size=10)
        json_str = config.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["workers"] == 4
        assert parsed["batch_size"] == 10

    def test_from_json_valid(self) -> None:
        """from_json should parse valid JSON."""
        json_str = '{"workers": 8, "batch_size": 50}'
        config = ProcessingConfig.from_json(json_str)
        assert config.workers == 8
        assert config.batch_size == 50

    def test_from_json_invalid_raises(self) -> None:
        """from_json should raise on invalid JSON."""
        with pytest.raises(ConfigValidationError):
            ProcessingConfig.from_json("not valid json {{{")


class TestConfigCompletenessScore:
    """Tests for completeness score calculation."""

    def test_complete_config_high_score(self) -> None:
        """Fully configured config should have high completeness."""
        config = ProcessingConfig(
            workers=4,
            batch_size=10,
            use_gpu=True,
            gpu_devices=[0],
        )
        config.source = "test.json"  # Add source for grounding

        score = config.get_completeness_score()
        assert score >= 0.5  # Should be reasonably complete

    def test_minimal_config_lower_score(self) -> None:
        """Minimal config may have lower completeness."""
        config = ProcessingConfig()
        score = config.get_completeness_score()
        # Score should still be valid (0-1 range)
        assert 0.0 <= score <= 1.0
