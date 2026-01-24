"""Unit tests for core.config.config_base module."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import Field

from core.config.config_base import (
    BaseConfig,
    ConfigError,
    ConfigValidationError,
)


class ConcreteConfig(BaseConfig):
    """Concrete implementation for testing BaseConfig."""

    name: str = Field(default="test", description="Config name")
    value: int = Field(default=0, ge=0, description="Non-negative value")
    items: list[str] = Field(default_factory=list, description="List of items")

    def validate_semantics(self) -> list[str]:
        """Validate semantic rules."""
        errors = []
        if self.name == "invalid":
            errors.append("Name cannot be 'invalid'")
        if self.value > 100 and len(self.items) == 0:
            errors.append("High value requires at least one item")
        return errors


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_exception(self) -> None:
        """ConfigError should be an Exception."""
        assert issubclass(ConfigError, Exception)

    def test_config_error_message(self) -> None:
        """ConfigError should store message."""
        error = ConfigError("test message")
        assert str(error) == "test message"


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_validation_error_is_config_error(self) -> None:
        """ConfigValidationError should inherit from ConfigError."""
        assert issubclass(ConfigValidationError, ConfigError)

    def test_validation_error_stores_errors(self) -> None:
        """ConfigValidationError should store error list."""
        errors = ["error1", "error2"]
        error = ConfigValidationError("Validation failed", errors)
        assert error.errors == errors

    def test_validation_error_message_includes_errors(self) -> None:
        """ConfigValidationError message should include error details."""
        errors = ["error1", "error2"]
        error = ConfigValidationError("Validation failed", errors)
        assert "error1" in str(error)
        assert "error2" in str(error)


class TestBaseConfig:
    """Tests for BaseConfig abstract base class."""

    def test_default_values(self) -> None:
        """Config should have default values."""
        config = ConcreteConfig()
        assert config.config_version == "1.0"
        assert config.name == "test"
        assert config.value == 0
        assert config.items == []

    def test_created_at_auto_set(self) -> None:
        """created_at should be auto-set if not provided."""
        config = ConcreteConfig()
        assert config.created_at is not None
        assert isinstance(config.created_at, datetime)

    def test_created_at_can_be_provided(self) -> None:
        """created_at can be explicitly provided."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        config = ConcreteConfig(created_at=timestamp)
        assert config.created_at == timestamp

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = ConcreteConfig(name="custom", value=42, items=["a", "b"])
        assert config.name == "custom"
        assert config.value == 42
        assert config.items == ["a", "b"]

    def test_pydantic_validation_negative_value(self) -> None:
        """Pydantic should reject negative values when ge=0."""
        with pytest.raises(ValueError):
            ConcreteConfig(value=-1)

    def test_pydantic_validation_extra_fields_forbidden(self) -> None:
        """Pydantic should reject extra fields (extra='forbid')."""
        with pytest.raises(ValueError):
            ConcreteConfig(unknown_field="value")


class TestBaseConfigSemanticValidation:
    """Tests for semantic validation."""

    def test_validate_semantics_valid(self) -> None:
        """Valid config should pass semantic validation."""
        config = ConcreteConfig(name="valid", value=50)
        errors = config.validate_semantics()
        assert errors == []

    def test_validate_semantics_invalid_name(self) -> None:
        """Invalid name should fail semantic validation."""
        config = ConcreteConfig(name="invalid")
        errors = config.validate_semantics()
        assert len(errors) == 1
        assert "invalid" in errors[0].lower()

    def test_validate_semantics_high_value_no_items(self) -> None:
        """High value without items should fail semantic validation."""
        config = ConcreteConfig(value=150, items=[])
        errors = config.validate_semantics()
        assert len(errors) == 1
        assert "item" in errors[0].lower()

    def test_validate_full_valid(self) -> None:
        """validate_full should pass for valid config."""
        config = ConcreteConfig(name="valid", value=50, items=["x"])
        config.validate_full()  # Should not raise

    def test_validate_full_invalid(self) -> None:
        """validate_full should raise for invalid config."""
        config = ConcreteConfig(name="invalid")
        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate_full()
        assert "invalid" in str(exc_info.value).lower()


class TestBaseConfigSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self) -> None:
        """to_dict should return dictionary representation."""
        config = ConcreteConfig(name="test", value=10, items=["a"])
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test"
        assert d["value"] == 10
        assert d["items"] == ["a"]

    def test_to_dict_excludes_none(self) -> None:
        """to_dict should exclude None values."""
        config = ConcreteConfig(source=None)
        d = config.to_dict()
        assert "source" not in d

    def test_to_json(self) -> None:
        """to_json should return valid JSON string."""
        config = ConcreteConfig(name="test", value=10)
        json_str = config.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["value"] == 10

    def test_to_json_with_indent(self) -> None:
        """to_json should support custom indent."""
        config = ConcreteConfig()
        json_str = config.to_json(indent=4)
        assert "    " in json_str  # 4-space indent


class TestBaseConfigDeserialization:
    """Tests for deserialization methods."""

    def test_from_dict_valid(self) -> None:
        """from_dict should create config from valid dictionary."""
        data = {"name": "from_dict", "value": 25, "items": ["x", "y"]}
        config = ConcreteConfig.from_dict(data)
        assert config.name == "from_dict"
        assert config.value == 25
        assert config.items == ["x", "y"]

    def test_from_dict_invalid_schema(self) -> None:
        """from_dict should raise for invalid schema."""
        data = {"value": -1}  # Negative value not allowed
        with pytest.raises(ConfigValidationError):
            ConcreteConfig.from_dict(data)

    def test_from_dict_invalid_semantics(self) -> None:
        """from_dict should raise for invalid semantics."""
        data = {"name": "invalid"}
        with pytest.raises(ConfigValidationError):
            ConcreteConfig.from_dict(data)

    def test_from_json_valid(self) -> None:
        """from_json should create config from valid JSON."""
        json_str = '{"name": "from_json", "value": 30}'
        config = ConcreteConfig.from_json(json_str)
        assert config.name == "from_json"
        assert config.value == 30

    def test_from_json_invalid_json(self) -> None:
        """from_json should raise for invalid JSON."""
        with pytest.raises(ConfigValidationError):
            ConcreteConfig.from_json("not valid json")

    def test_from_json_invalid_schema(self) -> None:
        """from_json should raise for invalid schema."""
        json_str = '{"value": "not_an_int"}'
        with pytest.raises(ConfigValidationError):
            ConcreteConfig.from_json(json_str)


class TestBaseConfigRoundTrip:
    """Tests for serialization round-trip."""

    def test_dict_round_trip(self) -> None:
        """Config should survive dict round-trip."""
        original = ConcreteConfig(name="roundtrip", value=42, items=["a", "b"])
        d = original.to_dict()
        restored = ConcreteConfig.from_dict(d)
        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.items == original.items

    def test_json_round_trip(self) -> None:
        """Config should survive JSON round-trip."""
        original = ConcreteConfig(name="json_trip", value=99, items=["x"])
        json_str = original.to_json()
        restored = ConcreteConfig.from_json(json_str)
        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.items == original.items
