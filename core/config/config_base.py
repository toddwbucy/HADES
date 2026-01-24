"""
Base Configuration Classes
==========================

Provides Pydantic-based configuration models with validation and serialization.
Supports hierarchical configuration sources (environment > file > defaults).
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation errors."""

    def __init__(self, message: str, errors: list[str]):
        self.errors = errors
        super().__init__(f"{message}: {'; '.join(errors)}")


class BaseConfig(BaseModel, ABC):
    """
    Abstract base for all configuration models.

    Provides Pydantic validation, serialization, and hierarchical merging.
    Subclasses should implement validate_semantics() for domain-specific validation.
    """

    # Metadata fields
    config_version: str = Field(default="1.0", description="Configuration schema version")
    created_at: datetime | None = Field(default=None, description="Configuration creation timestamp")
    source: str | None = Field(default=None, description="Configuration source identifier")

    model_config = ConfigDict(
        extra="forbid",  # Strict validation
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=False,
    )

    def __init__(self, **data):
        # Set creation timestamp if not provided
        if 'created_at' not in data:
            data['created_at'] = datetime.now(UTC)
        super().__init__(**data)

    @abstractmethod
    def validate_semantics(self) -> list[str]:
        """
        Validate semantic consistency beyond schema validation.

        Override in subclasses to add domain-specific validation rules.

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    def validate_full(self) -> None:
        """
        Perform full validation including semantic checks.

        Raises:
            ConfigValidationError: If validation fails
        """
        # Note: Pydantic schema validation occurs automatically at __init__.
        # This method focuses on semantic validation that goes beyond schema.

        # Semantic validation
        semantic_errors = self.validate_semantics()
        if semantic_errors:
            raise ConfigValidationError("Semantic validation failed", semantic_errors)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON format.

        Args:
            indent: JSON indentation

        Returns:
            Configuration as JSON string
        """
        return self.model_dump_json(exclude_none=True, indent=indent)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Args:
            data: Configuration data

        Returns:
            Configuration instance

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            instance = cls(**data)
            instance.validate_full()
            return instance
        except ValidationError as e:
            raise ConfigValidationError(
                "Failed to create from dict",
                [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            ) from e

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON configuration string

        Returns:
            Configuration instance

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigValidationError("Invalid JSON format", [str(e)]) from e

    @classmethod
    def from_file(cls: type[T], file_path: str | Path) -> T:
        """
        Load configuration from a JSON file.

        Note: This method only supports JSON format. For YAML files or
        hierarchical configuration loading, use ConfigManager.load_config()
        instead.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Configuration instance with source set to the file path

        Raises:
            ConfigValidationError: If file cannot be read or JSON is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigValidationError(f"Configuration file not found: {path}", [])

        try:
            with open(path, encoding='utf-8') as f:
                content = f.read()

            instance = cls.from_json(content)
            instance.source = str(path)
            return instance

        except OSError as e:
            raise ConfigValidationError(f"Failed to read configuration file: {path}", [str(e)]) from e

    def save_to_file(self, file_path: str | Path) -> None:
        """
        Save configuration to file.

        Args:
            file_path: Path to save configuration

        Raises:
            ConfigError: If file cannot be saved
        """
        path = Path(file_path)
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())

            self.source = str(path)
            logger.info(f"Configuration saved to {path}")

        except OSError as e:
            raise ConfigError(f"Failed to save configuration to {path}: {e}") from e

    def merge(self: T, other: T) -> T:
        """
        Merge with another configuration of the same type.

        Values from 'other' override values from 'self'. Nested dicts
        are deep-merged.

        Args:
            other: Configuration to merge (higher priority)

        Returns:
            New merged configuration instance
        """
        # Get base data
        base_data = self.to_dict()
        other_data = other.to_dict()

        # Deep merge logic
        merged_data = self._deep_merge_dicts(base_data, other_data)

        # Create new instance
        return self.__class__.from_dict(merged_data)

    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (higher priority)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
                result[key] = BaseConfig._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def get_completeness_score(self) -> float:
        """
        Calculate configuration completeness score.

        Evaluates configuration quality based on:
        - Local coherence (no validation errors)
        - Field completeness (required fields present)
        - Actionability (all fields have values)
        - Grounding (source and version info present)

        Returns:
            Completeness score between 0.0 and 1.0
        """
        # Base implementation - subclasses should override
        semantic_errors = self.validate_semantics()

        # Local coherence (no validation errors)
        local_coherence = 1.0 if not semantic_errors else max(0.0, 1.0 - len(semantic_errors) / 10)

        # Field completeness (all required fields present)
        required_fields = []
        for field_name, field_info in self.__class__.model_fields.items():
            # Check if field is required (Pydantic v2)
            is_required = field_info.is_required()
            if is_required:
                required_fields.append(field_name)

        # Calculate field completeness, handling case where there are no required fields
        if required_fields:
            field_completeness = sum(1.0 for field in required_fields
                                if getattr(self, field, None) is not None) / len(required_fields)
        else:
            field_completeness = 1.0  # Perfect if no fields are required

        # Actionability (configuration is complete and usable)
        field_values = [getattr(self, field, None) for field in self.__class__.model_fields]
        actionability = sum(1.0 for value in field_values if value is not None) / len(field_values)

        # Grounding (source and versioning information present)
        grounding = 0.5 * (1.0 if self.source else 0.0) + 0.5 * (1.0 if self.config_version else 0.0)

        # Equal weights (0.25 each)
        completeness_score = 0.25 * (local_coherence + field_completeness + actionability + grounding)

        return min(1.0, max(0.0, completeness_score))


class ProcessingConfig(BaseConfig):
    """
    Configuration for document processing operations.

    Covers worker allocation, batch processing, resource limits,
    timeout settings, and GPU configuration.
    """

    # Worker configuration
    workers: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Number of worker processes"
    )

    # Batch processing
    batch_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Batch size for processing"
    )

    # Resource limits
    memory_limit_gb: float | None = Field(
        default=None,
        ge=0.1,
        description="Memory limit in GB"
    )

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Processing timeout in seconds"
    )

    # GPU configuration
    use_gpu: bool = Field(
        default=False,
        description="Enable GPU acceleration"
    )

    gpu_devices: list[int] | None = Field(
        default=None,
        description="GPU device IDs to use"
    )

    def validate_semantics(self) -> list[str]:
        """
        Validate processing configuration semantics.

        Returns:
            List of validation errors
        """
        errors = []

        # GPU validation
        if self.use_gpu and self.gpu_devices is not None:
            if not self.gpu_devices:
                errors.append("GPU devices list cannot be empty when GPU is enabled")
            elif any(device < 0 for device in self.gpu_devices):
                errors.append("GPU device IDs must be non-negative")

        # Resource validation
        if self.memory_limit_gb is not None and self.memory_limit_gb < 1.0:
            if self.workers > 2:
                errors.append(f"Memory limit {self.memory_limit_gb}GB too low for {self.workers} workers")

        # Worker/batch relationship
        if self.batch_size > self.workers * 100:
            errors.append(f"Batch size {self.batch_size} may be too large for {self.workers} workers")

        return errors


class StorageConfig(BaseConfig):
    """
    Configuration for storage operations.

    Covers database connection settings, file storage paths,
    and connection parameters with validation.
    """

    # Database connection
    host: str = Field(
        default="localhost",
        description="Database host"
    )

    port: int = Field(
        default=8529,
        ge=1,
        le=65535,
        description="Database port"
    )

    username: str = Field(
        default="root",
        description="Database username"
    )

    password: str = Field(
        default="",
        exclude=True,  # Exclude from serialization to prevent exposure
        description="Database password"
    )

    database: str = Field(
        default="academy_store",
        min_length=1,
        description="Database name"
    )

    # File storage paths
    pdf_directory: Path | None = Field(
        default=None,
        description="Directory containing PDF files"
    )

    staging_directory: Path | None = Field(
        default=None,
        description="Temporary staging directory"
    )

    # Connection settings
    connection_timeout: int = Field(
        default=30,
        ge=1,
        description="Connection timeout in seconds"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum connection retry attempts"
    )

    def validate_semantics(self) -> list[str]:
        """
        Validate storage configuration semantics.

        Returns:
            List of validation errors
        """
        errors = []

        # Path validation
        if self.pdf_directory is not None:
            if not self.pdf_directory.exists():
                errors.append(f"PDF directory does not exist: {self.pdf_directory}")
            elif not self.pdf_directory.is_dir():
                errors.append(f"PDF directory path is not a directory: {self.pdf_directory}")

        if self.staging_directory is not None:
            # Staging directory can be created if it doesn't exist
            try:
                self.staging_directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create staging directory {self.staging_directory}: {e}")

        # Database validation
        if not self.database.strip():
            errors.append("Database name cannot be empty or whitespace")

        if self.port <= 0 or self.port > 65535:
            errors.append(f"Invalid port number: {self.port}")

        return errors
