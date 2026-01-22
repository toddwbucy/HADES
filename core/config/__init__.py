"""
Core Configuration Module
=========================

Provides hierarchical, validated configuration management for HADES components.

Key Components:
- BaseConfig: Abstract configuration foundation with Pydantic validation
- ConfigLoader: Hierarchical source resolution with schema validation
- ConfigManager: Centralized management with caching and factory patterns
- Built-in configs: ProcessingConfig, StorageConfig for common patterns
"""

from typing import Optional

from .config_base import (
    BaseConfig,
    ProcessingConfig,
    StorageConfig,
    ConfigError,
    ConfigValidationError
)

from .config_loader import (
    ConfigLoader,
    ConfigFormat,
    ConfigSource,
    ConfigSchema
)

from .config_manager import (
    ConfigManager,
    ConfigScope,
    ConfigCache,
    ConfigRegistration,
    config_manager
)

# Version information
__version__ = "1.0.0"

# Export main interfaces
__all__ = [
    # Base classes and errors
    'BaseConfig',
    'ProcessingConfig',
    'StorageConfig',
    'ConfigError',
    'ConfigValidationError',

    # Loader components
    'ConfigLoader',
    'ConfigFormat',
    'ConfigSource',
    'ConfigSchema',

    # Manager components
    'ConfigManager',
    'ConfigScope',
    'ConfigCache',
    'ConfigRegistration',

    # Global instance
    'config_manager',
]


def get_config(name: str, instance_id: Optional[str] = None, **overrides):
    """
    Convenience function to get configuration from global manager.

    Args:
        name: Configuration name
        instance_id: Optional instance identifier
        **overrides: Runtime configuration overrides

    Returns:
        Configuration instance

    Example:
        >>> processing_config = get_config('processing', workers=8)
        >>> storage_config = get_config('storage', 'arxiv-pipeline')
    """
    return config_manager.get(name, instance_id, **overrides)


def register_config(name: str, config_class, scope: ConfigScope = ConfigScope.COMPONENT, **kwargs):
    """
    Convenience function to register configuration with global manager.

    Args:
        name: Configuration name
        config_class: Configuration class
        scope: Configuration scope
        **kwargs: Additional registration parameters

    Example:
        >>> register_config('my_processor', MyProcessorConfig, ConfigScope.COMPONENT)
    """
    config_manager.register(name, config_class, scope, **kwargs)


def create_config_schema(config_class) -> dict:
    """
    Generate JSON schema from Pydantic configuration class.

    Args:
        config_class: Pydantic configuration class

    Returns:
        JSON schema dictionary

    Example:
        >>> schema = create_config_schema(ProcessingConfig)
        >>> config_manager._loader.register_schema('processing', schema)
    """
    # Pydantic v2 uses model_json_schema()
    if hasattr(config_class, 'model_json_schema'):
        return config_class.model_json_schema()
    # Pydantic v1 fallback uses schema()
    elif hasattr(config_class, 'schema'):
        return config_class.schema()
    else:
        # Fallback for non-Pydantic classes
        return {
            'type': 'object',
            'properties': {},
            'required': []
        }


# Default configuration examples for documentation
DEFAULT_PROCESSING_CONFIG = {
    'workers': 4,
    'batch_size': 10,
    'timeout_seconds': 300,
    'use_gpu': False
}

DEFAULT_STORAGE_CONFIG = {
    'host': 'localhost',
    'port': 8529,
    'username': 'root',
    'password': '',
    'database': 'academy_store',
    'connection_timeout': 30,
    'max_retries': 3
}
