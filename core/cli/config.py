"""Configuration handling for HADES CLI.

Loads configuration from YAML, environment variables, and CLI arguments.
Override priority (highest wins):
  1. CLI arguments
  2. Environment variables
  3. YAML config file
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config file location (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "hades.yaml"


def _load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file (uses DEFAULT_CONFIG_PATH if not specified)

    Returns:
        Configuration dictionary, empty dict if file not found or empty

    Raises:
        TypeError: If YAML file exists but contains non-mapping content
            (e.g., a list or scalar instead of a dict)
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            loaded = yaml.safe_load(f)
        # Handle empty file
        if loaded is None:
            return {}
        # Validate that loaded content is a mapping (dict)
        if not isinstance(loaded, dict):
            raise TypeError(
                f"Config file {path} must contain a YAML mapping (dict) at the top level, "
                f"got {type(loaded).__name__}. Expected format: 'key: value' pairs."
            )
        return loaded
    logger.debug(f"Config file not found: {path}")
    return {}


def _get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Get a nested value from a dict using dot-notation keys.

    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key not found

    Returns:
        Value at the nested key path, or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""

    # Service settings
    service_socket: str = "/run/hades/embedder.sock"
    fallback_to_local: bool = True
    timeout_ms: int = 30000

    # Model settings
    model_name: str = "jinaai/jina-embeddings-v4"
    dimension: int = 2048
    max_tokens: int = 32768
    use_fp16: bool = True
    normalize: bool = True

    # Batch settings
    batch_size: int = 48
    batch_size_small: int = 8

    # Chunking settings
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 200


@dataclass
class SearchConfig:
    """Search configuration."""

    limit: int = 10
    max_limit: int = 100
    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3


@dataclass
class RocchioConfig:
    """Rocchio relevance feedback configuration."""

    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15


@dataclass
class SyncConfig:
    """Sync configuration."""

    default_lookback_days: int = 7
    batch_size: int = 8
    max_results: int = 1000


@dataclass
class CLIConfig:
    """CLI configuration resolved from YAML + environment variables."""

    # ArangoDB settings
    arango_password: str
    arango_host: str = "localhost"
    arango_port: int = 8529
    arango_database: str = "arxiv_datastore"
    arango_ro_socket: str | None = None
    arango_rw_socket: str | None = None

    # ArXiv settings
    pdf_base_path: Path = field(default_factory=lambda: Path("/bulk-store/arxiv-data/pdf"))
    latex_base_path: Path = field(default_factory=lambda: Path("/bulk-store/arxiv-data/src"))

    # Processing settings
    use_gpu: bool = True
    device: str = "cuda:2"  # Default to inference GPU, keep training GPUs free

    # Collection profile
    collection_profile: str = "arxiv"

    # Nested configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rocchio: RocchioConfig = field(default_factory=RocchioConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)


def get_config(config_path: Path | None = None) -> CLIConfig:
    """Load configuration from YAML and environment variables.

    Configuration sources (override priority, highest wins):
        1. Environment variables (ARANGO_*, HADES_*)
        2. YAML config file (core/config/hades.yaml)

    Required environment variables:
        ARANGO_PASSWORD: Password for ArangoDB authentication

    Optional environment variables:
        ARANGO_HOST: ArangoDB host (default: localhost)
        ARANGO_PORT: ArangoDB port (default: 8529)
        HADES_DATABASE: Database name (default: arxiv_datastore)
        ARANGO_RO_SOCKET: Read-only Unix socket path for ArangoDB
        ARANGO_RW_SOCKET: Read-write Unix socket path for ArangoDB
        HADES_PDF_PATH: Base path for arxiv PDF storage
        HADES_LATEX_PATH: Base path for arxiv LaTeX source storage
        HADES_USE_GPU: Whether to use GPU for embeddings (true/false)
        HADES_EMBEDDER_SOCKET: Embedding service Unix socket path

    Note:
        CUDA_VISIBLE_DEVICES is handled separately by the CLI layer
        (see _set_gpu in main.py) before this function is called.

    Returns:
        CLIConfig with resolved values

    Raises:
        ValueError: If required environment variables are missing
    """
    # Load YAML defaults
    yaml_config = _load_yaml_config(config_path)

    # Get password (required, env only - never in YAML for security)
    password = os.environ.get("ARANGO_PASSWORD")
    if not password:
        raise ValueError("ARANGO_PASSWORD environment variable is required")

    # Determine device from env or YAML
    use_gpu_env = os.environ.get("HADES_USE_GPU")
    if use_gpu_env is not None:
        use_gpu = use_gpu_env.lower() in ("true", "1", "yes")
    else:
        use_gpu = _get_nested(yaml_config, "gpu", "enabled", default=True)

    if use_gpu:
        device = _get_nested(yaml_config, "gpu", "device", default="cuda")
    else:
        device = "cpu"

    # Parse port with validation
    port_str = os.environ.get("ARANGO_PORT") or str(_get_nested(yaml_config, "database", "port", default=8529))
    try:
        arango_port = int(port_str)
    except ValueError as e:
        raise ValueError(f"ARANGO_PORT must be a number, got: {port_str}") from e

    # Build embedding config
    embedding_yaml = yaml_config.get("embedding", {})
    embedding_config = EmbeddingConfig(
        service_socket=os.environ.get("HADES_EMBEDDER_SOCKET")
        or _get_nested(embedding_yaml, "service", "socket", default="/run/hades/embedder.sock"),
        fallback_to_local=_get_nested(embedding_yaml, "service", "fallback_to_local", default=True),
        timeout_ms=_get_nested(embedding_yaml, "service", "timeout_ms", default=30000),
        model_name=_get_nested(embedding_yaml, "model", "name", default="jinaai/jina-embeddings-v4"),
        dimension=_get_nested(embedding_yaml, "model", "dimension", default=2048),
        max_tokens=_get_nested(embedding_yaml, "model", "max_tokens", default=32768),
        use_fp16=_get_nested(embedding_yaml, "model", "use_fp16", default=True),
        normalize=_get_nested(embedding_yaml, "model", "normalize", default=True),
        batch_size=_get_nested(embedding_yaml, "batch", "size", default=48),
        batch_size_small=_get_nested(embedding_yaml, "batch", "size_small", default=8),
        chunk_size_tokens=_get_nested(embedding_yaml, "chunking", "size_tokens", default=500),
        chunk_overlap_tokens=_get_nested(embedding_yaml, "chunking", "overlap_tokens", default=200),
    )

    # Build search config
    search_yaml = yaml_config.get("search", {})
    search_config = SearchConfig(
        limit=_get_nested(search_yaml, "limit", default=10),
        max_limit=_get_nested(search_yaml, "max_limit", default=100),
        hybrid_vector_weight=_get_nested(search_yaml, "hybrid", "vector_weight", default=0.7),
        hybrid_keyword_weight=_get_nested(search_yaml, "hybrid", "keyword_weight", default=0.3),
    )

    # Build rocchio config (use _get_nested to safely handle non-dict values)
    rocchio_config = RocchioConfig(
        alpha=_get_nested(yaml_config, "rocchio", "alpha", default=1.0),
        beta=_get_nested(yaml_config, "rocchio", "beta", default=0.75),
        gamma=_get_nested(yaml_config, "rocchio", "gamma", default=0.15),
    )

    # Build sync config (use _get_nested to safely handle non-dict values)
    sync_config = SyncConfig(
        default_lookback_days=_get_nested(yaml_config, "sync", "default_lookback_days", default=7),
        batch_size=_get_nested(yaml_config, "sync", "batch_size", default=8),
        max_results=_get_nested(yaml_config, "sync", "max_results", default=1000),
    )

    # Socket paths: env overrides YAML
    ro_socket = os.environ.get("ARANGO_RO_SOCKET") or _get_nested(yaml_config, "database", "sockets", "readonly")
    rw_socket = os.environ.get("ARANGO_RW_SOCKET") or _get_nested(yaml_config, "database", "sockets", "readwrite")

    return CLIConfig(
        arango_password=password,
        arango_host=os.environ.get("ARANGO_HOST") or _get_nested(yaml_config, "database", "host", default="localhost"),
        arango_port=arango_port,
        arango_database=os.environ.get("HADES_DATABASE")
        or _get_nested(yaml_config, "database", "database", default="arxiv_datastore"),
        arango_ro_socket=ro_socket,
        arango_rw_socket=rw_socket,
        pdf_base_path=Path(
            os.environ.get("HADES_PDF_PATH")
            or _get_nested(yaml_config, "arxiv", "pdf_base_path", default="/bulk-store/arxiv-data/pdf")
        ),
        latex_base_path=Path(
            os.environ.get("HADES_LATEX_PATH")
            or _get_nested(yaml_config, "arxiv", "latex_base_path", default="/bulk-store/arxiv-data/src")
        ),
        use_gpu=use_gpu,
        device=device,
        embedding=embedding_config,
        search=search_config,
        rocchio=rocchio_config,
        sync=sync_config,
    )


def get_embedder_client(config: CLIConfig | None = None):
    """Get an EmbedderClient configured from CLI config.

    Uses the embedding service over Unix socket. No local fallback —
    if the service is down, commands fail with a clear error.

    Args:
        config: CLI configuration (loaded if not provided)

    Returns:
        EmbedderClient instance (use as context manager)
    """
    if config is None:
        config = get_config()
    from core.services.embedder_client import EmbedderClient

    return EmbedderClient(
        socket_path=config.embedding.service_socket,
        timeout=config.embedding.timeout_ms / 1000.0,
        fallback_to_local=False,
    )


def get_embedder_service_config(config_path: Path | None = None) -> dict:
    """Load embedder service configuration from YAML and environment variables.

    Configuration sources (override priority, highest wins):
        1. Environment variables (HADES_EMBEDDER_*)
        2. YAML config file (core/config/hades.yaml)

    Environment variables:
        HADES_EMBEDDER_DEVICE: GPU device (e.g., cuda:0, cuda:2, cpu)
        HADES_EMBEDDER_MODEL: Model name
        HADES_EMBEDDER_FP16: Use half precision (true/false)
        HADES_EMBEDDER_BATCH_SIZE: Batch size for embedding
        HADES_EMBEDDER_IDLE_TIMEOUT: Seconds before unloading idle model

    Returns:
        Dictionary with embedder service configuration:
            - device: GPU device string
            - model_name: Model identifier
            - use_fp16: Whether to use half precision
            - batch_size: Batch size for embedding
            - idle_timeout: Seconds before unloading model (0 = never)
    """
    yaml_config = _load_yaml_config(config_path)

    # Device priority: env var > service-specific yaml > global gpu yaml > default
    device = os.environ.get("HADES_EMBEDDER_DEVICE")
    if not device:
        device = _get_nested(yaml_config, "embedding", "service", "device")
    if not device:
        device = _get_nested(yaml_config, "gpu", "device")
    if not device:
        device = "cuda:2"  # Default to inference GPU

    # Check if GPU is disabled
    use_gpu_env = os.environ.get("HADES_USE_GPU")
    if use_gpu_env is not None:
        use_gpu = use_gpu_env.lower() in ("true", "1", "yes")
    else:
        use_gpu = _get_nested(yaml_config, "gpu", "enabled", default=True)

    if not use_gpu:
        device = "cpu"

    return {
        "device": device,
        "model_name": os.environ.get("HADES_EMBEDDER_MODEL")
        or _get_nested(yaml_config, "embedding", "model", "name", default="jinaai/jina-embeddings-v4"),
        "use_fp16": (
            os.environ.get("HADES_EMBEDDER_FP16", "").lower() in ("true", "1", "yes")
            if os.environ.get("HADES_EMBEDDER_FP16")
            else _get_nested(yaml_config, "embedding", "model", "use_fp16", default=True)
        ),
        "batch_size": int(
            os.environ.get("HADES_EMBEDDER_BATCH_SIZE")
            or _get_nested(yaml_config, "embedding", "batch", "size", default=48)
        ),
        "idle_timeout": int(
            os.environ.get("HADES_EMBEDDER_IDLE_TIMEOUT")
            or _get_nested(yaml_config, "embedding", "service", "idle_timeout", default=300)
        ),
    }


def get_arango_config(config: CLIConfig | None = None, read_only: bool = True) -> dict:
    """Get ArangoDB connection configuration.

    Args:
        config: CLI configuration (loaded if not provided)
        read_only: Whether to use read-only socket if available

    Returns:
        Dictionary suitable for ArangoDB client initialization
    """
    if config is None:
        config = get_config()

    socket_path = config.arango_ro_socket if read_only else config.arango_rw_socket

    return {
        "host": config.arango_host,
        "port": config.arango_port,
        "database": config.arango_database,
        "username": "root",
        "password": config.arango_password,
        "socket_path": socket_path,
    }
