"""Configuration handling for HADES CLI.

Resolves environment variables and provides typed configuration access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLIConfig:
    """CLI configuration resolved from environment variables."""

    # ArangoDB settings
    arango_password: str
    arango_host: str = "localhost"
    arango_port: int = 8529
    arango_database: str = "arxiv_datastore"
    arango_ro_socket: str | None = None
    arango_rw_socket: str | None = None

    # ArXiv settings
    pdf_base_path: Path = Path("/bulk-store/arxiv-data/pdf")
    latex_base_path: Path = Path("/bulk-store/arxiv-data/src")

    # Processing settings
    use_gpu: bool = True
    device: str = "cuda"


def get_config() -> CLIConfig:
    """Load configuration from environment variables.

    Required environment variables:
        ARANGO_PASSWORD: Password for ArangoDB authentication

    Optional environment variables:
        ARANGO_HOST: ArangoDB host (default: localhost)
        ARANGO_PORT: ArangoDB port (default: 8529)
        HADES_DATABASE: Database name (default: arxiv_datastore)
        ARANGO_RO_SOCKET: Read-only Unix socket path
        ARANGO_RW_SOCKET: Read-write Unix socket path
        HADES_PDF_PATH: Base path for PDF storage
        HADES_USE_GPU: Whether to use GPU (default: true)
        CUDA_VISIBLE_DEVICES: GPU device selection

    Returns:
        CLIConfig with resolved values

    Raises:
        ValueError: If required environment variables are missing
    """
    password = os.environ.get("ARANGO_PASSWORD")
    if not password:
        raise ValueError("ARANGO_PASSWORD environment variable is required")

    # Determine device
    # When CUDA_VISIBLE_DEVICES is set, PyTorch sees devices as 0, 1, 2...
    # regardless of the actual GPU indices. So we always use "cuda:0" or "cuda"
    use_gpu = os.environ.get("HADES_USE_GPU", "true").lower() in ("true", "1", "yes")

    if use_gpu:
        device = "cuda"  # PyTorch will use the first visible GPU
    else:
        device = "cpu"

    # Parse port with validation
    port_str = os.environ.get("ARANGO_PORT", "8529")
    try:
        arango_port = int(port_str)
    except ValueError as e:
        raise ValueError(f"ARANGO_PORT must be a number, got: {port_str}") from e

    return CLIConfig(
        arango_password=password,
        arango_host=os.environ.get("ARANGO_HOST", "localhost"),
        arango_port=arango_port,
        arango_database=os.environ.get("HADES_DATABASE", "arxiv_datastore"),
        arango_ro_socket=os.environ.get("ARANGO_RO_SOCKET"),
        arango_rw_socket=os.environ.get("ARANGO_RW_SOCKET"),
        pdf_base_path=Path(os.environ.get("HADES_PDF_PATH", "/bulk-store/arxiv-data/pdf")),
        latex_base_path=Path(os.environ.get("HADES_LATEX_PATH", "/bulk-store/arxiv-data/src")),
        use_gpu=use_gpu,
        device=device,
    )


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
