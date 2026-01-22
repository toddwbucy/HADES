"""
Structured Logging
==================

Centralized logging configuration using structlog for structured JSON logging.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile
import threading

import structlog


def _get_log_directory() -> Path:
    """
    Get a writable log directory.

    Priority:
    1. LOG_DIR environment variable
    2. Current working directory / logs
    3. System temp directory / hades_logs

    Returns:
        Path to writable log directory
    """
    # Try LOG_DIR env var first
    if env_log_dir := os.environ.get("LOG_DIR"):
        log_dir = Path(env_log_dir)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            if os.access(log_dir, os.W_OK):
                return log_dir
        except OSError:
            pass

    # Try current working directory
    try:
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        if os.access(log_dir, os.W_OK):
            return log_dir
    except OSError:
        pass

    # Fall back to temp directory
    log_dir = Path(tempfile.gettempdir()) / "hades_logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def _validate_log_level(log_level: str) -> int:
    """
    Validate and convert log level string to numeric value.

    Args:
        log_level: Log level name (case-insensitive)

    Returns:
        Numeric logging level

    Raises:
        ValueError: If log_level is not a valid logging level name
    """
    level_name = str(log_level).upper()

    # Check against known level names
    valid_levels = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
    if level_name not in valid_levels:
        raise ValueError(
            f"Invalid log level: '{log_level}'. "
            f"Must be one of: {', '.join(sorted(valid_levels))}"
        )

    return getattr(logging, level_name)

# Global flag and lock for thread-safe initialization
_logging_initialized = False
_init_lock = threading.Lock()


class LogManager:
    """
    Centralized logging configuration using structlog.

    Features:
    - Structured JSON logging
    - Automatic log rotation
    - Context preservation (run_id, processor)
    - Multiple output targets
    - No database logging
    """

    @staticmethod
    def setup(log_level: str = "INFO"):
        """
        Setup logging configuration.

        Args:
            log_level: Default log level
        """
        global _logging_initialized

        # Fast path: already initialized
        if _logging_initialized:
            return

        # Thread-safe initialization
        with _init_lock:
            # Double-check after acquiring lock
            if _logging_initialized:
                return

            # Validate and convert log level
            numeric_level = _validate_log_level(log_level)

            # Setup log directory (configurable, with safe fallbacks)
            log_dir = _get_log_directory()

            # Configure Python logging
            logging.basicConfig(
                level=numeric_level,
                format='%(message)s'
            )

            # Setup handlers
            handlers = []

            # Main log file with rotation (10MB, keep 5 backups)
            main_handler = RotatingFileHandler(
                log_dir / "processors.log",
                maxBytes=10_485_760,
                backupCount=5
            )
            main_handler.setLevel(numeric_level)
            handlers.append(main_handler)

            # Error log file (10MB, keep 3 backups)
            error_handler = RotatingFileHandler(
                log_dir / "errors.log",
                maxBytes=10_485_760,
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            handlers.append(error_handler)

            # Add handlers to root logger
            root_logger = logging.getLogger()
            for handler in handlers:
                root_logger.addHandler(handler)

            # Configure structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.CallsiteParameterAdder(
                        parameters=[
                            structlog.processors.CallsiteParameter.FILENAME,
                            structlog.processors.CallsiteParameter.LINENO,
                        ]
                    ),
                    structlog.processors.dict_tracebacks,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            _logging_initialized = True

            # Log initialization
            logger = structlog.get_logger()
            logger.info("logging_initialized", log_dir=str(log_dir), level=log_level)

    @staticmethod
    def get_logger(processor_name: str, run_id: str):
        """
        Get a logger instance with processor context.

        Args:
            processor_name: Name of the processor
            run_id: Run ID for this processing session

        Returns:
            Configured logger instance
        """
        # Ensure logging is setup
        if not _logging_initialized:
            LogManager.setup()

        return structlog.get_logger().bind(
            processor=processor_name,
            run_id=run_id
        )
