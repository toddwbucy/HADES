#!/usr/bin/env python3
"""Extractor Factory with Registry Pattern.

Factory pattern for creating document extractors using a registry.
Supports auto-detection of appropriate extractor based on file extension.

Usage:
    # Auto-detect extractor for a file
    extractor = ExtractorFactory.for_file("paper.pdf")

    # Create specific extractor
    extractor = ExtractorFactory.create("docling")

    # List available extractors
    ExtractorFactory.list_available()

    # Register custom extractor
    @ExtractorFactory.register("custom", extensions=[".xyz"])
    class CustomExtractor(ExtractorBase):
        ...
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

from .extractors_base import ExtractorBase, ExtractorConfig

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """Factory for creating document extractors using registry pattern.

    Manages extractor instantiation with auto-detection based on
    file extension and support for lazy loading.

    Example:
        # Auto-detect appropriate extractor
        extractor = ExtractorFactory.for_file("document.pdf")
        result = extractor.extract("document.pdf")

        # Create specific extractor
        extractor = ExtractorFactory.create("docling")

        # Register custom extractor
        @ExtractorFactory.register("myformat", extensions=[".xyz", ".abc"])
        class MyExtractor(ExtractorBase):
            ...
    """

    # Registry mapping extractor names to classes
    _registry: ClassVar[dict[str, type[ExtractorBase]]] = {}

    # Mapping of file extensions to extractor names
    _extension_map: ClassVar[dict[str, str]] = {}

    # Track which extractors have been auto-registered
    _auto_registered: ClassVar[set[str]] = set()

    @classmethod
    def register(cls, name: str, extensions: list[str] | None = None):
        """Decorator to register an extractor class.

        Args:
            name: Name to register under (e.g., "docling", "latex")
            extensions: Optional list of file extensions this extractor handles

        Returns:
            Decorator function

        Example:
            @ExtractorFactory.register("myformat", extensions=[".xyz"])
            class MyExtractor(ExtractorBase):
                ...
        """
        def decorator(extractor_class: type[ExtractorBase]) -> type[ExtractorBase]:
            cls._registry[name] = extractor_class
            logger.debug(f"Registered extractor: {name}")

            # Register extensions if provided
            if extensions:
                for ext in extensions:
                    ext_lower = ext.lower()
                    if not ext_lower.startswith("."):
                        ext_lower = f".{ext_lower}"
                    cls._extension_map[ext_lower] = name
                    logger.debug(f"  → extension {ext_lower}")

            return extractor_class
        return decorator

    @classmethod
    def create(
        cls,
        extractor_type: str,
        config: ExtractorConfig | None = None,
        **kwargs,
    ) -> ExtractorBase:
        """Create an extractor instance by type name.

        Args:
            extractor_type: Type of extractor ("docling", "latex", "code", etc.)
            config: Optional extractor configuration
            **kwargs: Additional arguments for the extractor

        Returns:
            Extractor instance

        Raises:
            ValueError: If no extractor registered for the type
        """
        # Try auto-registration if not already registered
        if extractor_type not in cls._registry:
            cls._auto_register(extractor_type)

        if extractor_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"No extractor registered for '{extractor_type}'. "
                f"Available: {available}"
            )

        extractor_class = cls._registry[extractor_type]
        logger.info(f"Creating {extractor_type} extractor")

        if config is not None:
            return extractor_class(config=config, **kwargs)
        return extractor_class(**kwargs)

    @classmethod
    def for_file(
        cls,
        file_path: str | Path,
        config: ExtractorConfig | None = None,
        **kwargs,
    ) -> ExtractorBase:
        """Get an appropriate extractor for a file based on its extension.

        Args:
            file_path: Path to the file
            config: Optional extractor configuration
            **kwargs: Additional arguments for the extractor

        Returns:
            Extractor instance appropriate for the file type

        Raises:
            ValueError: If no extractor can handle the file type
        """
        # Ensure built-in extractors are registered
        cls._ensure_registered()

        path = Path(file_path)
        ext = path.suffix.lower()

        # Check extension map
        if ext in cls._extension_map:
            extractor_type = cls._extension_map[ext]
            return cls.create(extractor_type, config=config, **kwargs)

        # Try to find an extractor that supports this extension
        for name, extractor_class in cls._registry.items():
            try:
                # Instantiate with config/kwargs to avoid false negatives
                temp_instance = extractor_class(config=config, **kwargs)
                if ext in [f.lower() for f in temp_instance.supported_formats]:
                    # Return the already-configured instance instead of creating another
                    return temp_instance
            except Exception as e:
                logger.debug(f"Extractor '{name}' probe failed for {ext}: {e}")
                continue

        # Fallback to robust extractor if available
        if "robust" in cls._registry:
            logger.warning(f"No specific extractor for {ext}, using robust fallback")
            return cls.create("robust", config=config, **kwargs)

        raise ValueError(f"No extractor available for file type: {ext}")

    @classmethod
    def _auto_register(cls, extractor_type: str) -> None:
        """Auto-register an extractor type on first use.

        Enables lazy loading of extractor modules.

        Args:
            extractor_type: Type of extractor to register
        """
        if extractor_type in cls._auto_registered:
            return

        try:
            if extractor_type == "docling":
                from .extractors_docling import DoclingExtractor
                cls._registry["docling"] = DoclingExtractor
                # Register extensions from DoclingExtractor.supported_formats
                for ext in DoclingExtractor().supported_formats:
                    cls._extension_map[ext.lower()] = "docling"
                cls._auto_registered.add(extractor_type)
                logger.debug("Auto-registered docling extractor")

            elif extractor_type == "latex":
                from .extractors_latex import LaTeXExtractor
                cls._registry["latex"] = LaTeXExtractor
                cls._extension_map[".tex"] = "latex"
                cls._auto_registered.add(extractor_type)
                logger.debug("Auto-registered latex extractor")

            elif extractor_type == "code":
                from .extractors_code import CodeExtractor
                cls._registry["code"] = CodeExtractor
                for ext in [".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"]:
                    cls._extension_map[ext] = "code"
                cls._auto_registered.add(extractor_type)
                logger.debug("Auto-registered code extractor")

            # Note: TreeSitterExtractor is a symbol extractor, not a document extractor.
            # It doesn't implement ExtractorBase interface (extract/extract_batch).
            # Use TreeSitterExtractor directly for code symbol extraction.

            elif extractor_type == "robust":
                from .extractors_robust import RobustExtractor
                cls._registry["robust"] = RobustExtractor
                cls._auto_registered.add(extractor_type)
                logger.debug("Auto-registered robust extractor")

            else:
                # Mark unknown types to prevent repeated warnings
                cls._auto_registered.add(extractor_type)
                logger.warning(f"Unknown extractor type for auto-registration: {extractor_type}")

        except ImportError:
            logger.exception(f"Could not auto-register {extractor_type}")

    @classmethod
    def _ensure_registered(cls) -> None:
        """Ensure all built-in extractors are registered."""
        for extractor_type in ["docling", "latex", "code", "robust"]:
            if extractor_type not in cls._registry:
                cls._auto_register(extractor_type)

    @classmethod
    def list_available(cls) -> dict[str, dict[str, Any]]:
        """List available extractors and their supported formats.

        Returns:
            Dictionary mapping extractor names to their info
        """
        cls._ensure_registered()

        available = {}
        for name, extractor_class in cls._registry.items():
            try:
                # Get class-level info
                info: dict[str, Any] = {
                    "class": extractor_class.__name__,
                    "module": extractor_class.__module__,
                }

                # Try to get supported formats without full instantiation
                try:
                    temp = extractor_class()
                    info["supported_formats"] = temp.supported_formats
                    info["supports_gpu"] = temp.supports_gpu
                    info["supports_batch"] = temp.supports_batch
                    info["supports_ocr"] = temp.supports_ocr
                except Exception as e:
                    logger.debug("Could not introspect extractor '%s': %s", name, e)

                available[name] = info

            except Exception as e:
                available[name] = {"error": str(e)}

        return available

    @classmethod
    def get_extensions_map(cls) -> dict[str, str]:
        """Get the mapping of file extensions to extractor types.

        Returns:
            Dictionary mapping extensions to extractor names
        """
        cls._ensure_registered()
        return dict(cls._extension_map)

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if any extractor supports a given file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            True if an extractor can handle this extension
        """
        cls._ensure_registered()
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext in cls._extension_map
