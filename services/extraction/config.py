"""Extraction service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for the extraction service."""

    socket_path: str = "/run/hades/extractor.sock"
    device: str = "auto"
    use_ocr: bool = False
    idle_timeout_seconds: float = 900.0  # 15 min — unload VLM after idle
    formula_enrichment: bool = False  # Extra model inference is opt-in

    @classmethod
    def from_env(cls) -> ExtractionConfig:
        if os.environ.get("HADES_EXTRACTOR_FALLBACK", "").lower() in ("1", "true", "yes"):
            raise ValueError("HADES_EXTRACTOR_FALLBACK is no longer supported; Docling failures fail the document")
        return cls(
            socket_path=os.environ.get(
                "HADES_EXTRACTOR_SOCKET", cls.socket_path
            ),
            device=os.environ.get("HADES_EXTRACTOR_DEVICE", cls.device),
            use_ocr=os.environ.get("HADES_EXTRACTOR_OCR", "").lower()
            in ("1", "true", "yes"),
            idle_timeout_seconds=float(
                os.environ.get("HADES_EXTRACTOR_IDLE_TIMEOUT", cls.idle_timeout_seconds)
            ),
            formula_enrichment=os.environ.get(
                "HADES_EXTRACTOR_FORMULA_ENRICHMENT", "false"
            ).lower()
            in ("1", "true", "yes"),
        )
