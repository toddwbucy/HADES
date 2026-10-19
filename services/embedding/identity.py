"""Portable artifact/receipt primitives for the future verified Jina adapter.

This module deliberately imports no ML runtime and performs no discovery or
registration. A manifest describes explicitly selected assets; it does not prove
their origin, completeness, or that a live model loaded them. The registering
adapter must supply those guarantees before issuing a serving receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import BinaryIO

MANIFEST_FORMAT = "hades-model-manifest-v1"
RECEIPT_FORMAT = "hades-provider-receipt-v1"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_FILES = 4096


def _require(condition: bool, message: str) -> None:
    """Reject an invalid contract without silently normalizing its identity."""
    if not condition:
        raise ValueError(message)


def _hex(value: str, digits: int) -> None:
    """Validate the lowercase wire encoding of a digest."""
    _require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{digits}}}", value) is not None,
             f"digest must contain exactly {digits} lowercase hexadecimal digits")


def _path(path: str) -> None:
    """Accept unambiguous relative POSIX paths, preserving Unicode and backslashes."""
    _require(isinstance(path, str), "asset path must be a string")
    # Strict encoding rejects lone surrogates; no Unicode normalization is applied.
    _require(0 < len(path.encode("utf-8")) <= 4096 and "\0" not in path
             and all(part not in ("", ".", "..") for part in path.split("/")),
             "asset path must be an unambiguous relative POSIX path")


def _exact_fields(value: dict, fields: set[str]) -> None:
    """Require exact schemas rather than discarding unknown identity fields."""
    _require(isinstance(value, dict) and set(value) == fields, "invalid identity fields")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys, including keys with equal values."""
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class ArtifactFile:
    """One selected asset, with a logical path independent of installation root."""

    path: str
    bytes: int
    md5: str

    def __post_init__(self) -> None:
        """Keep all constructed entries valid, including direct callers."""
        _path(self.path)
        _require(type(self.bytes) is int and 0 <= self.bytes <= 2**64 - 1,
                 "asset bytes must be a u64 integer")
        _hex(self.md5, 32)

    @classmethod
    def from_stream(cls, path: str, source: BinaryIO) -> ArtifactFile:
        """Hash a caller-owned binary stream in bounded memory, without trusting its path."""
        _path(path)
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        while block := source.read(64 * 1024):
            digest.update(block)
            size += len(block)
        return cls(path, size, digest.hexdigest())

    def to_dict(self) -> dict:
        """Return the exact file-entry wire shape."""
        return {"path": self.path, "bytes": self.bytes, "md5": self.md5}


@dataclass(frozen=True)
class ArtifactManifest:
    """An explicit asset set; use canonical_bytes for persisted/hash input bytes."""

    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        """Enforce immutable entries and reject duplicate/file-directory aliases."""
        _require(type(self.files) is tuple and 0 < len(self.files) <= MAX_MANIFEST_FILES,
                 "manifest must contain 1..=4096 required assets")
        _require(all(isinstance(file, ArtifactFile) for file in self.files), "invalid asset entry")
        paths = {file.path for file in self.files}
        _require(len(paths) == len(self.files), "duplicate asset path")
        for path in paths:
            parts = path.split("/")
            _require(all("/".join(parts[:i]) not in paths for i in range(1, len(parts))),
                     "asset path is also used as a directory")
        self.canonical_bytes()

    @classmethod
    def from_json(cls, encoded: bytes) -> ArtifactManifest:
        """Parse bounded strict UTF-8 JSON, rejecting unknown formats and fields."""
        _require(len(encoded) <= MAX_MANIFEST_BYTES, "manifest exceeds 1 MiB")
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
        _exact_fields(value, {"format", "files"})
        _require(value["format"] == MANIFEST_FORMAT, "unknown manifest format")
        _require(type(value["files"]) is list, "files must be an array")
        files = []
        for entry in value["files"]:
            _exact_fields(entry, {"path", "bytes", "md5"})
            files.append(ArtifactFile(**entry))
        return cls(tuple(files))

    def canonical_bytes(self) -> bytes:
        """Encode sorted UTF-8 paths and keys, with exact v1 JSON string escaping."""
        value = {"format": MANIFEST_FORMAT, "files": [
            file.to_dict() for file in sorted(self.files, key=lambda file: file.path.encode("utf-8"))
        ]}
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        _require(len(encoded) <= MAX_MANIFEST_BYTES, "canonical manifest exceeds 1 MiB")
        return encoded

    def digest(self) -> str:
        """Hash canonical manifest bytes, excluding this aggregate digest itself."""
        return hashlib.md5(self.canonical_bytes(), usedforsecurity=False).hexdigest()

    def check_digest(self, expected: str) -> None:
        """Compare to a trusted prior registration, without claiming filesystem verification."""
        _hex(expected, 32)
        _require(self.digest() == expected, "artifact manifest MD5 mismatch; reverify registration")


@dataclass(frozen=True)
class ProviderReceipt:
    """Snapshot/space/generation identity, only to be issued after verified loading."""

    artifact_md5: str
    space_id: str
    serving_generation: str

    def __post_init__(self) -> None:
        """Validate identity syntax without claiming to establish trust."""
        _hex(self.artifact_md5, 32)
        _hex(self.space_id, 64)
        generation = self.serving_generation
        _require(isinstance(generation, str) and bool(generation.strip())
                 and len(generation.encode("utf-8")) <= 4096
                 and not any(ord(c) <= 31 or 127 <= ord(c) <= 159 for c in generation),
                 "serving generation must be nonempty, bounded, and contain no controls")

    @classmethod
    def from_dict(cls, value: dict) -> ProviderReceipt:
        """Read the exact Rust-compatible receipt shape; missing identity fails closed."""
        _exact_fields(value, {"format", "artifact_md5", "space_id", "serving_generation"})
        _require(value["format"] == RECEIPT_FORMAT, "unknown provider receipt format")
        return cls(value["artifact_md5"], value["space_id"], value["serving_generation"])

    def to_dict(self) -> dict:
        """Emit the receipt to pin in a request and repeat in its response."""
        return {"format": RECEIPT_FORMAT, "artifact_md5": self.artifact_md5,
                "space_id": self.space_id, "serving_generation": self.serving_generation}

    def check_matches(self, expected: ProviderReceipt) -> None:
        """Reject artifact, behavior, or reload drift at admission/response boundaries."""
        _require(self == expected, "provider receipt changed; discard vectors and reverify binding")
