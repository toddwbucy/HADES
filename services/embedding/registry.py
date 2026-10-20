"""Local administrative registration of approved, already downloaded Jina assets.

Run from services/ with ``python -m embedding.registry register --help``.
This is an operator command, never a browser-controlled source or digest API.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

from .identity import ArtifactManifest, MAX_MANIFEST_BYTES, MAX_MANIFEST_FILES, _exact_fields, _unique_object
from .snapshot import (DEFAULT_MAX_BYTES, ServingSnapshot, _stamp, copy_assets,
                       freeze_tree, remove_private_tree)

REGISTRATION_FORMAT = "hades-model-registration-v1"
MODEL_ID = "jinaai/jina-embeddings-v4"
_RUNTIME_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt", ".py", ".tiktoken", ".vocab", ".merges"}
_METADATA_DIRS = {".git", ".cache", "__pycache__"}


def _provider_id(value: str) -> None:
    """Restrict provider IDs to bounded names usable as registry keys, never paths."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("provider ID must match [a-z][a-z0-9_-]{0,63}")


def _private_directory(path: Path) -> None:
    """Require an existing service-owned 0700 directory without following its final link."""
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("registry directories must be owned by this account with mode 0700")


def _inventory(source: Path) -> dict:
    """Select Jina runtime files, excluding known cache/documentation metadata.

    Unknown file types and directory links are refused so assets are not silently
    omitted. Cache file links are allowed and their resolved contents are copied.
    Stamps let registration reject a source changing across the complete copy.
    """
    result = {}
    pending = [source]
    count = 0
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            count += 1
            if count > MAX_MANIFEST_FILES * 2:
                raise ValueError("source inventory exceeds registration limit")
            if path.name in _METADATA_DIRS or path.name in {".gitattributes", ".gitignore", "results.json"}:
                continue
            if path.suffix == ".md" or path.name.upper().startswith(("LICENSE", "NOTICE")):
                continue
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
                continue
            resolved = path.stat()
            if not stat.S_ISREG(resolved.st_mode) or path.suffix not in _RUNTIME_SUFFIXES:
                raise ValueError("source contains an unsupported asset or directory link; use a clean model snapshot")
            result[path.relative_to(source).as_posix()] = (_stamp(info), _stamp(resolved))
    if not 0 < len(result) <= MAX_MANIFEST_FILES:
        raise ValueError("source must contain 1..=4096 runtime assets")
    return result


def _read_json(path: Path, limit: int, require_object: bool = True):
    """Read bounded UTF-8 metadata with duplicate-key rejection."""
    with path.open("rb") as stream:
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("registration metadata exceeds size limit")
    value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    if require_object and not isinstance(value, dict):
        raise ValueError("registration metadata must be a JSON object")
    return value


def _validate_jina_assets(root: Path, manifest: ArtifactManifest) -> None:
    """Check local tokenizer, safetensors, adapter weights, and auto_map dependencies.

    Custom Python code is trusted operator-selected code, not statically proven
    dependency-free. The verified loader additionally requires offline operation.
    """
    paths = {file.path for file in manifest.files}
    required = {"config.json", "tokenizer_config.json", "tokenizer.json",
                "preprocessor_config.json", "adapters/adapter_config.json"}
    if not required <= paths:
        raise ValueError("Jina registration is missing required model, tokenizer, processor, or adapter assets")
    if "model.safetensors.index.json" in paths:
        index = _read_json(root / "model.safetensors.index.json", 4 * MAX_MANIFEST_BYTES)
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights or any(
            not isinstance(path, str) or path not in paths or not path.endswith(".safetensors")
            for path in weights.values()
        ):
            raise ValueError("weight index references missing or unsupported shards")
    elif "model.safetensors" not in paths:
        raise ValueError("Jina registration requires local safetensors weights")
    for logical in sorted(paths):
        if logical.endswith(".json") and not logical.endswith("tokenizer.json"):
            config = _read_json(root / logical, 4 * MAX_MANIFEST_BYTES, require_object=logical in required)
            if not isinstance(config, dict):
                continue
            mapping = config.get("auto_map", {})
            if not isinstance(mapping, dict):
                raise ValueError("auto_map must be an object")
            for reference in mapping.values():
                for entry in reference if isinstance(reference, list) else [reference]:
                    if entry is None:
                        continue
                    if not isinstance(entry, str) or "--" in entry or "." not in entry:
                        raise ValueError("remote or invalid auto_map references cannot be registered")
                    module = entry.rsplit(".", 1)[0].replace(".", "/") + ".py"
                    if module not in paths:
                        raise ValueError("auto_map references missing local model code")
            if Path(logical).name == "adapter_config.json":
                parent = Path(logical).parent
                if not any((parent / name).as_posix() in paths for name in
                           ("adapter_model.safetensors", "adapter_model.bin")):
                    raise ValueError("adapter configuration has no local adapter weights")


@dataclass(frozen=True)
class RegisteredModel:
    """Expected identity read from a protected local registration, not an HTTP claim."""

    provider_id: str
    model_id: str
    revision: str
    manifest: ArtifactManifest
    artifacts: Path
    sessions: Path

    def prepare(self, max_bytes: int = DEFAULT_MAX_BYTES) -> ServingSnapshot:
        """Reverify persisted assets into a private serving copy for a fresh load."""
        _private_directory(self.sessions)
        return ServingSnapshot.prepare(self.artifacts, self.manifest, self.sessions, max_bytes)


class ModelRegistry:
    """An append-only registry; existing IDs cannot be silently rebound to new weights."""

    def __init__(self, root: Path) -> None:
        """Create private storage under an explicitly selected operator-owned directory."""
        self.root = Path(root).absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _private_directory(self.root)
        for name in ("records", "artifacts", "sessions"):
            path = self.root / name
            path.mkdir(mode=0o700, exist_ok=True)
            _private_directory(path)

    @contextmanager
    def _locked(self):
        """Serialize registration and reads, including independent operator processes."""
        for path in (self.root, self.root / "records", self.root / "artifacts", self.root / "sessions"):
            _private_directory(path)
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "rb+") as lock:
            info = os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("unsafe registry lock ownership or permissions")
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def register(self, provider_id: str, source: Path, revision: str,
                 max_bytes: int = DEFAULT_MAX_BYTES) -> RegisteredModel:
        """Copy approved local assets and atomically publish their expected MD5.

        The operator supplies a resolved Hugging Face commit, not a mutable tag.
        Approval of that source is the trust boundary; MD5 does not authenticate it.
        """
        _provider_id(provider_id)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a resolved 40-character lowercase commit ID")
        source = Path(source).resolve(strict=True)
        with self._locked():
            record_path = self.root / "records" / f"{provider_id}.json"
            if record_path.exists() or record_path.is_symlink():
                raise ValueError("provider ID already registered; choose a new ID for changed artifacts")
            original = _inventory(source)
            snapshot_id = "model_" + uuid.uuid4().hex
            root = self.root / "artifacts" / snapshot_id
            root.mkdir(mode=0o700)
            temp = None
            published = False
            try:
                manifest = copy_assets(source, root, list(original), max_bytes)
                if original != _inventory(source):
                    raise ValueError("model source changed during registration")
                _validate_jina_assets(root, manifest)
                freeze_tree(root)
                for directory, _, _ in os.walk(root, topdown=False):
                    self._sync_directory(Path(directory))
                self._sync_directory(root.parent)
                record = {"format": REGISTRATION_FORMAT, "provider_id": provider_id,
                          "adapter": "jina-v4", "model_id": MODEL_ID, "revision": revision,
                          "manifest": json.loads(manifest.canonical_bytes()),
                          "manifest_md5": manifest.digest(), "snapshot": snapshot_id}
                encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                fd, name = tempfile.mkstemp(prefix=".register_", dir=self.root / "records")
                temp = Path(name)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                # link is exclusive even if another actor bypasses our lock.
                os.link(temp, record_path)
                published = True
                self._sync_directory(record_path.parent)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
                if not published:
                    remove_private_tree(root)
        return self.load(provider_id)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        """Flush a published directory entry before reporting registration success."""
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def load(self, provider_id: str) -> RegisteredModel:
        """Read an existing protected record and validate its manifest and local paths."""
        _provider_id(provider_id)
        with self._locked():
            path = self.root / "records" / f"{provider_id}.json"
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("registration records must be private regular files owned by this account")
            value = _read_json(path, 2 * MAX_MANIFEST_BYTES)
            _exact_fields(value, {"format", "provider_id", "adapter", "model_id", "revision",
                                  "manifest", "manifest_md5", "snapshot"})
            if (value["format"] != REGISTRATION_FORMAT or value["provider_id"] != provider_id
                    or value["adapter"] != "jina-v4" or value["model_id"] != MODEL_ID
                    or not isinstance(value["revision"], str)
                    or not re.fullmatch(r"[0-9a-f]{40}", value["revision"])
                    or not isinstance(value["snapshot"], str)
                    or not re.fullmatch(r"model_[0-9a-f]{32}", value["snapshot"])):
                raise ValueError("invalid model registration identity")
            manifest = ArtifactManifest.from_json(json.dumps(value["manifest"], ensure_ascii=False,
                                                              separators=(",", ":")).encode())
            manifest.check_digest(value["manifest_md5"])
            artifacts = self.root / "artifacts" / value["snapshot"]
            info = artifacts.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o277:
                raise ValueError("registered artifact directory must be private and read-only")
            return RegisteredModel(provider_id, value["model_id"], value["revision"], manifest,
                                   artifacts, self.root / "sessions")


def main(argv: list[str] | None = None) -> int:
    """Expose local register/inspect operations without starting an ML runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="register an approved local Jina snapshot")
    register.add_argument("--provider", required=True)
    register.add_argument("--source", required=True, type=Path)
    register.add_argument("--revision", required=True)
    register.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    inspect = commands.add_parser("inspect", help="inspect a previously trusted registration")
    inspect.add_argument("--provider", required=True)
    args = parser.parse_args(argv)
    try:
        registry = ModelRegistry(args.registry)
        model = (registry.register(args.provider, args.source, args.revision, args.max_bytes)
                 if args.command == "register" else registry.load(args.provider))
        print(json.dumps({"provider_id": model.provider_id, "model_id": model.model_id,
                          "revision": model.revision, "manifest_md5": model.manifest.digest(),
                          "files": len(model.manifest.files), "loaded_verified": False}))
    except (OSError, ValueError) as error:
        parser.exit(1, f"registration failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
