#!/usr/bin/env python3
"""Prepare private, versioned evaluation passages without services or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_CORPUS_BYTES = 64 * 1024 * 1024
MAX_PASSAGES = 20_000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def spans(raw, limit):
    """Partition UTF-8 bytes losslessly, preferring a newline within each cap."""
    raw.decode("utf-8")  # Never replace invalid bytes and corrupt citation offsets.
    start = 0
    while start < len(raw):
        end = min(start + limit, len(raw))
        if end < len(raw):
            newline = raw.rfind(b"\n", start, end)
            if newline >= start:
                end = newline + 1
            else:
                while end > start and raw[end] & 0xC0 == 0x80:
                    end -= 1
        if end == start:
            raise ValueError("byte cap cannot hold the next UTF-8 character")
        yield start, end
        start = end


def prepare(snapshot, max_bytes=4096):
    """Verify snapshot files before returning an unlabeled passage artifact."""
    if type(max_bytes) is not int or not 4 <= max_bytes <= 65536:
        raise ValueError("max_bytes must be between 4 and 65536")
    snapshot = Path(snapshot).resolve(strict=True)
    with (snapshot / "manifest.json").open("rb") as stream:
        manifest_bytes = stream.read(1024 * 1024 + 1)
    if len(manifest_bytes) > 1024 * 1024:
        raise ValueError("manifest exceeds byte limit")
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_PASSAGES:
        raise ValueError("manifest needs a bounded nonempty document list")
    root = (snapshot / "documents").resolve(strict=True)
    if not root.is_relative_to(snapshot):
        raise ValueError("documents directory escapes snapshot")
    seen, documents, sources = set(), [], []
    total = 0
    for entry in entries:
        name = entry["path"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid document path")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("document path must be normalized and relative")
        if name in seen:
            raise ValueError("duplicate document path")
        seen.add(name)
        expected = entry["bytes"]
        if type(expected) is not int or not 0 <= expected <= MAX_DOCUMENT_BYTES:
            raise ValueError("document exceeds byte limit")
        total += expected
        if total > MAX_CORPUS_BYTES:
            raise ValueError("corpus exceeds byte limit")
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("document escapes snapshot")
        with path.open("rb") as stream:
            raw = stream.read(expected + 1)
        if len(raw) != expected or digest(raw) != entry["sha256"]:
            raise ValueError("snapshot size/hash mismatch")
        source_hash = digest(raw)
        sources.append({"source": name, "bytes": expected, "sha256": source_hash})
        line = 1
        for start, end in spans(raw, max_bytes):
            if len(documents) >= MAX_PASSAGES:
                raise ValueError("corpus exceeds passage limit")
            text_bytes = raw[start:end]
            newline_count = text_bytes.count(b"\n")
            end_line = line + newline_count - int(text_bytes.endswith(b"\n"))
            identity = json.dumps([name, source_hash, start, end], separators=(",", ":")).encode()
            documents.append({"id": "passage-" + digest(identity), "source": name,
                "source_sha256": source_hash, "start_byte": start, "end_byte": end,
                "start_line": line, "end_line": end_line,
                "text_sha256": digest(text_bytes), "text": text_bytes.decode("utf-8")})
            line += newline_count
    if not documents:
        raise ValueError("corpus contains no passages")
    return {"version": 1, "manifest_sha256": digest(manifest_bytes),
        "preparer_sha256": digest(Path(__file__).read_bytes()),
        "passage_policy": {"name": "utf8_newline_bytes_v1", "max_bytes": max_bytes,
            "overlap_bytes": 0, "byte_offsets": "zero-based, end-exclusive",
            "line_offsets": "one-based, inclusive"},
        "scoring_policy": "complete_top10", "sources": sources,
        "documents": documents, "queries": [],
        "limitations": ["Evaluation passages, not production extraction or chunking parity.",
            "No query judgments, inference, quality score or implementation validation."]}


def write_private(path, artifact):
    """Create a private artifact exclusively; never replace an existing result."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=4096)
    args = parser.parse_args()
    artifact = prepare(args.snapshot, args.max_bytes)
    write_private(args.output, artifact)
    print(json.dumps({"sources": len(artifact["sources"]), "passages": len(artifact["documents"]),
        "artifact_sha256": digest(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
