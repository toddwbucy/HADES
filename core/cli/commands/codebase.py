"""Codebase knowledge graph CLI commands.

Provides ingest, update, and stats commands for indexing the
repository's Python files into the codebase knowledge graph.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.cli.commands.database import _make_client
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)
from core.database.codebase_collections import (
    CODEBASE_COLLECTIONS,
    CodebaseCollections,
    ensure_codebase_collections,
)
from core.database.keys import chunk_key

logger = logging.getLogger(__name__)

# Directories to always skip
_DEFAULT_EXCLUDE = {"Acheron", "__pycache__", ".git", ".tox", ".mypy_cache", ".ruff_cache"}


def _git_python_files(repo_root: str, exclude: set[str] | None = None) -> list[str]:
    """List tracked + untracked (but not ignored) Python files via git."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=repo_root,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {out.stderr.strip()}")

    skip = exclude or _DEFAULT_EXCLUDE
    files: list[str] = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Check if any path component is in the exclude set
        parts = Path(line).parts
        if any(p in skip for p in parts):
            continue
        files.append(line)

    return files


def codebase_ingest(
    path: str,
    start_time: float,
    *,
    force: bool = False,
    exclude: list[str] | None = None,
    collections: CodebaseCollections | None = None,
) -> CLIResponse:
    """Ingest repository Python files into the codebase knowledge graph.

    1. git ls-files to get file list
    2. Extract + AST-chunk each file via CodeProcessor
    3. Store file docs, chunks, embeddings
    4. Resolve imports → create edges
    """
    repo_root = str(Path(path).resolve())
    cols = collections or CODEBASE_COLLECTIONS

    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="codebase.ingest",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_codebase_collections(client, db_name, cols)

        # Get file list
        exclude_set = _DEFAULT_EXCLUDE | set(exclude or [])
        py_files = _git_python_files(repo_root, exclude_set)
        if not py_files:
            return error_response(
                command="codebase.ingest",
                code=ErrorCode.FILE_NOT_FOUND,
                message="No Python files found in repository",
                start_time=start_time,
            )

        # Check existing files (skip unchanged unless --force)
        existing_hashes: dict[str, str] = {}
        if not force:
            existing_hashes = _get_existing_hashes(client, db_name, cols)

        from core.processors.code_processor import CodeProcessor

        processor = CodeProcessor(embedder=None)

        files_processed = 0
        chunks_created = 0
        files_skipped = 0
        file_results: list[dict[str, Any]] = []

        for i, rel_path in enumerate(py_files):
            abs_path = Path(repo_root) / rel_path
            if not abs_path.exists():
                continue

            result = processor.process_file(abs_path, repo_root)

            # Skip unchanged files
            if not force and result.symbol_hash:
                if existing_hashes.get(result.file_key) == result.symbol_hash:
                    files_skipped += 1
                    file_results.append({
                        "rel_path": result.rel_path,
                        "file_key": result.file_key,
                        "symbols": result.metadata.get("symbols", {}),
                    })
                    continue

            # Store file document
            file_doc: dict[str, Any] = {
                "_key": result.file_key,
                "rel_path": result.rel_path,
                "symbol_hash": result.symbol_hash,
                "num_chunks": len(result.chunks),
                "symbols": result.metadata.get("symbols", {}),
                "code_metrics": result.metadata.get("code_metrics", {}),
            }
            client.request(
                "POST",
                f"/_db/{db_name}/_api/document/{cols.files}",
                json=file_doc,
                params={"overwriteMode": "replace"},
            )

            # Remove existing chunks for this file to avoid orphans when chunk count changes
            client.query(
                "FOR c IN @@chunks FILTER c.file_key == @file_key REMOVE c IN @@chunks",
                bind_vars={"@chunks": cols.chunks, "file_key": result.file_key},
            )

            # Store chunks
            for chunk in result.chunks:
                ck = chunk_key(result.file_key, chunk.index)
                chunk_doc: dict[str, Any] = {
                    "_key": ck,
                    "file_key": result.file_key,
                    "rel_path": result.rel_path,
                    "chunk_index": chunk.index,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                }
                client.request(
                    "POST",
                    f"/_db/{db_name}/_api/document/{cols.chunks}",
                    json=chunk_doc,
                    params={"overwriteMode": "replace"},
                )
                chunks_created += 1

            files_processed += 1
            file_results.append({
                "rel_path": result.rel_path,
                "file_key": result.file_key,
                "symbols": result.metadata.get("symbols", {}),
            })

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i + 1}/{len(py_files)}] processed...",
                    file=sys.stderr,
                )

        # Resolve imports and create edges
        from core.database.import_resolver import ImportResolver

        known_rel_paths = {fr["rel_path"] for fr in file_results}
        resolver = ImportResolver(repo_root, known_rel_paths)
        edges = resolver.resolve_all(file_results)

        # Clear stale import edges for all resolved files before inserting new ones.
        # This ensures removed imports don't leave orphan edges.
        all_resolved_ids = [f"{cols.files}/{fr['file_key']}" for fr in file_results]
        if all_resolved_ids:
            try:
                client.query(
                    "FOR e IN @@edges FILTER e.type == 'imports' AND e._from IN @from_ids REMOVE e IN @@edges",
                    bind_vars={"@edges": cols.edges, "from_ids": all_resolved_ids},
                )
            except Exception:
                logger.warning("Failed to clear stale import edges")

        edges_created = 0
        for edge in edges:
            edge_doc = {
                "_key": f"{edge['_from_key']}__{edge['_to_key']}__imports",
                "_from": f"{cols.files}/{edge['_from_key']}",
                "_to": f"{cols.files}/{edge['_to_key']}",
                "type": "imports",
                "module": edge.get("module", ""),
            }
            client.request(
                "POST",
                f"/_db/{db_name}/_api/document/{cols.edges}",
                json=edge_doc,
                params={"overwriteMode": "replace"},
            )
            edges_created += 1

        return success_response(
            command="codebase.ingest",
            data={
                "files_processed": files_processed,
                "files_skipped": files_skipped,
                "files_total": len(py_files),
                "chunks_created": chunks_created,
                "edges_created": edges_created,
            },
            start_time=start_time,
        )

    except Exception as e:
        logger.exception("Codebase ingest failed")
        return error_response(
            command="codebase.ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def codebase_update(
    path: str,
    start_time: float,
    *,
    collections: CodebaseCollections | None = None,
) -> CLIResponse:
    """Incrementally update changed files.

    Compares symbol_hash of on-disk files vs stored codebase_files docs.
    Re-processes only changed files and re-materializes their edges.
    """
    # Update is just ingest without --force (which already skips unchanged)
    return codebase_ingest(path, start_time, force=False, collections=collections)


def codebase_stats(
    start_time: float,
    *,
    collections: CodebaseCollections | None = None,
) -> CLIResponse:
    """Show codebase collection statistics."""
    cols = collections or CODEBASE_COLLECTIONS

    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="codebase.stats",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        counts: dict[str, int] = {}
        for name in (cols.files, cols.chunks, cols.embeddings, cols.edges):
            resp = client.request("GET", f"/_db/{db_name}/_api/collection/{name}/count")
            if resp.get("error"):
                counts[name] = -1
            else:
                counts[name] = resp.get("count", 0)

        return success_response(
            command="codebase.stats",
            data={
                "files": counts.get(cols.files, 0),
                "chunks": counts.get(cols.chunks, 0),
                "embeddings": counts.get(cols.embeddings, 0),
                "edges": counts.get(cols.edges, 0),
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="codebase.stats",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def _get_existing_hashes(
    client: Any,
    db_name: str,
    cols: CodebaseCollections,
) -> dict[str, str]:
    """Get file_key → symbol_hash mapping from existing codebase_files."""
    try:
        results = client.query(
            "FOR f IN @@col RETURN { key: f._key, hash: f.symbol_hash }",
            bind_vars={"@col": cols.files},
        )
        return {r["key"]: r["hash"] for r in results if r.get("hash")}
    except Exception:
        return {}
