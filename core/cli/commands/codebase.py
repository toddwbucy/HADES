"""Codebase knowledge graph CLI commands.

Provides ingest, update, and stats commands for indexing the
repository's Python and Rust files into the codebase knowledge graph.
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
_DEFAULT_EXCLUDE = {"Acheron", "__pycache__", ".git", ".tox", ".mypy_cache", ".ruff_cache", "target"}


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


def _git_rust_files(repo_root: str, exclude: set[str] | None = None) -> list[str]:
    """List tracked + untracked (but not ignored) Rust files via git."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.rs"],
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
        parts = Path(line).parts
        if any(p in skip for p in parts):
            continue
        files.append(line)

    return files


def _find_crate_roots(repo_root: str, rs_files: list[str]) -> dict[str, list[str]]:
    """Group .rs files by their nearest Cargo.toml crate root.

    Returns:
        Dict mapping crate root (relative to repo_root) to list of .rs file
        paths (relative to repo_root).
    """
    crate_map: dict[str, list[str]] = {}

    for rs_file in rs_files:
        # Walk up from the .rs file to find the nearest Cargo.toml
        rs_path = Path(repo_root) / rs_file
        candidate = rs_path.parent
        crate_root = None
        while candidate != Path(repo_root).parent:
            if (candidate / "Cargo.toml").exists():
                crate_root = str(candidate)
                break
            candidate = candidate.parent

        if crate_root is None:
            continue  # No Cargo.toml found — skip this file

        crate_map.setdefault(crate_root, []).append(rs_file)

    return crate_map


def _analyze_rust_crate(
    crate_root: str,
    rs_files: list[str],
    repo_root: str,
    client: Any,
    db_name: str,
    cols: CodebaseCollections,
) -> dict[str, int]:
    """Analyze Rust files in a single crate and store results.

    1. Start rust-analyzer session for the crate
    2. Extract symbols from each .rs file
    3. Store rust_analyzer attribute on file nodes
    4. Materialize symbol nodes and edges via RustEdgeResolver

    Returns:
        Dict with counts: symbols_created, edges_created
    """
    import hashlib
    import shutil

    from core.analyzers.rust_analyzer_client import RustAnalyzerSession
    from core.analyzers.rust_edge_resolver import RustEdgeResolver
    from core.analyzers.rust_symbol_extractor import RustSymbolExtractor
    from core.database.keys import file_key

    if not shutil.which("rust-analyzer"):
        logger.warning("rust-analyzer not installed — skipping Rust analysis")
        return {"rust_symbols_created": 0, "rust_edges_created": 0}

    crate_name = Path(crate_root).name
    print(f"  rust-analyzer: analyzing crate '{crate_name}'...", file=sys.stderr)

    try:
        with RustAnalyzerSession(crate_root, timeout=120) as session:
            extractor = RustSymbolExtractor(session, include_calls=True, include_incoming=False)

            file_nodes: list[dict[str, Any]] = []

            # Track all files we attempt (for stale cleanup even on failure)
            all_attempted_paths: list[str] = []

            for rs_file in rs_files:
                abs_path = Path(repo_root) / rs_file
                crate_rel = str(abs_path.relative_to(crate_root))
                all_attempted_paths.append(rs_file)

                try:
                    ra_data = extractor.extract_file(crate_rel)
                except Exception:
                    logger.warning("Failed to extract symbols from %s", rs_file)
                    # Clear stale data for this file on failure
                    fk = file_key(rs_file)
                    try:
                        client.request(
                            "PATCH",
                            f"/_db/{db_name}/_api/document/{cols.files}/{fk}",
                            json={"rust_analyzer": None},
                        )
                    except Exception:
                        pass
                    continue

                # Update the file node with rust_analyzer attribute
                fk = file_key(rs_file)
                client.request(
                    "PATCH",
                    f"/_db/{db_name}/_api/document/{cols.files}/{fk}",
                    json={"rust_analyzer": ra_data},
                )

                file_nodes.append({
                    "rel_path": rs_file,
                    "rust_analyzer": ra_data,
                })

                logger.debug(
                    "Analyzed %s: %d symbols",
                    rs_file,
                    len(ra_data.get("symbols", [])),
                )

            # Clear stale symbols and edges for ALL attempted files BEFORE inserting new ones.
            # This uses the OLD symbol IDs so renamed/removed symbols get cleaned up.
            rust_edge_types = ["defines", "calls", "implements", "pyo3_exposes", "ffi_exposes"]

            # Check whether symbol/edge collections exist yet (they may not on first run)
            col_resp = client.request("GET", f"/_db/{db_name}/_api/collection")
            existing_cols = {c["name"] for c in col_resp.get("result", [])}
            has_symbols = cols.symbols in existing_cols
            has_edges = cols.edges in existing_cols

            if has_symbols:
                # Get old symbol keys before deletion
                stale_keys = client.query(
                    "FOR s IN @@symbols FILTER s.file_path IN @paths RETURN s._key",
                    bind_vars={"@symbols": cols.symbols, "paths": all_attempted_paths},
                )
                stale_symbol_ids = [f"{cols.symbols}/{k}" for k in stale_keys]
                file_ids = [f"{cols.files}/{file_key(p)}" for p in all_attempted_paths]
                stale_ids = file_ids + stale_symbol_ids

                if has_edges and stale_ids:
                    client.query(
                        "FOR e IN @@edges FILTER e.type IN @types AND (e._from IN @ids OR e._to IN @ids) REMOVE e IN @@edges",
                        bind_vars={"@edges": cols.edges, "types": rust_edge_types, "ids": stale_ids},
                    )

                # Now remove old symbols
                client.query(
                    "FOR s IN @@symbols FILTER s.file_path IN @paths REMOVE s IN @@symbols",
                    bind_vars={"@symbols": cols.symbols, "paths": all_attempted_paths},
                )

            if not file_nodes:
                return {"rust_symbols_created": 0, "rust_edges_created": 0}

            # Materialize symbol nodes and edges
            resolver = RustEdgeResolver(file_nodes)
            symbol_docs = resolver.build_symbol_nodes()
            edge_docs = resolver.build_edges()

            # Insert symbol nodes
            symbols_created = 0
            for doc in symbol_docs:
                client.request(
                    "POST",
                    f"/_db/{db_name}/_api/document/{cols.symbols}",
                    json=doc,
                    params={"overwriteMode": "replace"},
                )
                symbols_created += 1

            # Insert edges
            edges_created = 0
            for edge in edge_docs:
                identity = f"{edge['_from']}|{edge['_to']}|{edge['type']}"
                edge["_key"] = hashlib.sha256(identity.encode()).hexdigest()
                client.request(
                    "POST",
                    f"/_db/{db_name}/_api/document/{cols.edges}",
                    json=edge,
                    params={"overwriteMode": "replace"},
                )
                edges_created += 1

            print(
                f"  rust-analyzer: {symbols_created} symbols, {edges_created} edges from {len(file_nodes)} files",
                file=sys.stderr,
            )

            return {
                "rust_symbols_created": symbols_created,
                "rust_edges_created": edges_created,
            }

    except Exception:
        logger.exception("Rust analysis failed for crate %s", crate_root)
        return {"rust_symbols_created": 0, "rust_edges_created": 0}


def codebase_ingest(
    path: str,
    start_time: float,
    *,
    force: bool = False,
    exclude: list[str] | None = None,
    collections: CodebaseCollections | None = None,
) -> CLIResponse:
    """Ingest repository Python and Rust files into the codebase knowledge graph.

    1. git ls-files to get Python + Rust file lists
    2. Extract + AST-chunk each Python file via CodeProcessor
    3. Store file docs, chunks, embeddings
    4. Resolve Python imports → create edges
    5. If Rust files exist: run rust-analyzer → store symbols + edges
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

        # Get file lists
        exclude_set = _DEFAULT_EXCLUDE | set(exclude or [])
        py_files = _git_python_files(repo_root, exclude_set)
        rs_files = _git_rust_files(repo_root, exclude_set)
        if not py_files and not rs_files:
            return error_response(
                command="codebase.ingest",
                code=ErrorCode.FILE_NOT_FOUND,
                message="No Python or Rust files found in repository",
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
                logger.exception("Failed to clear stale import edges")
                raise

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

        # ── Rust analysis ──────────────────────────────────────────
        rust_stats: dict[str, int] = {"rust_symbols_created": 0, "rust_edges_created": 0}

        if rs_files:
            import hashlib

            from core.database.keys import file_key as _file_key

            changed_rs_files: list[str] = []
            rs_hashes: dict[str, str] = {}  # rel_path → content_hash (written after success)
            rs_skipped = 0

            for rs_file in rs_files:
                abs_path = Path(repo_root) / rs_file
                if not abs_path.exists():
                    continue

                content_hash = hashlib.sha256(abs_path.read_bytes()).hexdigest()
                fk = _file_key(rs_file)

                # Skip unchanged files when not forcing
                if not force and existing_hashes.get(fk) == content_hash:
                    rs_skipped += 1
                    continue

                # Store file doc WITHOUT symbol_hash — hash is written after analysis succeeds
                rs_doc: dict[str, Any] = {
                    "_key": fk,
                    "rel_path": rs_file,
                    "language": "rust",
                }
                client.request(
                    "POST",
                    f"/_db/{db_name}/_api/document/{cols.files}",
                    json=rs_doc,
                    params={"overwriteMode": "update"},
                )
                changed_rs_files.append(rs_file)
                rs_hashes[rs_file] = content_hash

            files_skipped += rs_skipped

            # Group changed .rs files by crate. When any file in a crate changed,
            # re-analyze the ENTIRE crate (Rust analysis is crate-scoped — a change
            # in one file can alter calls/implements edges for siblings).
            if changed_rs_files:
                # Build full crate map for ALL .rs files so we know every file per crate
                full_crate_map = _find_crate_roots(repo_root, rs_files)
                changed_crate_map = _find_crate_roots(repo_root, changed_rs_files)
                analyzed_files: set[str] = set()

                for crate_root_path in changed_crate_map:
                    # Analyze all files in the crate, not just the changed ones
                    all_crate_files = full_crate_map.get(crate_root_path, changed_crate_map[crate_root_path])
                    crate_stats = _analyze_rust_crate(
                        crate_root_path, all_crate_files, repo_root, client, db_name, cols,
                    )
                    rust_stats["rust_symbols_created"] += crate_stats["rust_symbols_created"]
                    rust_stats["rust_edges_created"] += crate_stats["rust_edges_created"]
                    # Only mark files as up-to-date if analysis produced results
                    if crate_stats["rust_symbols_created"] > 0 or crate_stats["rust_edges_created"] > 0:
                        analyzed_files.update(all_crate_files)

                # Write content hash only for files whose crate was successfully analyzed
                for rs_file in analyzed_files:
                    content_hash = rs_hashes.get(rs_file)
                    if content_hash:
                        fk = _file_key(rs_file)
                        client.request(
                            "PATCH",
                            f"/_db/{db_name}/_api/document/{cols.files}/{fk}",
                            json={"symbol_hash": content_hash},
                        )

        return success_response(
            command="codebase.ingest",
            data={
                "files_processed": files_processed,
                "files_skipped": files_skipped,
                "files_total": len(py_files) + len(rs_files),
                "chunks_created": chunks_created,
                "edges_created": edges_created,
                "rust_files": len(rs_files),
                **rust_stats,
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
        for name in (cols.files, cols.symbols, cols.chunks, cols.embeddings, cols.edges):
            resp = client.request("GET", f"/_db/{db_name}/_api/collection/{name}/count")
            if resp.get("error"):
                counts[name] = -1
            else:
                counts[name] = resp.get("count", 0)

        return success_response(
            command="codebase.stats",
            data={
                "files": counts.get(cols.files, 0),
                "symbols": counts.get(cols.symbols, 0),
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
