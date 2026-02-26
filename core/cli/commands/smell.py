"""Smell compliance CLI commands.

Provides three commands forming the primitive layer for automated graph
integration sessions and the Hermes compliance panel:

    hades smell check <path>   - Static lint against NL graph forbidden patterns
    hades smell verify <path>  - Verify CS-XX/Eq-N refs exist in NL graph + have edges
    hades smell report <path>  - Full audit: check + verify + embedding probe
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)

# Regex to find CS-XX references in source files
_CS_REF_RE = re.compile(r"\bCS-(\d+)\b")
_EQ_REF_RE = re.compile(r"\bEq-(\d+)\b")

# Comment prefixes per file extension
_COMMENT_PREFIXES: dict[str, list[str]] = {
    ".py": ["#"],
    ".rs": ["//", "///", "//!"],
    ".cu": ["//"],
    ".cpp": ["//"],
    ".c": ["//"],
    ".ts": ["//"],
    ".js": ["//"],
    ".toml": ["#"],
    ".yaml": ["#"],
    ".yml": ["#"],
    ".sh": ["#"],
    ".bash": ["#"],
}

# Extensions to treat as text source files
_SOURCE_EXTENSIONS = set(_COMMENT_PREFIXES.keys()) | {".md", ".txt", ".json"}


def _is_comment_line(line: str, ext: str) -> bool:
    """Return True if this line is purely a comment (starts with comment prefix after whitespace)."""
    stripped = line.lstrip()
    for prefix in _COMMENT_PREFIXES.get(ext, ["#", "//"]):
        if stripped.startswith(prefix):
            return True
    return False


def _collect_source_files(path: str) -> list[Path]:
    """Collect all source files under path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return [p]
    files = []
    for ext in _SOURCE_EXTENSIONS:
        files.extend(p.rglob(f"*{ext}"))
    # Skip hidden dirs and common noise
    return [
        f for f in files
        if not any(part.startswith(".") for part in f.parts)
        and "__pycache__" not in f.parts
        and "Acheron" not in f.parts
    ]


def _load_static_smells(client: Any, db_name: str, smell_collection: str) -> list[dict[str, Any]]:
    """Load all smells that have forbidden_patterns from the NL graph."""
    results = client.query(
        """
        FOR doc IN @@col
            FILTER doc.forbidden_patterns != null AND LENGTH(doc.forbidden_patterns) > 0
            RETURN {
                _key: doc._key,
                smell_id: doc.smell_id,
                name: doc.name,
                forbidden_patterns: doc.forbidden_patterns
            }
        """,
        bind_vars={"@col": smell_collection},
    )
    return results


def smell_check(
    path: str,
    start_time: float,
    smell_collection: str = "nl_code_smells",
) -> CLIResponse:
    """Static lint — check source files against NL graph forbidden patterns.

    Loads forbidden_patterns from the configured database's smell collection
    and scans all source files under path for violations.

    Returns autonomous PASS/FAIL with violation details.
    """
    from core.cli.commands.database import _make_client

    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="smell.check",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        smells = _load_static_smells(client, db_name, smell_collection)
    except Exception as e:
        return error_response(
            command="smell.check",
            code=ErrorCode.DATABASE_ERROR,
            message=f"Failed to load smell patterns: {e}",
            start_time=start_time,
        )
    finally:
        client.close()

    if not smells:
        return error_response(
            command="smell.check",
            code=ErrorCode.VALIDATION_ERROR,
            message=f"No smells with forbidden_patterns found in {smell_collection}",
            start_time=start_time,
        )

    files = _collect_source_files(path)
    if not files:
        return error_response(
            command="smell.check",
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"No source files found at {path}",
            start_time=start_time,
        )

    violations: list[dict[str, Any]] = []

    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for line_no, line in enumerate(lines, start=1):
            # Skip pure comment lines for word-level checks (not pattern token checks)
            is_comment = _is_comment_line(line, ext)

            for smell in smells:
                for pattern in smell["forbidden_patterns"]:
                    if pattern in line:
                        # CS-13 specifically excludes comment context
                        if is_comment and smell.get("smell_id") == 13:
                            continue
                        violations.append({
                            "smell_key": smell["_key"],
                            "smell_id": smell.get("smell_id"),
                            "smell_name": smell["name"],
                            "file": str(file_path),
                            "line": line_no,
                            "pattern": pattern,
                            "content": line.rstrip(),
                        })

    passed = len(violations) == 0
    return success_response(
        command="smell.check",
        data={
            "passed": passed,
            "violations": violations,
            "files_checked": len(files),
            "smells_loaded": len(smells),
            "violation_count": len(violations),
        },
        start_time=start_time,
    )


def _extract_cs_refs_from_file(file_path: Path) -> dict[str, list[int]]:
    """Extract all CS-XX references from comment lines in a source file.

    Returns mapping: "CS-32" -> [line_numbers]
    """
    ext = file_path.suffix.lower()
    refs: dict[str, list[int]] = {}
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return refs

    for line_no, line in enumerate(lines, start=1):
        if not _is_comment_line(line, ext):
            continue
        for match in _CS_REF_RE.finditer(line):
            cs_id = f"CS-{match.group(1)}"
            refs.setdefault(cs_id, []).append(line_no)

    return refs


def smell_verify(
    path: str,
    start_time: float,
    smell_collection: str = "nl_code_smells",
    compliance_collection: str = "nl_smell_compliance_edges",
    metadata_collection: str = "arxiv_metadata",
) -> CLIResponse:
    """Verify CS-XX references from source file(s) exist in the NL graph.

    For each CS-XX reference found in comments:
    1. Checks the smell node exists in the graph
    2. Checks a compliance edge exists in nl_smell_compliance_edges

    Returns: verified_refs, missing_from_graph, unlinked_claims
    """
    from core.cli.commands.database import _make_client

    files = _collect_source_files(path)
    if not files:
        return error_response(
            command="smell.verify",
            code=ErrorCode.FILE_NOT_FOUND,
            message=f"No source files found at {path}",
            start_time=start_time,
        )

    # Extract all CS-XX references from all files
    all_refs: dict[str, dict[str, list[int]]] = {}  # file -> {cs_id -> [lines]}
    for file_path in files:
        refs = _extract_cs_refs_from_file(file_path)
        if refs:
            all_refs[str(file_path)] = refs

    if not all_refs:
        return success_response(
            command="smell.verify",
            data={
                "files_scanned": len(files),
                "refs_found": 0,
                "verified_refs": [],
                "missing_from_graph": [],
                "unlinked_claims": [],
            },
            start_time=start_time,
        )

    # Connect to DB and verify
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="smell.verify",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        verified_refs: list[dict[str, Any]] = []
        missing_from_graph: list[dict[str, Any]] = []
        unlinked_claims: list[dict[str, Any]] = []

        # Collect all unique CS-IDs across files
        all_cs_ids: set[str] = set()
        for refs in all_refs.values():
            all_cs_ids.update(refs.keys())

        # Batch lookup: find all smell nodes matching CS-XX ids
        smell_map: dict[str, dict[str, Any]] = {}
        for cs_id in all_cs_ids:
            num = cs_id[3:]  # strip "CS-"
            # Match by smell_id (numeric) or by name prefix "CS-<num>:"
            results = client.query(
                """
                FOR doc IN @@col
                    FILTER doc.smell_id == TO_NUMBER(@num) OR STARTS_WITH(doc.name, @prefix)
                    RETURN {_key: doc._key, _id: doc._id, smell_id: doc.smell_id, name: doc.name}
                """,
                bind_vars={"@col": smell_collection, "num": num, "prefix": f"CS-{num}:"},
            )
            if results:
                smell_map[cs_id] = results[0]

        # For each file, check compliance edges
        for file_str, refs in all_refs.items():
            file_path = Path(file_str)
            file_stem = file_path.stem  # e.g. "conductor-rs"

            # Find arxiv_metadata document for this file (if ingested).
            # Try several key patterns: exact stem, stem-ext (e.g. "conductor-rs"),
            # and stem with dashes replacing underscores.
            doc_id: str | None = None
            ext_label = file_path.suffix.lstrip(".").lower()  # "rs", "py", "cu", etc.
            candidate_keys = [
                file_stem,                          # "conductor"
                f"{file_stem}-{ext_label}",         # "conductor-rs"
                file_stem.replace("_", "-"),        # underscore → dash
                f"{file_stem.replace('_', '-')}-{ext_label}",
            ]
            try:
                docs = client.query(
                    "FOR d IN @@col FILTER d._key IN @keys RETURN d._id",
                    bind_vars={"@col": metadata_collection, "keys": candidate_keys},
                )
                if docs:
                    doc_id = docs[0]
            except Exception:
                pass  # File not ingested — compliance edges won't exist

            for cs_id, line_numbers in refs.items():
                entry = {
                    "cs_id": cs_id,
                    "file": file_str,
                    "lines": line_numbers,
                }

                smell_node = smell_map.get(cs_id)
                if not smell_node:
                    missing_from_graph.append({**entry, "reason": "smell node not found in graph"})
                    continue

                entry["smell_key"] = smell_node["_key"]
                entry["smell_name"] = smell_node["name"]

                # Check for compliance edge
                edge_exists = False
                edge_data: dict[str, Any] | None = None
                if doc_id:
                    edges = client.query(
                        """
                        FOR e IN @@col
                            FILTER e._from == @from AND e._to == @to
                            RETURN e
                        """,
                        bind_vars={
                            "@col": compliance_collection,
                            "from": doc_id,
                            "to": smell_node["_id"],
                        },
                    )
                    if edges:
                        edge_exists = True
                        edge_data = {
                            "enforcement_type": edges[0].get("enforcement_type"),
                            "claim_summary": edges[0].get("claim_summary"),
                            "claiming_methods": edges[0].get("claiming_methods"),
                        }

                if edge_exists:
                    verified_refs.append({**entry, "edge": edge_data})
                else:
                    reason = "no compliance edge" if doc_id else "file not ingested (no doc_id)"
                    unlinked_claims.append({**entry, "reason": reason})

        return success_response(
            command="smell.verify",
            data={
                "files_scanned": len(files),
                "refs_found": sum(len(r) for r in all_refs.values()),
                "verified_refs": verified_refs,
                "missing_from_graph": missing_from_graph,
                "unlinked_claims": unlinked_claims,
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="smell.verify",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def smell_report(
    path: str,
    start_time: float,
    smell_collection: str = "nl_code_smells",
    compliance_collection: str = "nl_smell_compliance_edges",
    pr_diff: str | None = None,
) -> CLIResponse:
    """Full compliance audit: static check + ref verification + embedding probe.

    Combines smell_check and smell_verify, then probes embedding similarity
    between the source file(s) and each claimed smell node.

    Args:
        path: File or directory path to audit
        start_time: Start time for duration calculation
        smell_collection: NL smell collection name
        compliance_collection: Compliance edges collection
        pr_diff: Optional path to unified diff file (limits to changed files)
    """
    # If pr_diff is specified, filter to changed files only
    changed_files: set[str] | None = None

    if pr_diff:
        diff_path = Path(pr_diff)
        if not diff_path.exists():
            return error_response(
                command="smell.report",
                code=ErrorCode.FILE_NOT_FOUND,
                message=f"PR diff file not found: {pr_diff}",
                start_time=start_time,
            )
        changed_files = _parse_diff_files(diff_path)
        if not changed_files:
            return success_response(
                command="smell.report",
                data={
                    "path": path,
                    "pr_diff": pr_diff,
                    "files_in_diff": 0,
                    "message": "No source files changed in diff",
                    "passed": True,
                    "static_check": {"passed": True, "violations": [], "files_checked": 0},
                    "ref_verification": {"verified_refs": [], "missing_from_graph": [], "unlinked_claims": []},
                    "embedding_probe": [],
                },
                start_time=start_time,
            )

    # Run static check
    check_result = smell_check(path, start_time, smell_collection=smell_collection)
    if not check_result.success:
        return error_response(
            command="smell.report",
            code=ErrorCode.PROCESSING_FAILED,
            message=f"Static check failed: {check_result.error}",
            start_time=start_time,
        )
    check_data = check_result.data or {}

    # Filter violations to changed files if pr_diff provided
    if changed_files is not None:
        violations = [
            v for v in check_data.get("violations", [])
            if Path(v["file"]).name in changed_files or str(v["file"]) in changed_files
        ]
        check_data = {**check_data, "violations": violations, "violation_count": len(violations)}

    # Run ref verification
    verify_result = smell_verify(
        path, start_time,
        smell_collection=smell_collection,
        compliance_collection=compliance_collection,
    )
    if not verify_result.success:
        return error_response(
            command="smell.report",
            code=ErrorCode.PROCESSING_FAILED,
            message=f"Ref verification failed: {verify_result.error}",
            start_time=start_time,
        )
    verify_data = verify_result.data or {}

    # Embedding probe: for each verified compliance edge, compute cosine similarity
    probe_results: list[dict[str, Any]] = []
    verified_refs = verify_data.get("verified_refs", [])

    if verified_refs:
        try:
            from core.services.embedder_client import EmbedderClient

            embedder = EmbedderClient()

            for ref in verified_refs:
                file_str = ref.get("file", "")
                cs_id = ref.get("cs_id", "")
                smell_name = ref.get("smell_name", "")

                try:
                    file_text = Path(file_str).read_text(encoding="utf-8", errors="replace")
                    # Truncate to ~8000 chars to avoid memory issues with large files
                    if len(file_text) > 8000:
                        file_text = file_text[:8000]

                    # Embed file text (as passage) and smell name (as query)
                    file_vec = embedder.embed_texts([file_text], task="retrieval.passage")[0].tolist()
                    smell_vec = embedder.embed_texts([smell_name], task="retrieval.query")[0].tolist()

                    similarity = _cosine_similarity(file_vec, smell_vec)
                    probe_results.append({
                        "cs_id": cs_id,
                        "smell_name": smell_name,
                        "file": file_str,
                        "cosine_similarity": round(similarity, 4),
                        "pass": similarity >= 0.5,
                    })

                except Exception as probe_err:
                    probe_results.append({
                        "cs_id": cs_id,
                        "smell_name": smell_name,
                        "file": file_str,
                        "error": str(probe_err),
                        "pass": None,
                    })

        except (ImportError, Exception) as e:
            # Embedding service not available — report without probe
            print(f"[smell.report] Embedding probe skipped: {e}", file=sys.stderr)
            probe_results = [
                {
                    "cs_id": ref.get("cs_id"),
                    "smell_name": ref.get("smell_name"),
                    "file": ref.get("file"),
                    "error": "embedding service unavailable",
                    "pass": None,
                }
                for ref in verified_refs
            ]

    # Overall pass: only STATIC violations are blockers; unlinked_claims are informational
    static_passed = check_data.get("passed", True)
    has_unlinked = len(verify_data.get("unlinked_claims", [])) > 0

    return success_response(
        command="smell.report",
        data={
            "path": path,
            "pr_diff": pr_diff,
            "passed": static_passed,
            "has_unlinked_claims": has_unlinked,
            "static_check": {
                "passed": static_passed,
                "violations": check_data.get("violations", []),
                "violation_count": check_data.get("violation_count", 0),
                "files_checked": check_data.get("files_checked", 0),
            },
            "ref_verification": {
                "refs_found": verify_data.get("refs_found", 0),
                "verified_refs": verify_data.get("verified_refs", []),
                "missing_from_graph": verify_data.get("missing_from_graph", []),
                "unlinked_claims": verify_data.get("unlinked_claims", []),
            },
            "embedding_probe": probe_results,
        },
        start_time=start_time,
    )


def _parse_diff_files(diff_path: Path) -> set[str]:
    """Extract changed filenames from a unified diff."""
    changed: list[str] = []
    text = diff_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        # Unified diff: "+++ b/path/to/file.rs" or "--- a/path/to/file.rs"
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            file_str = line[6:].strip()
            if file_str != "/dev/null":
                changed.append(file_str)
    return set(changed)
