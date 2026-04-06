"""Python edge resolver — materializes symbol nodes and edges from file-node data.

Reads the ``python_ast`` attribute from codebase_files nodes and derives:
- codebase_symbols child documents (function/class/method/constant level)
- codebase_edges: defines, calls

This is the Python counterpart of ``RustEdgeResolver``.  The file node is
the source of truth; this resolver is a "materialized view" that projects
fine-grained queryable nodes from it.

Usage:
    resolver = PythonEdgeResolver(file_nodes)
    symbols = resolver.build_symbol_nodes()
    edges = resolver.build_edges()
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.database.keys import file_key

logger = logging.getLogger(__name__)


def symbol_key(file_rel_path: str, qualified_name: str) -> str:
    """Build an ArangoDB-safe key for a Python codebase_symbols document.

    Mirrors ``rust_edge_resolver.symbol_key`` in structure so Python and
    Rust symbols share the same key-building convention.

    Examples:
        >>> symbol_key("core/database/pool.py", "Pool.acquire")
        'core_database_pool_py__Pool_acquire'
        >>> symbol_key("setup.py", "main")
        'setup_py__main'
    """
    fk = file_key(file_rel_path)
    # Replace '.' with underscore (Python scope separator)
    safe_name = qualified_name.replace(".", "_")
    # Replace all non-alphanumeric/underscore chars with single underscore
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", safe_name)
    # Collapse 3+ underscores to 2
    safe_name = re.sub(r"_{3,}", "__", safe_name).strip("_")
    return f"{fk}__{safe_name}"


class PythonEdgeResolver:
    """Resolve Python AST file data into symbol nodes and edges.

    Takes file nodes (or dicts with ``rel_path`` + ``python_ast`` attribute)
    and produces two outputs:
    - Symbol documents for ``codebase_symbols`` collection
    - Edge documents for ``codebase_edges`` collection

    Args:
        file_nodes: List of dicts, each with ``"rel_path"`` (str) and
            ``"python_ast"`` (dict from ``PythonAstExtractor.extract_file``).
    """

    def __init__(self, file_nodes: list[dict[str, Any]]) -> None:
        self._file_nodes = file_nodes
        # Index: qualified_name → list of (file_rel_path, symbol_key)
        self._symbol_index: dict[str, list[tuple[str, str]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build a lookup from qualified symbol names to their keys."""
        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            pa = node.get("python_ast", {})
            for sym in pa.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if qname:
                    sk = symbol_key(rel_path, qname)
                    entry = (rel_path, sk)
                    self._symbol_index.setdefault(qname, []).append(entry)

    def build_symbol_nodes(self) -> list[dict[str, Any]]:
        """Build codebase_symbols documents from file-node data.

        Each symbol becomes a document with:
        - _key: file-scoped unique key
        - name, qualified_name, kind, visibility, signature
        - file_path: parent file rel_path
        - line range, parent context
        - decorators, bases

        Returns:
            List of symbol documents ready for batch insert.
        """
        symbols: list[dict[str, Any]] = []

        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            pa = node.get("python_ast", {})

            for sym in pa.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if not qname:
                    continue

                sk = symbol_key(rel_path, qname)

                doc: dict[str, Any] = {
                    "_key": sk,
                    "name": sym.get("name", ""),
                    "qualified_name": qname,
                    "kind": sym.get("kind", ""),
                    "visibility": sym.get("visibility", "public"),
                    "signature": sym.get("signature", ""),
                    "file_path": rel_path,
                    "language": "python",
                    "start_line": sym.get("start_line", 0),
                    "end_line": sym.get("end_line", 0),
                    "parent_symbol": sym.get("parent_symbol"),
                    "decorators": sym.get("decorators", []),
                    "bases": sym.get("bases", []),
                    "analyzed_at": pa.get("analyzed_at", ""),
                }
                symbols.append(doc)

        logger.info(
            "Built %d symbol nodes from %d files",
            len(symbols),
            len(self._file_nodes),
        )
        return symbols

    def build_edges(self) -> list[dict[str, Any]]:
        """Build codebase_edges from file-node data.

        Edge types:
        - defines: codebase_files/{file_key} -> codebase_symbols/{symbol_key}
        - calls: codebase_symbols/{caller} -> codebase_symbols/{callee}

        Returns:
            List of edge documents ready for batch insert.
        """
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            pa = node.get("python_ast", {})
            fk = file_key(rel_path)

            for sym in pa.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if not qname:
                    continue
                sk = symbol_key(rel_path, qname)

                # 1. defines: file -> symbol
                edge_key = (f"codebase_files/{fk}", f"codebase_symbols/{sk}", "defines")
                if edge_key not in seen:
                    seen.add(edge_key)
                    edges.append({
                        "_from": f"codebase_files/{fk}",
                        "_to": f"codebase_symbols/{sk}",
                        "type": "defines",
                        "file_path": rel_path,
                        "symbol_name": qname,
                    })

                # 2. calls: symbol -> called symbol
                for call in sym.get("calls", []):
                    target_sk = self._resolve_call_target(call, rel_path, qname)
                    if target_sk:
                        edge_key = (f"codebase_symbols/{sk}", f"codebase_symbols/{target_sk}", "calls")
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append({
                                "_from": f"codebase_symbols/{sk}",
                                "_to": f"codebase_symbols/{target_sk}",
                                "type": "calls",
                                "caller": qname,
                                "callee": call.get("qualified_name", call.get("name", "")),
                            })

        logger.info(
            "Built %d edges from %d files",
            len(edges),
            len(self._file_nodes),
        )
        return edges

    def _resolve_call_target(
        self, call: dict[str, str], caller_file: str, caller_qname: str,
    ) -> str | None:
        """Resolve a call target to a symbol key.

        Resolution strategies:
        1. Exact qualified_name match
        2. ``self.method`` → look up ``ClassName.method`` in parent class
        3. Bare name match (prefer same file)
        """
        target_qname = call.get("qualified_name", "")
        target_name = call.get("name", "")

        # Strategy 1: exact qualified name
        if target_qname and target_qname in self._symbol_index:
            sk = self._pick_best_match(self._symbol_index[target_qname], caller_file)
            if sk:
                return sk

        # Strategy 2: self.method → ClassName.method
        if target_qname.startswith("self.") and "." in caller_qname:
            # caller is ClassName.method_name → parent class is ClassName
            class_name = caller_qname.rsplit(".", 1)[0]
            resolved = f"{class_name}.{target_name}"
            if resolved in self._symbol_index:
                sk = self._pick_best_match(self._symbol_index[resolved], caller_file)
                if sk:
                    return sk

        # Strategy 3: bare name — prefer same file
        if target_name and target_name in self._symbol_index:
            sk = self._pick_best_match(self._symbol_index[target_name], caller_file)
            if sk:
                return sk

        return None

    @staticmethod
    def _pick_best_match(
        entries: list[tuple[str, str]], prefer_file: str,
    ) -> str | None:
        """Pick the best match from index entries, preferring same-file."""
        if not entries:
            return None
        for file_path, sk in entries:
            if file_path == prefer_file:
                return sk
        # Fallback: first match
        return entries[0][1]
