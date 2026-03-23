"""Rust edge resolver — materializes symbol nodes and edges from file-node data.

Reads the `rust_analyzer` attribute from codebase_files nodes and derives:
- codebase_symbols child documents (method/struct/trait/constant level)
- codebase_edges: defines, calls, implements, pyo3_exposes, ffi_exposes

The file node is the source of truth. This resolver is a "materialized view"
that projects fine-grained queryable nodes from it. Re-ingesting a file
re-extracts the attribute; running this resolver cascades updates to children.

Usage:
    resolver = RustEdgeResolver(file_nodes)
    symbols = resolver.build_symbol_nodes()
    edges = resolver.build_edges()
"""

from __future__ import annotations

import logging
from typing import Any

from core.database.keys import file_key

logger = logging.getLogger(__name__)


def symbol_key(file_rel_path: str, qualified_name: str) -> str:
    """Build an ArangoDB-safe key for a codebase_symbols document.

    Combines the file key with the qualified symbol name.

    Examples:
        >>> symbol_key("src/model.rs", "Model::new")
        'src_model_rs__Model__new'
    """
    fk = file_key(file_rel_path)
    # Replace :: with __ for ArangoDB key safety
    safe_name = qualified_name.replace("::", "__").replace(".", "_").replace("/", "_")
    return f"{fk}__{safe_name}"


class RustEdgeResolver:
    """Resolve rust-analyzer file data into symbol nodes and edges.

    Takes file nodes (or dicts with rel_path + rust_analyzer attribute)
    and produces two outputs:
    - Symbol documents for codebase_symbols collection
    - Edge documents for codebase_edges collection

    Args:
        file_nodes: List of dicts, each with "rel_path" (str) and
            "rust_analyzer" (dict from RustSymbolExtractor.extract_file).
    """

    def __init__(self, file_nodes: list[dict[str, Any]]) -> None:
        self._file_nodes = file_nodes
        # Index: qualified_name → list of (file_rel_path, symbol_key)
        # Lists handle name collisions (e.g., "add" in multiple files)
        self._symbol_index: dict[str, list[tuple[str, str]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build a lookup from qualified symbol names to their keys."""
        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            ra = node.get("rust_analyzer", {})
            for sym in ra.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if qname:
                    sk = symbol_key(rel_path, qname)
                    entry = (rel_path, sk)
                    self._symbol_index.setdefault(qname, []).append(entry)
                    # Also index by file-scoped name for cross-file call resolution
                    name = sym.get("name", "")
                    if name and name != qname:
                        file_qname = f"{rel_path}::{name}"
                        self._symbol_index.setdefault(file_qname, []).append(entry)

    def build_symbol_nodes(self) -> list[dict[str, Any]]:
        """Build codebase_symbols documents from file-node data.

        Each symbol becomes a document with:
        - _key: file-scoped unique key
        - name, qualified_name, kind, visibility, signature
        - file_path: parent file rel_path
        - line range, impl context
        - boundary flags (pyo3, ffi, unsafe)
        - derives, attributes

        Returns:
            List of symbol documents ready for batch insert.
        """
        symbols: list[dict[str, Any]] = []

        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            ra = node.get("rust_analyzer", {})

            for sym in ra.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if not qname:
                    continue

                sk = symbol_key(rel_path, qname)

                doc: dict[str, Any] = {
                    "_key": sk,
                    "name": sym.get("name", ""),
                    "qualified_name": qname,
                    "kind": sym.get("kind", ""),
                    "visibility": sym.get("visibility", "private"),
                    "signature": sym.get("signature", ""),
                    "file_path": rel_path,
                    "start_line": sym.get("start_line", 0),
                    "end_line": sym.get("end_line", 0),
                    "parent_symbol": sym.get("parent_symbol"),
                    "impl_trait": sym.get("impl_trait"),
                    "is_pyo3": sym.get("is_pyo3", False),
                    "is_ffi": sym.get("is_ffi", False),
                    "is_unsafe": sym.get("is_unsafe", False),
                    "derives": sym.get("derives", []),
                    "python_name": sym.get("python_name"),
                    "analyzed_at": ra.get("analyzed_at", ""),
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
        - defines: codebase_files/{file_key} → codebase_symbols/{symbol_key}
        - calls: codebase_symbols/{caller} → codebase_symbols/{callee}
        - implements: codebase_symbols/{method} → codebase_symbols/{trait}
        - pyo3_exposes: codebase_symbols/{rust_fn} → (marker edge, no target)
        - ffi_exposes: codebase_symbols/{rust_fn} → (marker edge, no target)

        Returns:
            List of edge documents ready for batch insert.
        """
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()  # (from, to, type) dedup

        for node in self._file_nodes:
            rel_path = node.get("rel_path", "")
            ra = node.get("rust_analyzer", {})
            fk = file_key(rel_path)

            for sym in ra.get("symbols", []):
                qname = sym.get("qualified_name", "")
                if not qname:
                    continue
                sk = symbol_key(rel_path, qname)

                # 1. defines: file → symbol
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

                # 2. calls: symbol → called symbol
                for call in sym.get("calls", []):
                    target_sk = self._resolve_call_target(call, rel_path)
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

                # 3. implements: method with impl_trait → trait symbol
                impl_trait = sym.get("impl_trait")
                if impl_trait and sym.get("kind") in ("method", "function"):
                    # Look for the trait in the index — prefer same-file match
                    trait_entries = self._symbol_index.get(impl_trait, [])
                    trait_sk = self._pick_best_match(trait_entries, rel_path)
                    if trait_sk:
                        edge_key = (f"codebase_symbols/{sk}", f"codebase_symbols/{trait_sk}", "implements")
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append({
                                "_from": f"codebase_symbols/{sk}",
                                "_to": f"codebase_symbols/{trait_sk}",
                                "type": "implements",
                                "implementor": qname,
                                "trait": impl_trait,
                            })

                # 4. pyo3_exposes: marker edge for PyO3 boundary
                if sym.get("is_pyo3"):
                    edge_key = (f"codebase_symbols/{sk}", "", "pyo3_exposes")
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append({
                            "_from": f"codebase_symbols/{sk}",
                            "_to": f"codebase_symbols/{sk}",  # self-edge as marker
                            "type": "pyo3_exposes",
                            "symbol_name": qname,
                            "python_name": sym.get("python_name", sym.get("name", "")),
                        })

                # 5. ffi_exposes: marker edge for FFI boundary
                if sym.get("is_ffi"):
                    edge_key = (f"codebase_symbols/{sk}", "", "ffi_exposes")
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append({
                            "_from": f"codebase_symbols/{sk}",
                            "_to": f"codebase_symbols/{sk}",  # self-edge as marker
                            "type": "ffi_exposes",
                            "symbol_name": qname,
                        })

        logger.info(
            "Built %d edges from %d files",
            len(edges),
            len(self._file_nodes),
        )
        return edges

    def _resolve_call_target(
        self, call: dict[str, Any], caller_file: str
    ) -> str | None:
        """Resolve a call target to a symbol key.

        Tries multiple resolution strategies:
        1. Exact qualified_name match in the symbol index
        2. File-scoped match (target_file::name)
        3. Same-file match (caller_file::name)
        4. Bare name with file preference

        Returns:
            The target symbol key, or None if unresolvable.
        """
        target_qname = call.get("qualified_name", "")
        target_name = call.get("name", "")
        target_file = call.get("file", "")

        # Strategy 1: exact qualified name — prefer target file match
        if target_qname and target_qname in self._symbol_index:
            prefer_file = target_file or caller_file
            sk = self._pick_best_match(self._symbol_index[target_qname], prefer_file)
            if sk:
                return sk

        # Strategy 2: file-scoped name
        if target_file and target_name:
            file_scoped = f"{target_file}::{target_name}"
            entries = self._symbol_index.get(file_scoped, [])
            sk = self._pick_best_match(entries, target_file)
            if sk:
                return sk

        # Strategy 3: same-file name
        if target_name:
            same_file = f"{caller_file}::{target_name}"
            entries = self._symbol_index.get(same_file, [])
            sk = self._pick_best_match(entries, caller_file)
            if sk:
                return sk

        # Strategy 4: bare name — prefer target file, fallback to first
        if target_name and target_name in self._symbol_index:
            prefer_file = target_file or caller_file
            sk = self._pick_best_match(self._symbol_index[target_name], prefer_file)
            if sk:
                return sk

        return None

    @staticmethod
    def _pick_best_match(
        entries: list[tuple[str, str]], prefer_file: str
    ) -> str | None:
        """Pick the best symbol key from a list of candidates.

        Prefers entries whose file matches prefer_file.
        Falls back to the first entry if no file match.

        Returns:
            The symbol key, or None if entries is empty.
        """
        if not entries:
            return None
        # Prefer same-file match
        for rel_path, sk in entries:
            if rel_path == prefer_file:
                return sk
        # Fallback: first entry
        return entries[0][1]
