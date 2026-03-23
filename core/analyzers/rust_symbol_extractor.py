"""Rust symbol extraction via rust-analyzer LSP.

Extracts rich symbol data from Rust files and structures it for storage
as the `rust_analyzer` attribute on codebase_files nodes. This is the
canonical data from which codebase_symbols child nodes and edges are derived.

The extractor uses four LSP requests:
- documentSymbol → hierarchical symbol tree
- hover → resolved type signatures
- callHierarchy/outgoingCalls → cross-file call graph
- callHierarchy/incomingCalls → reverse call graph

Usage:
    with RustAnalyzerSession(crate_root) as session:
        extractor = RustSymbolExtractor(session)
        file_data = extractor.extract_file("src/lib.rs")
        # file_data is the dict to store as file_node["rust_analyzer"]
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.analyzers.rust_analyzer_client import RustAnalyzerSession

logger = logging.getLogger(__name__)

# LSP SymbolKind enum values → human-readable names
_SYMBOL_KIND_MAP: dict[int, str] = {
    1: "file", 2: "module", 3: "namespace", 4: "package",
    5: "class", 6: "method", 7: "property", 8: "field",
    9: "constructor", 10: "enum", 11: "interface", 12: "function",
    13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum_member", 23: "struct", 24: "event",
    25: "operator", 26: "type_parameter",
}

# PyO3 attribute patterns
_PYO3_PATTERNS = re.compile(
    r"#\[(pyclass|pymethods|pyfunction|pymodule|pyo3\(|pyproto)]"
)

# FFI patterns
_FFI_PATTERNS = re.compile(
    r'(extern\s+"C"|#\[no_mangle\]|#\[export_name)'
)

# Visibility patterns in detail/signature strings
_VISIBILITY_PATTERN = re.compile(
    r"^(pub(\s*\([^)]*\))?)\s+"
)


class RustSymbolExtractor:
    """Extract structured symbol data from Rust files via rust-analyzer.

    Produces a dict suitable for storing as the `rust_analyzer` attribute
    on a codebase_files document.

    Args:
        session: An initialized RustAnalyzerSession.
        include_calls: Whether to extract call hierarchy (slower but richer).
        include_incoming: Whether to extract incoming calls (even slower).
    """

    def __init__(
        self,
        session: RustAnalyzerSession,
        include_calls: bool = True,
        include_incoming: bool = False,
    ) -> None:
        self._session = session
        self._include_calls = include_calls
        self._include_incoming = include_incoming

    def extract_file(
        self,
        file_path: str | Path,
        file_content: str | None = None,
    ) -> dict[str, Any]:
        """Extract all symbol data for a single Rust file.

        Args:
            file_path: Path to the .rs file (relative to crate root or absolute).
            file_content: Optional file content for attribute scanning.
                If not provided, reads from disk.

        Returns:
            Dict suitable for file_node["rust_analyzer"]:
            {
                "symbols": [...],
                "impl_blocks": [...],
                "pyo3_exports": [...],
                "ffi_boundaries": [...],
                "analyzed_at": "..."
            }
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self._session.crate_root / path

        # Read content for attribute scanning
        if file_content is None:
            try:
                file_content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                file_content = ""

        # Get document symbols from rust-analyzer
        raw_symbols = self._session.document_symbols(file_path)

        # Flatten and enrich symbols
        symbols = self._process_symbols(raw_symbols, file_path, file_content)

        # Group impl blocks
        impl_blocks = self._extract_impl_blocks(symbols)

        # Identify PyO3 exports and FFI boundaries
        pyo3_exports = [s["name"] for s in symbols if s.get("is_pyo3")]
        ffi_boundaries = [s["name"] for s in symbols if s.get("is_ffi")]

        return {
            "symbols": symbols,
            "impl_blocks": impl_blocks,
            "pyo3_exports": pyo3_exports,
            "ffi_boundaries": ffi_boundaries,
            "analyzed_at": datetime.now(UTC).isoformat(),
        }

    def extract_crate(
        self,
        rs_files: list[str | Path],
    ) -> dict[str, dict[str, Any]]:
        """Extract symbol data for all .rs files in a crate.

        Args:
            rs_files: List of .rs file paths.

        Returns:
            Dict mapping relative file path → rust_analyzer data.
        """
        results: dict[str, dict[str, Any]] = {}

        for file_path in rs_files:
            rel = str(file_path)
            try:
                data = self.extract_file(file_path)
                results[rel] = data
                logger.debug(
                    "Extracted %d symbols from %s",
                    len(data["symbols"]),
                    rel,
                )
            except Exception:
                logger.exception("Failed to extract symbols from %s", rel)
                results[rel] = self._empty_result()

        return results

    def _process_symbols(
        self,
        raw_symbols: list[dict[str, Any]],
        file_path: str | Path,
        file_content: str,
    ) -> list[dict[str, Any]]:
        """Flatten and enrich the hierarchical documentSymbol tree.

        Walks the symbol tree, extracting each symbol with:
        - name, qualified_name, kind, visibility, signature
        - line range, parent context
        - attributes (PyO3, FFI, derive, unsafe)
        - call hierarchy (if enabled)
        """
        symbols: list[dict[str, Any]] = []
        lines = file_content.splitlines()

        def walk(
            sym: dict[str, Any],
            parent_name: str | None = None,
            parent_kind: str | None = None,
            impl_trait: str | None = None,
        ) -> None:
            name = sym.get("name", "")
            kind_id = sym.get("kind", 0)
            kind = _SYMBOL_KIND_MAP.get(kind_id, f"unknown_{kind_id}")
            detail = sym.get("detail", "")

            # Range info
            range_info = sym.get("range", {})
            sel_range = sym.get("selectionRange", {})
            start_line = range_info.get("start", {}).get("line", 0)
            end_line = range_info.get("end", {}).get("line", 0)
            sel_line = sel_range.get("start", {}).get("line", start_line)
            sel_char = sel_range.get("start", {}).get("character", 0)

            # Detect impl blocks — rust-analyzer reports them as top-level
            # symbols with names like "impl Foo" or "impl Bar for Foo"
            current_impl_trait = impl_trait
            impl_self_type = parent_name
            if parent_name and parent_name.startswith("impl "):
                impl_body = parent_name[5:]  # Strip "impl "
                if " for " in impl_body:
                    parts = impl_body.split(" for ", 1)
                    current_impl_trait = parts[0].strip()
                    impl_self_type = parts[1].strip()
                else:
                    impl_self_type = impl_body.strip()

            # Build qualified name
            if impl_self_type and impl_self_type != parent_name:
                # Method inside impl block — qualify under the struct
                qualified = f"{impl_self_type}::{name}"
            elif parent_name and not parent_name.startswith("impl "):
                qualified = f"{parent_name}::{name}"
            else:
                qualified = name

            # Extract visibility from source text at selectionRange line
            # (range.start may point to attributes above the symbol)
            visibility = self._parse_visibility_from_source(lines, sel_line)

            # Get hover for resolved signature (pub items only to limit LSP calls)
            signature = detail or ""
            if visibility in ("pub", "pub(crate)") and kind in ("function", "method", "struct", "enum", "constant"):
                hover_result = self._session.hover(file_path, line=sel_line, character=sel_char)
                if hover_result:
                    hover_sig = self._extract_signature_from_hover(hover_result)
                    if hover_sig:
                        signature = hover_sig

            # Scan source lines for attributes
            attrs = self._scan_attributes(lines, start_line, sel_line)

            is_pyo3 = any(_PYO3_PATTERNS.search(a) for a in attrs)
            is_ffi = any(_FFI_PATTERNS.search(a) for a in attrs)
            is_unsafe = "unsafe" in (detail or "") or any("unsafe" in a for a in attrs)
            derives = self._extract_derives(attrs)

            # Extract python_name from #[pyo3(name = "...")]
            python_name = name
            for attr in attrs:
                pyo3_name_match = re.search(r'pyo3\(.*name\s*=\s*"([^"]+)"', attr)
                if pyo3_name_match:
                    python_name = pyo3_name_match.group(1)

            # Resolve parent_symbol — for methods in impl blocks, use the struct name
            effective_parent = impl_self_type if (impl_self_type and impl_self_type != parent_name) else parent_name

            # Build symbol entry
            symbol: dict[str, Any] = {
                "name": name,
                "qualified_name": qualified,
                "kind": kind,
                "visibility": visibility,
                "signature": signature,
                "start_line": start_line,
                "end_line": end_line,
                "parent_symbol": effective_parent,
                "impl_trait": current_impl_trait,
                "attributes": attrs,
                "is_pyo3": is_pyo3,
                "is_ffi": is_ffi,
                "is_unsafe": is_unsafe,
                "derives": derives,
                "python_name": python_name if is_pyo3 else None,
            }

            # Call hierarchy (expensive — only for functions/methods)
            if self._include_calls and kind in ("function", "method"):
                symbol["calls"] = self._get_outgoing_calls(file_path, sel_line, sel_char)
                if self._include_incoming:
                    symbol["called_by"] = self._get_incoming_calls(file_path, sel_line, sel_char)

            symbols.append(symbol)

            # Recurse into children
            for child in sym.get("children", []):
                # For impl blocks (kind 19 / "object"), pass the impl name so children
                # can extract struct name and trait from it
                if name.startswith("impl "):
                    child_parent = name  # "impl Foo" or "impl Bar for Foo"
                elif kind in ("struct", "enum", "module", "interface"):
                    child_parent = qualified
                else:
                    child_parent = parent_name

                walk(
                    child,
                    parent_name=child_parent,
                    parent_kind=kind,
                    impl_trait=current_impl_trait,
                )

        for sym in raw_symbols:
            walk(sym)

        return symbols

    def _get_outgoing_calls(
        self, file_path: str | Path, line: int, character: int
    ) -> list[dict[str, Any]]:
        """Get outgoing calls for a symbol, formatted for storage."""
        try:
            raw = self._session.call_hierarchy_outgoing(file_path, line, character)
        except Exception:
            logger.debug("Call hierarchy failed for %s:%d:%d", file_path, line, character)
            return []

        calls = []
        for item in raw:
            to = item.get("to", {})
            target_uri = to.get("uri", "")
            target_name = to.get("name", "")
            target_detail = to.get("detail", "")
            target_range = to.get("range", {}).get("start", {})

            # Convert URI to relative path
            target_file = self._uri_to_rel_path(target_uri)

            # Build qualified name from detail or name
            qualified = target_detail if target_detail else target_name

            calls.append({
                "qualified_name": qualified,
                "name": target_name,
                "file": target_file,
                "line": target_range.get("line", 0),
            })

        return calls

    def _get_incoming_calls(
        self, file_path: str | Path, line: int, character: int
    ) -> list[dict[str, Any]]:
        """Get incoming calls to a symbol, formatted for storage."""
        try:
            raw = self._session.call_hierarchy_incoming(file_path, line, character)
        except Exception:
            logger.debug("Incoming calls failed for %s:%d:%d", file_path, line, character)
            return []

        calls = []
        for item in raw:
            from_item = item.get("from", {})
            caller_uri = from_item.get("uri", "")
            caller_name = from_item.get("name", "")
            caller_detail = from_item.get("detail", "")
            caller_range = from_item.get("range", {}).get("start", {})

            calls.append({
                "qualified_name": caller_detail if caller_detail else caller_name,
                "name": caller_name,
                "file": self._uri_to_rel_path(caller_uri),
                "line": caller_range.get("line", 0),
            })

        return calls

    def _parse_visibility_from_source(self, lines: list[str], start_line: int) -> str:
        """Parse visibility from the source text at a symbol's start line.

        rust-analyzer's detail field does not include visibility, so we
        extract it from the actual source code.
        """
        if start_line >= len(lines):
            return "private"

        line = lines[start_line].strip()

        # Check for pub(crate), pub(super), pub(in path)
        pub_restricted = re.match(r"pub\s*\(([^)]+)\)", line)
        if pub_restricted:
            return f"pub({pub_restricted.group(1).strip()})"

        if line.startswith("pub "):
            return "pub"

        return "private"

    def _extract_signature_from_hover(self, hover: dict[str, Any]) -> str | None:
        """Extract the type signature from a hover response."""
        contents = hover.get("contents", {})
        if isinstance(contents, dict):
            value = contents.get("value", "")
        elif isinstance(contents, str):
            value = contents
        elif isinstance(contents, list):
            # MarkedString array
            value = "\n".join(
                c.get("value", c) if isinstance(c, dict) else str(c)
                for c in contents
            )
        else:
            return None

        # Extract Rust code blocks from markdown.
        # Hover often has multiple: first is module path, second is signature.
        # Take the last one that looks like a signature.
        rust_blocks = re.findall(r"```rust\n(.*?)```", value, re.DOTALL)
        if rust_blocks:
            # Prefer blocks that contain fn/struct/enum/trait/const/type/impl
            for block in reversed(rust_blocks):
                stripped = block.strip()
                if any(kw in stripped for kw in ("fn ", "struct ", "enum ", "trait ", "const ", "type ", "impl ")):
                    return stripped
            # Fallback to the last block
            return rust_blocks[-1].strip()

        # Fallback: return the whole value if it looks like a signature
        if "fn " in value or "struct " in value or "enum " in value:
            return value.strip()

        return None

    def _scan_attributes(self, lines: list[str], start_line: int, sel_line: int) -> list[str]:
        """Scan lines at and above a symbol for Rust attributes (#[...]).

        The range.start.line may point to the first attribute, while
        selectionRange.start.line points to the actual symbol name.
        We scan from range start up to (but not including) selectionRange.
        """
        attrs = []
        # Scan from range start line backwards, and between range start and selection
        # First: attributes between range.start and selectionRange.start
        for i in range(start_line, min(sel_line, len(lines))):
            line = lines[i].strip()
            if line.startswith("#[") or line.startswith("#!["):
                attrs.append(line)

        # Then: look above range.start for more attributes
        i = start_line - 1
        while i >= 0:
            line = lines[i].strip() if i < len(lines) else ""
            if line.startswith("#[") or line.startswith("#!["):
                attrs.append(line)
                i -= 1
            elif line.startswith("//") or line.startswith("///") or not line:
                # Skip comments and blank lines
                i -= 1
            else:
                break
        return list(reversed(attrs))

    def _extract_derives(self, attrs: list[str]) -> list[str]:
        """Extract derive macro names from attributes."""
        derives = []
        for attr in attrs:
            match = re.search(r"#\[derive\(([^)]+)\)\]", attr)
            if match:
                for name in match.group(1).split(","):
                    name = name.strip()
                    if name:
                        derives.append(name)
        return derives

    def _extract_impl_blocks(self, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group symbols into impl blocks by their parent and trait."""
        blocks: dict[tuple[str | None, str | None], list[str]] = {}

        for sym in symbols:
            if sym["kind"] in ("method", "function") and sym.get("parent_symbol"):
                key = (sym["parent_symbol"], sym.get("impl_trait"))
                if key not in blocks:
                    blocks[key] = []
                blocks[key].append(sym["name"])

        return [
            {
                "self_type": parent or "unknown",
                "trait": trait,
                "methods": methods,
            }
            for (parent, trait), methods in blocks.items()
        ]

    def _uri_to_rel_path(self, uri: str) -> str:
        """Convert a file:// URI to a path relative to the crate root."""
        if uri.startswith("file://"):
            abs_path = Path(uri[7:])  # Strip file://
            try:
                return str(abs_path.relative_to(self._session.crate_root))
            except ValueError:
                return str(abs_path)
        return uri

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty extraction result."""
        return {
            "symbols": [],
            "impl_blocks": [],
            "pyo3_exports": [],
            "ffi_boundaries": [],
            "analyzed_at": datetime.now(UTC).isoformat(),
        }
