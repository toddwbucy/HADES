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

from core.analyzers.lsp_client import LspError
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

# PyO3 attribute patterns — matches parameterized forms like #[pyclass(name = "...")]
_PYO3_PATTERNS = re.compile(
    r"#\[(?:pyclass|pymethods|pyfunction|pymodule|pyproto)(?:\([^]]*\))?\]|#\[pyo3\([^]]*\)\]"
)

# FFI patterns — matches Rust 2024 unsafe-wrapped forms like #[unsafe(no_mangle)]
_FFI_PATTERNS = re.compile(
    r'(extern\s+"C"|#\[(?:unsafe\()?no_mangle(?:\))?\]|#\[(?:unsafe\()?export_name\s*=\s*"[^"]+"(?:\))?\])'
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

        # Extract use-imports via goto_definition (file-level)
        imports = self._extract_use_imports(file_path, file_content)

        return {
            "symbols": symbols,
            "impl_blocks": impl_blocks,
            "pyo3_exports": pyo3_exports,
            "ffi_boundaries": ffi_boundaries,
            "imports": imports,
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
            parent_is_pyo3: bool = False,
            parent_is_ffi: bool = False,
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

            # Scan source lines for attributes
            attrs = self._scan_attributes(lines, start_line, sel_line)

            is_pyo3 = any(_PYO3_PATTERNS.search(a) for a in attrs) or parent_is_pyo3
            is_ffi = any(_FFI_PATTERNS.search(a) for a in attrs) or parent_is_ffi
            is_unsafe = "unsafe" in (detail or "") or any("unsafe" in a for a in attrs)

            # impl blocks are containers, not symbols — skip emitting them
            # but propagate their metadata (PyO3, FFI, trait) to children
            if name.startswith("impl "):
                for child in sym.get("children", []):
                    walk(
                        child,
                        parent_name=name,
                        parent_kind=kind,
                        impl_trait=current_impl_trait,
                        parent_is_pyo3=is_pyo3,
                        parent_is_ffi=is_ffi,
                    )
                return

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

            # Recurse into children (non-impl containers like struct, enum, module)
            for child in sym.get("children", []):
                if kind in ("struct", "enum", "module", "interface"):
                    child_parent = qualified
                else:
                    child_parent = parent_name

                walk(
                    child,
                    parent_name=child_parent,
                    parent_kind=kind,
                    impl_trait=current_impl_trait,
                    parent_is_pyo3=is_pyo3,
                    parent_is_ffi=is_ffi,
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

    # ── Use-import extraction ─────────────────────────────────────

    def _extract_use_imports(
        self, file_path: str | Path, file_content: str
    ) -> list[dict[str, Any]]:
        """Extract ``use`` declarations and resolve each imported name via LSP.

        Parses file content for ``use`` statements, identifies the terminal
        identifiers (the names actually brought into scope), and calls
        ``goto_definition`` on each to resolve to the definition site.

        Only imports that resolve to files within the crate root are kept
        (external crates like ``std``, ``serde`` are filtered out).

        Returns:
            List of import dicts::

                {
                    "name": "HashMap",
                    "use_statement": "use std::collections::HashMap;",
                    "source_line": 5,
                    "target_file": "src/collections/hash/map.rs",
                    "target_line": 42,
                    "target_name": "HashMap",
                    "qualified_name": "HashMap",
                }

            Only internal (in-crate) imports are included.
        """
        imports: list[dict[str, Any]] = []
        lines = file_content.splitlines()

        # Assemble complete use statements (may span multiple lines) and
        # record which original lines they came from.  Each entry is
        # (start_line_no, list_of_(line_no, raw_line) pairs).
        use_stmts = _collect_use_statements(lines)

        for start_line, span_lines in use_stmts:
            # Build a single normalized statement for parsing:
            # join continuation lines, strip inline comments.
            joined = " ".join(raw.strip() for _, raw in span_lines)
            # Strip trailing inline comment (after the semicolon)
            semi_pos = joined.find(";")
            if semi_pos != -1:
                joined = joined[: semi_pos + 1]

            # For single-line statements we pass the original line so
            # column positions map directly to the source.  For multiline
            # we pass the joined form and use (start_line, col) — the LSP
            # still resolves correctly because goto_definition only needs
            # a position that lands on the identifier token, and for
            # multiline use-stmts rust-analyzer associates the whole
            # statement with the opening line.
            if len(span_lines) == 1:
                orig_line_no, orig_line = span_lines[0]
                # Truncate at semicolon to strip inline comments.
                # Column positions are preserved since the prefix is unchanged.
                semi_pos = orig_line.find(";")
                parse_line = orig_line[: semi_pos + 1] if semi_pos != -1 else orig_line
                positions = self._parse_use_positions(parse_line, orig_line_no)
            else:
                # Multiline: parse from joined text.  Column positions
                # refer to the joined string, but we need real (line, col)
                # for goto_definition.  Build a mapping from joined-offset
                # back to (line_no, col).
                positions_raw = self._parse_use_positions(joined, start_line)
                positions = _remap_multiline_positions(
                    positions_raw, span_lines, joined,
                )

            if not positions:
                continue

            for name, lsp_line, lsp_char in positions:
                try:
                    locations = self._session.goto_definition(
                        file_path, lsp_line, lsp_char,
                        timeout=5.0, retries=1, retry_delay=0.5,
                    )
                except LspError as e:
                    logger.debug(
                        "goto_definition failed for use '%s' at %s:%d:%d: %s",
                        name, file_path, lsp_line, lsp_char, e,
                    )
                    continue

                if not locations:
                    continue

                loc = locations[0]
                target_uri = loc.get("uri", "")
                target_file = self._uri_to_rel_path(target_uri)

                # Filter: only keep imports that resolve within the crate
                if Path(target_file).is_absolute() or target_file == target_uri:
                    # Absolute path or unresolvable URI → external crate
                    continue

                target_line = loc.get("range", {}).get("start", {}).get("line", 0)

                imports.append({
                    "name": name,
                    "use_statement": joined.strip(),
                    "source_line": start_line,
                    "target_file": target_file,
                    "target_line": target_line,
                    "target_name": name,
                    "qualified_name": name,
                })

        logger.debug(
            "Extracted %d internal use-imports from %s", len(imports), file_path,
        )
        return imports

    @staticmethod
    def _parse_use_positions(
        line: str, line_no: int
    ) -> list[tuple[str, int, int]]:
        """Parse a ``use`` line and return ``(name, line, col)`` per imported symbol.

        Handles common Rust ``use`` patterns:

        - Simple: ``use crate::module::Name;``
        - Grouped: ``use crate::module::{A, B, C};``
        - Aliased: ``use crate::module::Name as Alias;``
        - Self: ``use crate::module::{self, Name};`` (``self`` skipped)
        - Glob: ``use crate::module::*;`` (skipped entirely)
        - Nested groups: ``use std::{io::{Read, Write}, fmt};``

        Returns:
            List of (name, 0-based line, 0-based character offset) tuples.
        """
        stripped = line.strip()
        if not stripped.startswith("use "):
            return []

        # Skip glob imports — can't resolve individually
        if stripped.rstrip(";").rstrip().endswith("*"):
            return []

        results: list[tuple[str, int, int]] = []

        # Check for grouped imports (any braces present)
        brace_open = stripped.find("{")
        if brace_open != -1:
            # Extract all terminal identifiers inside braces (handles nesting)
            # We find identifiers that are followed by , or } or " as " or ;
            # but NOT followed by :: (those are path segments, not terminals)
            _extract_group_terminals(stripped, line, line_no, results)
        else:
            # Simple: use path::to::Name; or use path::to::Name as Alias;
            body = stripped[4:].rstrip(";").strip()

            # Remove alias: "Name as Alias" → "Name"
            if " as " in body:
                body = body.split(" as ")[0].strip()

            # Get the terminal identifier (last path segment)
            parts = body.split("::")
            name = parts[-1].strip()
            if name and name.isidentifier() and name != "self":
                # Find the character position in the original line (word-boundary safe)
                search_end = len(line)
                if " as " in line:
                    # Constrain search to before the alias keyword
                    as_pos = line.find(" as ")
                    if as_pos != -1:
                        search_end = as_pos
                # Search backwards for a whole-word match
                idx = search_end
                while idx > 0:
                    idx = line.rfind(name, 0, idx)
                    if idx == -1:
                        break
                    if _is_word_boundary(line, idx, len(name)):
                        results.append((name, line_no, idx))
                        break
                    # Not a word boundary — keep searching leftward
                    idx -= 1

        return results

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
            "imports": [],
            "analyzed_at": datetime.now(UTC).isoformat(),
        }


# ── Module-level helpers for use-statement parsing ──────────────


def _is_word_boundary(line: str, start: int, length: int) -> bool:
    """Check that ``line[start:start+length]`` is a whole-word match.

    The character before *start* and after *start+length-1* (if they exist)
    must not be identifier characters (letter, digit, underscore).
    """
    if start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
        return False
    end = start + length
    if end < len(line) and (line[end].isalnum() or line[end] == "_"):
        return False
    return True


def _find_word(line: str, name: str, start: int = 0) -> int:
    """Find ``name`` in ``line`` at a word boundary, searching from *start*.

    Returns the index, or -1 if not found.
    """
    idx = start
    while True:
        idx = line.find(name, idx)
        if idx == -1:
            return -1
        if _is_word_boundary(line, idx, len(name)):
            return idx
        idx += 1


def _extract_group_terminals(
    use_stmt: str,
    original_line: str,
    line_no: int,
    results: list[tuple[str, int, int]],
    used_cols: set[int] | None = None,
) -> None:
    """Extract terminal identifiers from grouped/nested ``use`` statements.

    Walks the characters inside braces to find identifiers that are
    terminal (not followed by ``::``).  Handles nested groups like
    ``use std::{io::{Read, Write}, fmt::Display};``

    *used_cols* tracks column positions already claimed on this line so
    that duplicate identifiers (e.g. ``use a::{Foo, b::Foo}``) resolve
    to distinct positions.
    """
    if used_cols is None:
        used_cols = set()

    brace_start = use_stmt.find("{")
    brace_end = use_stmt.rfind("}")
    if brace_start == -1 or brace_end == -1:
        return

    inner = use_stmt[brace_start + 1 : brace_end]
    items = _split_respecting_braces(inner)

    for item in items:
        item = item.strip()
        if not item or item == "self":
            continue

        if "{" in item:
            # Nested group: "io::{Read, Write}" — recurse
            synthetic = f"use {item};"
            _extract_group_terminals(synthetic, original_line, line_no, results, used_cols)
            continue

        # Remove alias
        if " as " in item:
            item = item.split(" as ")[0].strip()

        # Skip globs
        if item.endswith("*"):
            continue

        # Get terminal name (last path segment)
        parts = item.split("::")
        name = parts[-1].strip()
        if name and name.isidentifier() and name != "self":
            # Search for a word-boundary match not yet claimed
            search_from = 0
            while True:
                idx = _find_word(original_line, name, search_from)
                if idx == -1:
                    break
                if idx not in used_cols:
                    used_cols.add(idx)
                    results.append((name, line_no, idx))
                    break
                search_from = idx + 1


def _split_respecting_braces(s: str) -> list[str]:
    """Split a string on commas, but respect nested ``{...}`` groups."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in s:
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


def _collect_use_statements(
    lines: list[str],
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Collect complete ``use`` statements, joining continuation lines.

    Rust ``use`` statements may span multiple lines::

        use std::{
            io::Read,
            fmt::Display,
        };

    Returns a list of ``(start_line_no, [(line_no, raw_line), ...])``.
    Each entry is one complete ``use ...;`` statement with all its
    contributing source lines.
    """
    result: list[tuple[int, list[tuple[int, str]]]] = []
    current: list[tuple[int, str]] | None = None
    start_line = 0

    for line_no, line in enumerate(lines):
        stripped = line.strip()

        if current is not None:
            # Continuation of a multiline use statement
            current.append((line_no, line))
            if ";" in stripped:
                result.append((start_line, current))
                current = None
            continue

        if not stripped.startswith("use "):
            continue

        if ";" in stripped:
            # Single-line use statement (most common case)
            result.append((line_no, [(line_no, line)]))
        else:
            # Multiline use statement — starts here, continues until ";"
            current = [(line_no, line)]
            start_line = line_no

    # Unterminated use statement at EOF — include as-is so partial
    # parsing can still extract what it can.
    if current is not None:
        result.append((start_line, current))

    return result


def _remap_multiline_positions(
    positions: list[tuple[str, int, int]],
    span_lines: list[tuple[int, str]],
    joined: str,
) -> list[tuple[str, int, int]]:
    """Map column positions in a joined string back to real (line, col).

    ``_parse_use_positions`` was given the joined (single-line) form of a
    multiline ``use`` statement.  The returned column offsets refer to that
    joined string.  This function finds which original source line each
    name actually lives on and computes the real column.

    Args:
        positions: ``(name, start_line, col_in_joined)`` from the parser.
        span_lines: ``[(line_no, raw_line), ...]`` — the original lines.
        joined: The joined string that was parsed.

    Returns:
        ``(name, real_line_no, real_col)`` tuples.
    """
    remapped: list[tuple[str, int, int]] = []
    for name, _, col_in_joined in positions:
        # For each name, find which original line contains it by
        # searching for the name token in each contributing line.
        found = False
        for real_line_no, raw_line in span_lines:
            idx = _find_word(raw_line, name)
            if idx != -1:
                remapped.append((name, real_line_no, idx))
                found = True
                break
        if not found:
            # Fallback: use the start line and joined column.
            # This can happen if the name only appears in the joined
            # form (unlikely but defensive).
            remapped.append((name, span_lines[0][0], col_in_joined))
    return remapped
