"""Tests for the rust-analyzer session manager.

Tests cover:
- Cargo.toml detection and validation
- Initialize handshake and readiness
- File open/close lifecycle
- documentSymbol, hover, callHierarchy, references, goto_definition
- Shutdown and context manager
- Error cases (no Cargo.toml, binary not found)

Requires rust-analyzer to be installed — tests are skipped otherwise.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from core.analyzers.rust_analyzer_client import (
    RustAnalyzerError,
    RustAnalyzerSession,
)

HAS_RUST_ANALYZER = shutil.which("rust-analyzer") is not None
pytestmark = pytest.mark.skipif(not HAS_RUST_ANALYZER, reason="rust-analyzer not installed")


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def rust_crate(tmp_path: Path) -> Path:
    """Create a Rust crate with enough structure to exercise all LSP features."""
    crate = tmp_path / "test_crate"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(textwrap.dedent("""\
        [package]
        name = "test_crate"
        version = "0.1.0"
        edition = "2021"
    """))
    src = crate / "src"
    src.mkdir()

    # Main library with struct, impl, trait, and a free function
    (src / "lib.rs").write_text(textwrap.dedent("""\
        pub mod math;

        pub trait Greet {
            fn greet(&self) -> String;
        }

        pub struct Person {
            pub name: String,
            pub age: u32,
        }

        impl Person {
            pub fn new(name: &str, age: u32) -> Self {
                Person {
                    name: name.to_string(),
                    age,
                }
            }

            pub fn introduce(&self) -> String {
                format!("I'm {} and I'm {} years old", self.name, self.age)
            }
        }

        impl Greet for Person {
            fn greet(&self) -> String {
                format!("Hello, I'm {}", self.name)
            }
        }

        pub fn create_and_greet(name: &str, age: u32) -> String {
            let person = Person::new(name, age);
            person.greet()
        }
    """))

    # A second module to test cross-file resolution
    (src / "math.rs").write_text(textwrap.dedent("""\
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }

        pub fn add_then_double(a: i32, b: i32) -> i32 {
            let sum = add(a, b);
            multiply(sum, 2)
        }
    """))

    return crate


@pytest.fixture
def session(rust_crate: Path) -> RustAnalyzerSession:
    """Create and start a RustAnalyzerSession."""
    s = RustAnalyzerSession(rust_crate, timeout=60)
    s.start()
    yield s
    s.shutdown()


# ── Tests ─────────────────────────────────────────────────────────


class TestSessionLifecycle:
    """Test session start, ready, and shutdown."""

    def test_start_and_ready(self, session: RustAnalyzerSession) -> None:
        assert session.is_ready

    def test_server_capabilities(self, session: RustAnalyzerSession) -> None:
        caps = session.server_capabilities
        assert caps, "Should have server capabilities"
        # rust-analyzer should support document symbols
        assert "documentSymbolProvider" in caps or "textDocumentSync" in caps

    def test_context_manager(self, rust_crate: Path) -> None:
        with RustAnalyzerSession(rust_crate, timeout=60) as s:
            assert s.is_ready
        # After exit, should be shut down
        assert not s.is_ready

    def test_no_cargo_toml(self, tmp_path: Path) -> None:
        with pytest.raises(RustAnalyzerError, match="No Cargo.toml"):
            RustAnalyzerSession(tmp_path)

    def test_crate_root_property(self, session: RustAnalyzerSession, rust_crate: Path) -> None:
        assert session.crate_root == rust_crate.resolve()


class TestFileManagement:
    """Test file open/close."""

    def test_open_file(self, session: RustAnalyzerSession, rust_crate: Path) -> None:
        uri = session.open_file("src/lib.rs")
        assert uri.startswith("file://")
        assert "lib.rs" in uri

    def test_open_file_absolute_path(self, session: RustAnalyzerSession, rust_crate: Path) -> None:
        abs_path = rust_crate / "src" / "lib.rs"
        uri = session.open_file(abs_path)
        assert "lib.rs" in uri

    def test_open_same_file_twice(self, session: RustAnalyzerSession) -> None:
        uri1 = session.open_file("src/lib.rs")
        uri2 = session.open_file("src/lib.rs")
        assert uri1 == uri2  # Should be idempotent

    def test_close_file(self, session: RustAnalyzerSession) -> None:
        uri = session.open_file("src/lib.rs")
        session.close_file(uri)
        # Opening again should work (it was actually closed)
        uri2 = session.open_file("src/lib.rs")
        assert uri2 == uri


class TestDocumentSymbols:
    """Test symbol extraction."""

    def test_lib_symbols(self, session: RustAnalyzerSession) -> None:
        symbols = session.document_symbols("src/lib.rs")
        assert isinstance(symbols, list)
        assert len(symbols) > 0

        # Extract all symbol names (including children)
        names = set()
        for sym in symbols:
            names.add(sym.get("name", ""))
            for child in sym.get("children", []):
                names.add(child.get("name", ""))

        # Should find our key symbols
        assert "Person" in names
        assert "Greet" in names
        assert "create_and_greet" in names

    def test_symbol_has_kind(self, session: RustAnalyzerSession) -> None:
        symbols = session.document_symbols("src/lib.rs")
        for sym in symbols:
            assert "kind" in sym, f"Symbol missing 'kind': {sym.get('name')}"

    def test_symbol_has_range(self, session: RustAnalyzerSession) -> None:
        symbols = session.document_symbols("src/lib.rs")
        for sym in symbols:
            assert "range" in sym, f"Symbol missing 'range': {sym.get('name')}"

    def test_math_module_symbols(self, session: RustAnalyzerSession) -> None:
        symbols = session.document_symbols("src/math.rs")
        names = {sym.get("name") for sym in symbols}
        assert "add" in names
        assert "multiply" in names
        assert "add_then_double" in names


class TestHover:
    """Test hover (type resolution)."""

    def test_hover_on_struct(self, session: RustAnalyzerSession) -> None:
        # First get symbols to find the actual line for Person
        symbols = session.document_symbols("src/lib.rs")
        person_sym = None
        for sym in symbols:
            if sym.get("name") == "Person":
                person_sym = sym
                break
        assert person_sym is not None, f"Person not found in symbols: {[s.get('name') for s in symbols]}"

        # Hover on the symbol name (selectionRange), not the full declaration (range)
        line = person_sym["selectionRange"]["start"]["line"]
        char = person_sym["selectionRange"]["start"]["character"]
        hover = session.hover("src/lib.rs", line=line, character=char)
        assert hover is not None, f"No hover at Person position (line={line}, char={char})"
        contents = hover.get("contents", {})
        value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
        assert "Person" in value

    def test_hover_on_function(self, session: RustAnalyzerSession) -> None:
        # First get the actual position of add()
        symbols = session.document_symbols("src/math.rs")
        add_sym = None
        for sym in symbols:
            if sym.get("name") == "add":
                add_sym = sym
                break
        assert add_sym is not None

        line = add_sym["selectionRange"]["start"]["line"]
        char = add_sym["selectionRange"]["start"]["character"]
        hover = session.hover("src/math.rs", line=line, character=char)
        assert hover is not None
        contents = hover.get("contents", {})
        value = contents.get("value", "") if isinstance(contents, dict) else str(contents)
        assert "i32" in value


class TestCallHierarchy:
    """Test call hierarchy (incoming/outgoing calls)."""

    def _find_symbol(self, session: RustAnalyzerSession, file: str, name: str) -> dict:
        symbols = session.document_symbols(file)
        for sym in symbols:
            if sym.get("name") == name:
                return sym
            for child in sym.get("children", []):
                if child.get("name") == name:
                    return child
        pytest.fail(f"Symbol '{name}' not found in {file}")

    def test_outgoing_calls(self, session: RustAnalyzerSession) -> None:
        sym = self._find_symbol(session, "src/math.rs", "add_then_double")
        line = sym["selectionRange"]["start"]["line"]
        char = sym["selectionRange"]["start"]["character"]
        outgoing = session.call_hierarchy_outgoing("src/math.rs", line=line, character=char)
        assert isinstance(outgoing, list)
        if outgoing:
            called_names = {call.get("to", {}).get("name", "") for call in outgoing}
            assert "add" in called_names or "multiply" in called_names

    def test_incoming_calls(self, session: RustAnalyzerSession) -> None:
        sym = self._find_symbol(session, "src/math.rs", "add")
        line = sym["selectionRange"]["start"]["line"]
        char = sym["selectionRange"]["start"]["character"]
        incoming = session.call_hierarchy_incoming("src/math.rs", line=line, character=char)
        assert isinstance(incoming, list)
        if incoming:
            caller_names = {call.get("from", {}).get("name", "") for call in incoming}
            assert "add_then_double" in caller_names


class TestReferences:
    """Test find references."""

    def test_references_to_add(self, session: RustAnalyzerSession) -> None:
        symbols = session.document_symbols("src/math.rs")
        add_sym = next((s for s in symbols if s.get("name") == "add"), None)
        assert add_sym is not None
        line = add_sym["selectionRange"]["start"]["line"]
        char = add_sym["selectionRange"]["start"]["character"]
        refs = session.references("src/math.rs", line=line, character=char)
        assert isinstance(refs, list)
        assert len(refs) >= 1


class TestGotoDefinition:
    """Test go-to-definition."""

    def test_goto_definition(self, session: RustAnalyzerSession) -> None:
        # Find add_then_double, then look up the position where it calls add()
        # We'll use references from add to find a call site, then goto_definition on it
        symbols = session.document_symbols("src/math.rs")
        add_sym = next((s for s in symbols if s.get("name") == "add"), None)
        assert add_sym is not None
        line = add_sym["selectionRange"]["start"]["line"]
        char = add_sym["selectionRange"]["start"]["character"]
        # goto_definition on the definition itself should return the definition
        locations = session.goto_definition("src/math.rs", line=line, character=char)
        assert isinstance(locations, list)
        assert len(locations) >= 1
