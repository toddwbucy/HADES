"""Tests for the Rust edge resolver.

Validates that symbol nodes and edges are correctly materialized
from rust_analyzer file-node attributes. No LSP needed — operates
on pre-built data structures.
"""

from __future__ import annotations

import pytest

from core.analyzers.rust_edge_resolver import RustEdgeResolver, symbol_key
from core.database.keys import file_key

# ── Fixtures ──────────────────────────────────────────────────────


def _make_file_node(
    rel_path: str,
    symbols: list[dict],
    impl_blocks: list[dict] | None = None,
    pyo3_exports: list[str] | None = None,
    ffi_boundaries: list[str] | None = None,
) -> dict:
    """Build a file node with rust_analyzer attribute."""
    return {
        "rel_path": rel_path,
        "rust_analyzer": {
            "symbols": symbols,
            "impl_blocks": impl_blocks or [],
            "pyo3_exports": pyo3_exports or [],
            "ffi_boundaries": ffi_boundaries or [],
            "analyzed_at": "2026-03-22T00:00:00+00:00",
        },
    }


@pytest.fixture
def math_file() -> dict:
    """File node for math.rs with three functions, one calling the other two."""
    return _make_file_node(
        "src/math.rs",
        symbols=[
            {
                "name": "add",
                "qualified_name": "add",
                "kind": "function",
                "visibility": "pub",
                "signature": "pub fn add(a: i32, b: i32) -> i32",
                "start_line": 0,
                "end_line": 2,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
                "calls": [],
            },
            {
                "name": "multiply",
                "qualified_name": "multiply",
                "kind": "function",
                "visibility": "pub",
                "signature": "pub fn multiply(a: i32, b: i32) -> i32",
                "start_line": 4,
                "end_line": 6,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
                "calls": [],
            },
            {
                "name": "add_then_double",
                "qualified_name": "add_then_double",
                "kind": "function",
                "visibility": "pub",
                "signature": "pub fn add_then_double(a: i32, b: i32) -> i32",
                "start_line": 8,
                "end_line": 11,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
                "calls": [
                    {"qualified_name": "add", "name": "add", "file": "src/math.rs", "line": 9},
                    {"qualified_name": "multiply", "name": "multiply", "file": "src/math.rs", "line": 10},
                ],
            },
        ],
    )


@pytest.fixture
def model_file() -> dict:
    """File node for model.rs with struct, trait, impl, and cross-file call."""
    return _make_file_node(
        "src/model.rs",
        symbols=[
            {
                "name": "Model",
                "qualified_name": "Model",
                "kind": "struct",
                "visibility": "pub",
                "signature": "pub struct Model",
                "start_line": 2,
                "end_line": 5,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": ["Debug", "Clone"],
                "python_name": None,
            },
            {
                "name": "Forward",
                "qualified_name": "Forward",
                "kind": "interface",
                "visibility": "pub",
                "signature": "pub trait Forward",
                "start_line": 7,
                "end_line": 9,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
            },
            {
                "name": "new",
                "qualified_name": "Model::new",
                "kind": "method",
                "visibility": "pub",
                "signature": "pub fn new(size: usize) -> Self",
                "start_line": 12,
                "end_line": 17,
                "parent_symbol": "Model",
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
                "calls": [],
            },
            {
                "name": "forward",
                "qualified_name": "Model::forward",
                "kind": "method",
                "visibility": "pub",
                "signature": "fn forward(&self, input: &[f64]) -> Vec<f64>",
                "start_line": 24,
                "end_line": 28,
                "parent_symbol": "Model",
                "impl_trait": "Forward",
                "is_pyo3": False,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": None,
                "calls": [
                    {"qualified_name": "add", "name": "add", "file": "src/math.rs", "line": 26},
                ],
            },
        ],
    )


@pytest.fixture
def pyo3_file() -> dict:
    """File node with PyO3-exported function."""
    return _make_file_node(
        "src/lib.rs",
        symbols=[
            {
                "name": "compute",
                "qualified_name": "compute",
                "kind": "function",
                "visibility": "pub",
                "signature": "pub fn compute(x: f64) -> f64",
                "start_line": 0,
                "end_line": 3,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": True,
                "is_ffi": False,
                "is_unsafe": False,
                "derives": [],
                "python_name": "py_compute",
            },
        ],
        pyo3_exports=["compute"],
    )


@pytest.fixture
def ffi_file() -> dict:
    """File node with FFI-exposed function."""
    return _make_file_node(
        "src/ffi.rs",
        symbols=[
            {
                "name": "init_cuda",
                "qualified_name": "init_cuda",
                "kind": "function",
                "visibility": "pub",
                "signature": 'extern "C" fn init_cuda()',
                "start_line": 0,
                "end_line": 5,
                "parent_symbol": None,
                "impl_trait": None,
                "is_pyo3": False,
                "is_ffi": True,
                "is_unsafe": True,
                "derives": [],
                "python_name": None,
            },
        ],
        ffi_boundaries=["init_cuda"],
    )


# ── symbol_key tests ─────────────────────────────────────────────


class TestSymbolKey:
    def test_basic(self) -> None:
        assert symbol_key("src/model.rs", "Model::new") == "src_model_rs__Model__new"

    def test_no_namespace(self) -> None:
        assert symbol_key("src/math.rs", "add") == "src_math_rs__add"

    def test_nested(self) -> None:
        result = symbol_key("src/lib.rs", "MyMod::Inner::foo")
        assert result == "src_lib_rs__MyMod__Inner__foo"


# ── Symbol node tests ────────────────────────────────────────────


class TestBuildSymbolNodes:
    def test_symbol_count(self, math_file: dict) -> None:
        resolver = RustEdgeResolver([math_file])
        symbols = resolver.build_symbol_nodes()
        assert len(symbols) == 3

    def test_symbol_has_key(self, math_file: dict) -> None:
        resolver = RustEdgeResolver([math_file])
        symbols = resolver.build_symbol_nodes()
        for sym in symbols:
            assert "_key" in sym
            assert sym["_key"]  # Non-empty

    def test_symbol_fields(self, math_file: dict) -> None:
        resolver = RustEdgeResolver([math_file])
        symbols = resolver.build_symbol_nodes()
        add = next(s for s in symbols if s["name"] == "add")
        assert add["qualified_name"] == "add"
        assert add["kind"] == "function"
        assert add["visibility"] == "pub"
        assert add["file_path"] == "src/math.rs"
        assert add["start_line"] == 0
        assert add["end_line"] == 2

    def test_struct_with_derives(self, model_file: dict) -> None:
        resolver = RustEdgeResolver([model_file])
        symbols = resolver.build_symbol_nodes()
        model = next(s for s in symbols if s["name"] == "Model")
        assert "Debug" in model["derives"]
        assert "Clone" in model["derives"]

    def test_pyo3_symbol(self, pyo3_file: dict) -> None:
        resolver = RustEdgeResolver([pyo3_file])
        symbols = resolver.build_symbol_nodes()
        compute = symbols[0]
        assert compute["is_pyo3"] is True
        assert compute["python_name"] == "py_compute"

    def test_multi_file(self, math_file: dict, model_file: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file])
        symbols = resolver.build_symbol_nodes()
        assert len(symbols) == 7  # 3 from math + 4 from model

    def test_no_symbols(self) -> None:
        empty = _make_file_node("src/empty.rs", symbols=[])
        resolver = RustEdgeResolver([empty])
        assert resolver.build_symbol_nodes() == []


# ── Edge tests ───────────────────────────────────────────────────


class TestBuildEdges:
    def test_defines_edges(self, math_file: dict) -> None:
        resolver = RustEdgeResolver([math_file])
        edges = resolver.build_edges()
        defines = [e for e in edges if e["type"] == "defines"]
        assert len(defines) == 3
        # All should come from the file
        fk = file_key("src/math.rs")
        for e in defines:
            assert e["_from"] == f"codebase_files/{fk}"
            assert e["_to"].startswith("codebase_symbols/")

    def test_calls_edges_same_file(self, math_file: dict) -> None:
        resolver = RustEdgeResolver([math_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        assert len(calls) == 2
        callees = {e["callee"] for e in calls}
        assert "add" in callees
        assert "multiply" in callees

    def test_cross_file_calls(self, math_file: dict, model_file: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        # Model::forward calls add from math.rs
        cross_file = [e for e in calls if e["caller"] == "Model::forward"]
        assert len(cross_file) == 1
        assert cross_file[0]["callee"] == "add"

    def test_implements_edge(self, model_file: dict) -> None:
        resolver = RustEdgeResolver([model_file])
        edges = resolver.build_edges()
        implements = [e for e in edges if e["type"] == "implements"]
        assert len(implements) == 1
        assert implements[0]["implementor"] == "Model::forward"
        assert implements[0]["trait"] == "Forward"

    def test_pyo3_exposes_edge(self, pyo3_file: dict) -> None:
        resolver = RustEdgeResolver([pyo3_file])
        edges = resolver.build_edges()
        pyo3 = [e for e in edges if e["type"] == "pyo3_exposes"]
        assert len(pyo3) == 1
        assert pyo3[0]["python_name"] == "py_compute"
        assert pyo3[0]["symbol_name"] == "compute"

    def test_ffi_exposes_edge(self, ffi_file: dict) -> None:
        resolver = RustEdgeResolver([ffi_file])
        edges = resolver.build_edges()
        ffi = [e for e in edges if e["type"] == "ffi_exposes"]
        assert len(ffi) == 1
        assert ffi[0]["symbol_name"] == "init_cuda"

    def test_no_duplicate_edges(self, math_file: dict) -> None:
        """Same edge should not appear twice."""
        resolver = RustEdgeResolver([math_file])
        edges = resolver.build_edges()
        edge_set = {(e["_from"], e["_to"], e["type"]) for e in edges}
        assert len(edge_set) == len(edges)

    def test_empty_file(self) -> None:
        empty = _make_file_node("src/empty.rs", symbols=[])
        resolver = RustEdgeResolver([empty])
        assert resolver.build_edges() == []


# ── Call resolution tests ────────────────────────────────────────


class TestCallResolution:
    def test_qualified_name_resolution(self, math_file: dict) -> None:
        """Calls with qualified_name matching index should resolve."""
        resolver = RustEdgeResolver([math_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        # Both calls from add_then_double should resolve
        assert len(calls) == 2

    def test_cross_file_resolution(self, math_file: dict, model_file: dict) -> None:
        """Calls referencing another file's symbol should resolve via file::name index."""
        resolver = RustEdgeResolver([math_file, model_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls" and e["caller"] == "Model::forward"]
        assert len(calls) == 1
        # Should point to math.rs's add
        target_key = symbol_key("src/math.rs", "add")
        assert calls[0]["_to"] == f"codebase_symbols/{target_key}"

    def test_unresolvable_call_skipped(self) -> None:
        """Calls to unknown symbols should be silently skipped."""
        node = _make_file_node(
            "src/main.rs",
            symbols=[{
                "name": "main",
                "qualified_name": "main",
                "kind": "function",
                "visibility": "pub",
                "start_line": 0,
                "end_line": 5,
                "calls": [
                    {"qualified_name": "unknown::function", "name": "function", "file": "src/unknown.rs", "line": 3},
                ],
            }],
        )
        resolver = RustEdgeResolver([node])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        assert len(calls) == 0
