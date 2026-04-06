"""Tests for the Python edge resolver.

Validates that symbol nodes and edges are correctly materialized
from python_ast file-node attributes.  Structurally mirrors
test_rust_edge_resolver.py.
"""

from __future__ import annotations

import pytest

from core.analyzers.python_edge_resolver import PythonEdgeResolver, symbol_key
from core.database.keys import file_key


# ── Helpers ───────────────────────────────────────────────────────


def _make_file_node(
    rel_path: str,
    symbols: list[dict],
    imports: list[dict] | None = None,
) -> dict:
    """Build a file node with python_ast attribute."""
    return {
        "rel_path": rel_path,
        "python_ast": {
            "symbols": symbols,
            "imports": imports or [],
            "analyzed_at": "2026-04-06T00:00:00+00:00",
        },
    }


# ── symbol_key ────────────────────────────────────────────────────


class TestSymbolKey:
    def test_basic(self):
        assert symbol_key("core/db.py", "Pool") == "core_db_py__Pool"

    def test_method(self):
        sk = symbol_key("core/db.py", "Pool.acquire")
        assert sk == "core_db_py__Pool_acquire"

    def test_dunder(self):
        sk = symbol_key("core/db.py", "Pool.__init__")
        assert sk == "core_db_py__Pool__init"

    def test_nested(self):
        sk = symbol_key("mod.py", "Outer.Inner.method")
        assert sk == "mod_py__Outer_Inner_method"


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def pool_file() -> dict:
    """File node for db.py with a class and two methods."""
    return _make_file_node(
        "core/db.py",
        symbols=[
            {
                "name": "Pool",
                "qualified_name": "Pool",
                "kind": "class",
                "visibility": "public",
                "signature": "class Pool",
                "start_line": 10,
                "end_line": 50,
                "parent_symbol": None,
                "decorators": [],
                "bases": [],
            },
            {
                "name": "__init__",
                "qualified_name": "Pool.__init__",
                "kind": "method",
                "visibility": "dunder",
                "signature": "def __init__(self, size: int)",
                "start_line": 12,
                "end_line": 20,
                "parent_symbol": "Pool",
                "decorators": [],
                "calls": [{"name": "connect", "qualified_name": "self.connect"}],
            },
            {
                "name": "connect",
                "qualified_name": "Pool.connect",
                "kind": "method",
                "visibility": "public",
                "signature": "def connect(self) -> Connection",
                "start_line": 22,
                "end_line": 30,
                "parent_symbol": "Pool",
                "decorators": [],
                "calls": [],
            },
        ],
    )


@pytest.fixture
def utils_file() -> dict:
    """File node for utils.py with a standalone function."""
    return _make_file_node(
        "core/utils.py",
        symbols=[
            {
                "name": "validate",
                "qualified_name": "validate",
                "kind": "function",
                "visibility": "public",
                "signature": "def validate(data: dict) -> bool",
                "start_line": 1,
                "end_line": 10,
                "parent_symbol": None,
                "decorators": [],
                "calls": [],
            },
        ],
    )


# ── Symbol nodes ──────────────────────────────────────────────────


class TestBuildSymbolNodes:
    def test_creates_symbol_docs(self, pool_file):
        resolver = PythonEdgeResolver([pool_file])
        docs = resolver.build_symbol_nodes()
        assert len(docs) == 3
        keys = {d["_key"] for d in docs}
        assert symbol_key("core/db.py", "Pool") in keys
        assert symbol_key("core/db.py", "Pool.__init__") in keys
        assert symbol_key("core/db.py", "Pool.connect") in keys

    def test_symbol_doc_fields(self, pool_file):
        resolver = PythonEdgeResolver([pool_file])
        docs = resolver.build_symbol_nodes()
        pool = [d for d in docs if d["name"] == "Pool"][0]
        assert pool["kind"] == "class"
        assert pool["visibility"] == "public"
        assert pool["language"] == "python"
        assert pool["file_path"] == "core/db.py"
        assert pool["start_line"] == 10
        assert pool["end_line"] == 50

    def test_multi_file(self, pool_file, utils_file):
        resolver = PythonEdgeResolver([pool_file, utils_file])
        docs = resolver.build_symbol_nodes()
        assert len(docs) == 4  # 3 from pool + 1 from utils


# ── Edges ─────────────────────────────────────────────────────────


class TestBuildEdges:
    def test_defines_edges(self, pool_file):
        resolver = PythonEdgeResolver([pool_file])
        edges = resolver.build_edges()
        defines = [e for e in edges if e["type"] == "defines"]
        # One defines edge per symbol
        assert len(defines) == 3
        fk = file_key("core/db.py")
        for e in defines:
            assert e["_from"] == f"codebase_files/{fk}"
            assert e["_to"].startswith("codebase_symbols/")

    def test_self_call_resolved(self, pool_file):
        """self.connect() in __init__ should resolve to Pool.connect."""
        resolver = PythonEdgeResolver([pool_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        assert len(calls) == 1
        e = calls[0]
        assert e["caller"] == "Pool.__init__"
        assert e["callee"] == "self.connect"
        # Target should be Pool.connect's symbol key
        expected_to = f"codebase_symbols/{symbol_key('core/db.py', 'Pool.connect')}"
        assert e["_to"] == expected_to

    def test_cross_file_call(self, pool_file, utils_file):
        """A call to 'validate' should resolve cross-file."""
        pool_file["python_ast"]["symbols"][2]["calls"] = [
            {"name": "validate", "qualified_name": "validate"},
        ]
        resolver = PythonEdgeResolver([pool_file, utils_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        cross = [c for c in calls if c["callee"] == "validate"]
        assert len(cross) == 1
        expected_to = f"codebase_symbols/{symbol_key('core/utils.py', 'validate')}"
        assert cross[0]["_to"] == expected_to

    def test_dedup_edges(self, pool_file):
        """Duplicate edges should be deduplicated."""
        # Add a second call to the same target
        pool_file["python_ast"]["symbols"][1]["calls"] = [
            {"name": "connect", "qualified_name": "self.connect"},
            {"name": "connect", "qualified_name": "self.connect"},
        ]
        resolver = PythonEdgeResolver([pool_file])
        edges = resolver.build_edges()
        calls = [e for e in edges if e["type"] == "calls"]
        assert len(calls) == 1

    def test_empty_file_nodes(self):
        resolver = PythonEdgeResolver([])
        assert resolver.build_symbol_nodes() == []
        assert resolver.build_edges() == []
