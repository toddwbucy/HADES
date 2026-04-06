"""Tests for Rust use-import extraction and edge resolution.

Part 1: _parse_use_positions — pure parsing, no LSP needed.
Part 2: Import edge resolution via RustEdgeResolver.
Part 3: Integration test with real rust-analyzer (skipped if not installed).
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from core.analyzers.rust_edge_resolver import RustEdgeResolver, symbol_key
from core.analyzers.rust_symbol_extractor import RustSymbolExtractor
from core.database.keys import file_key

# ── Part 1: _parse_use_positions ─────────────────────────────────


class TestParseUsePositions:
    """Test the static use-statement parser (no LSP needed)."""

    @staticmethod
    def _parse(line: str, line_no: int = 0) -> list[tuple[str, int, int]]:
        return RustSymbolExtractor._parse_use_positions(line, line_no)

    def test_simple_import(self) -> None:
        results = self._parse("use crate::module::Name;")
        assert len(results) == 1
        name, line, col = results[0]
        assert name == "Name"
        assert line == 0
        assert col == "use crate::module::Name;".find("Name")

    def test_aliased_import(self) -> None:
        results = self._parse("use crate::module::Name as Alias;")
        assert len(results) == 1
        name, _, col = results[0]
        assert name == "Name"
        # Should point to Name, not Alias
        assert col == "use crate::module::Name as Alias;".find("Name")

    def test_grouped_import(self) -> None:
        results = self._parse("use crate::module::{Alpha, Beta, Gamma};")
        names = {r[0] for r in results}
        assert names == {"Alpha", "Beta", "Gamma"}

    def test_grouped_with_self_skipped(self) -> None:
        results = self._parse("use crate::module::{self, Name};")
        names = {r[0] for r in results}
        assert "self" not in names
        assert "Name" in names

    def test_glob_skipped(self) -> None:
        results = self._parse("use crate::module::*;")
        assert len(results) == 0

    def test_grouped_glob_skipped(self) -> None:
        """A glob inside a group should be skipped but other items kept."""
        results = self._parse("use crate::{module::*, Name};")
        names = {r[0] for r in results}
        assert "Name" in names

    def test_nested_groups(self) -> None:
        results = self._parse("use std::{io::{Read, Write}, fmt::Display};")
        names = {r[0] for r in results}
        assert names == {"Read", "Write", "Display"}

    def test_not_a_use_statement(self) -> None:
        assert self._parse("let x = 5;") == []
        assert self._parse("// use crate::foo;") == []

    def test_line_number_preserved(self) -> None:
        results = self._parse("use crate::foo::Bar;", line_no=42)
        assert results[0][1] == 42

    def test_indented_use(self) -> None:
        """Leading whitespace should not break parsing."""
        results = self._parse("    use crate::foo::Bar;")
        assert len(results) == 1
        assert results[0][0] == "Bar"

    def test_pub_use(self) -> None:
        """pub use is not a 'use ' line (starts with 'pub'), should be skipped."""
        # pub use re-exports are different from imports — skip for now
        results = self._parse("pub use crate::foo::Bar;")
        assert len(results) == 0

    def test_single_segment(self) -> None:
        """use SomeName; (no :: path) should capture the name."""
        results = self._parse("use SomeName;")
        assert len(results) == 1
        assert results[0][0] == "SomeName"

    def test_alias_in_group(self) -> None:
        results = self._parse("use crate::{Foo as F, Bar};")
        names = {r[0] for r in results}
        assert "Foo" in names
        assert "Bar" in names
        assert "F" not in names

    def test_duplicate_name_in_group_gets_distinct_positions(self) -> None:
        """use a::{Foo, b::Foo} — each Foo should get a distinct column."""
        line = "use a::{Foo, b::Foo};"
        results = self._parse(line)
        assert len(results) == 2
        cols = [r[2] for r in results]
        assert cols[0] != cols[1], f"Both Foo got the same column: {cols}"

    def test_substring_not_matched(self) -> None:
        """use crate::add_things::add — should match 'add' not inside 'add_things'."""
        line = "use crate::add_things::add;"
        results = self._parse(line)
        assert len(results) == 1
        name, _, col = results[0]
        assert name == "add"
        # The column should point to the terminal "add", not inside "add_things"
        assert line[col:col + 3] == "add"
        assert col > line.find("add_things")


# ── Part 2: Import edge resolution ──────────────────────────────


def _make_file_node(
    rel_path: str,
    symbols: list[dict],
    imports: list[dict] | None = None,
) -> dict:
    """Build a file node with rust_analyzer attribute."""
    return {
        "rel_path": rel_path,
        "rust_analyzer": {
            "symbols": symbols,
            "impl_blocks": [],
            "pyo3_exports": [],
            "ffi_boundaries": [],
            "imports": imports or [],
            "analyzed_at": "2026-04-06T00:00:00+00:00",
        },
    }


@pytest.fixture
def math_file() -> dict:
    return _make_file_node(
        "src/math.rs",
        symbols=[
            {
                "name": "add",
                "qualified_name": "add",
                "kind": "function",
                "visibility": "pub",
                "start_line": 0,
                "end_line": 2,
            },
            {
                "name": "multiply",
                "qualified_name": "multiply",
                "kind": "function",
                "visibility": "pub",
                "start_line": 4,
                "end_line": 6,
            },
        ],
    )


@pytest.fixture
def model_file_with_imports(math_file: dict) -> dict:
    """model.rs that imports add and multiply from math.rs."""
    return _make_file_node(
        "src/model.rs",
        symbols=[
            {
                "name": "Model",
                "qualified_name": "Model",
                "kind": "struct",
                "visibility": "pub",
                "start_line": 4,
                "end_line": 7,
            },
        ],
        imports=[
            {
                "name": "add",
                "use_statement": "use crate::math::add;",
                "source_line": 0,
                "target_file": "src/math.rs",
                "target_line": 0,
                "target_name": "add",
                "qualified_name": "add",
            },
            {
                "name": "multiply",
                "use_statement": "use crate::math::multiply;",
                "source_line": 1,
                "target_file": "src/math.rs",
                "target_line": 4,
                "target_name": "multiply",
                "qualified_name": "multiply",
            },
        ],
    )


class TestImportEdgeResolution:
    """Test that import edges are correctly built from pre-extracted import data."""

    def test_import_edges_created(self, math_file: dict, model_file_with_imports: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file_with_imports])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]
        assert len(imports) == 2

    def test_import_edge_source_is_file(self, math_file: dict, model_file_with_imports: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file_with_imports])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]
        fk = file_key("src/model.rs")
        for e in imports:
            assert e["_from"] == f"codebase_files/{fk}"

    def test_import_edge_target_is_symbol(self, math_file: dict, model_file_with_imports: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file_with_imports])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]

        expected_targets = {
            f"codebase_symbols/{symbol_key('src/math.rs', 'add')}",
            f"codebase_symbols/{symbol_key('src/math.rs', 'multiply')}",
        }
        actual_targets = {e["_to"] for e in imports}
        assert actual_targets == expected_targets

    def test_import_edge_metadata(self, math_file: dict, model_file_with_imports: dict) -> None:
        resolver = RustEdgeResolver([math_file, model_file_with_imports])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]
        add_edge = next(e for e in imports if e["import_name"] == "add")
        assert add_edge["use_statement"] == "use crate::math::add;"
        assert add_edge["source_line"] == 0

    def test_no_duplicate_import_edges(self, math_file: dict) -> None:
        """Same import appearing twice should be deduplicated."""
        dup_imports = _make_file_node(
            "src/user.rs",
            symbols=[],
            imports=[
                {
                    "name": "add",
                    "use_statement": "use crate::math::add;",
                    "source_line": 0,
                    "target_file": "src/math.rs",
                    "target_line": 0,
                    "target_name": "add",
                    "qualified_name": "add",
                },
                {
                    "name": "add",
                    "use_statement": "use crate::math::add;",
                    "source_line": 1,
                    "target_file": "src/math.rs",
                    "target_line": 0,
                    "target_name": "add",
                    "qualified_name": "add",
                },
            ],
        )
        resolver = RustEdgeResolver([math_file, dup_imports])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]
        assert len(imports) == 1

    def test_unresolvable_import_skipped(self) -> None:
        """Imports pointing to files not in the file_nodes should be skipped."""
        node = _make_file_node(
            "src/main.rs",
            symbols=[],
            imports=[
                {
                    "name": "Ghost",
                    "use_statement": "use crate::ghost::Ghost;",
                    "source_line": 0,
                    "target_file": "src/ghost.rs",
                    "target_line": 0,
                    "target_name": "Ghost",
                    "qualified_name": "Ghost",
                },
            ],
        )
        resolver = RustEdgeResolver([node])
        edges = resolver.build_edges()
        imports = [e for e in edges if e["type"] == "imports"]
        assert len(imports) == 0

    def test_import_and_call_edges_coexist(self, math_file: dict) -> None:
        """A file can have both imports and call edges."""
        caller = _make_file_node(
            "src/caller.rs",
            symbols=[
                {
                    "name": "do_math",
                    "qualified_name": "do_math",
                    "kind": "function",
                    "visibility": "pub",
                    "start_line": 3,
                    "end_line": 6,
                    "calls": [
                        {"qualified_name": "add", "name": "add", "file": "src/math.rs", "line": 4},
                    ],
                },
            ],
            imports=[
                {
                    "name": "add",
                    "use_statement": "use crate::math::add;",
                    "source_line": 0,
                    "target_file": "src/math.rs",
                    "target_line": 0,
                    "target_name": "add",
                    "qualified_name": "add",
                },
            ],
        )
        resolver = RustEdgeResolver([math_file, caller])
        edges = resolver.build_edges()

        edge_types = {e["type"] for e in edges}
        assert "imports" in edge_types
        assert "calls" in edge_types
        assert "defines" in edge_types


# ── Part 3: Integration with rust-analyzer ───────────────────────

HAS_RUST_ANALYZER = shutil.which("rust-analyzer") is not None


@pytest.mark.skipif(not HAS_RUST_ANALYZER, reason="rust-analyzer not installed")
class TestUseImportIntegration:
    """End-to-end test: extract real use-imports from a Rust crate."""

    @pytest.fixture
    def rust_crate(self, tmp_path: Path) -> Path:
        crate = tmp_path / "import_test_crate"
        crate.mkdir()
        (crate / "Cargo.toml").write_text(textwrap.dedent("""\
            [package]
            name = "import_test_crate"
            version = "0.1.0"
            edition = "2021"
        """))
        src = crate / "src"
        src.mkdir()

        (src / "lib.rs").write_text(textwrap.dedent("""\
            pub mod math;
            pub mod consumer;
        """))

        (src / "math.rs").write_text(textwrap.dedent("""\
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }

            pub struct Calculator {
                pub total: i32,
            }
        """))

        (src / "consumer.rs").write_text(textwrap.dedent("""\
            use crate::math::add;
            use crate::math::Calculator;

            pub fn compute() -> i32 {
                let c = Calculator { total: 0 };
                add(c.total, 1)
            }
        """))

        return crate

    @pytest.fixture
    def session(self, rust_crate: Path):
        from core.analyzers.rust_analyzer_client import RustAnalyzerSession
        with RustAnalyzerSession(rust_crate, timeout=60) as s:
            yield s

    @pytest.fixture
    def extractor(self, session):
        return RustSymbolExtractor(session, include_calls=True, include_incoming=False)

    def test_imports_field_present(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/consumer.rs")
        assert "imports" in data

    def test_internal_imports_resolved(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/consumer.rs")
        imports = data["imports"]
        # Should have at least one internal import (add or Calculator)
        assert len(imports) > 0
        names = {imp["name"] for imp in imports}
        # At least one of the two use statements should resolve
        assert names & {"add", "Calculator"}, f"Expected add or Calculator in {names}"

    def test_import_has_target_file(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/consumer.rs")
        for imp in data["imports"]:
            assert imp["target_file"], f"Import {imp['name']} has no target_file"
            assert "math.rs" in imp["target_file"]

    def test_full_pipeline_edges(self, extractor: RustSymbolExtractor) -> None:
        """Extract all files, then build edges — imports should appear."""
        results = extractor.extract_crate(["src/lib.rs", "src/math.rs", "src/consumer.rs"])
        file_nodes = [
            {"rel_path": path, "rust_analyzer": data}
            for path, data in results.items()
        ]

        resolver = RustEdgeResolver(file_nodes)
        edges = resolver.build_edges()

        import_edges = [e for e in edges if e["type"] == "imports"]
        assert len(import_edges) > 0, "Expected at least one import edge from consumer.rs"

        # Import edges should come from consumer.rs
        consumer_fk = file_key("src/consumer.rs")
        consumer_imports = [e for e in import_edges if consumer_fk in e["_from"]]
        assert len(consumer_imports) > 0
