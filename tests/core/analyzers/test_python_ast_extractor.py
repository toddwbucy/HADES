"""Tests for the Python AST extractor.

Validates that symbols and imports are correctly extracted from Python
source code using the standard ``ast`` module.
"""

from __future__ import annotations

import textwrap

import pytest

from core.analyzers.python_ast_extractor import PythonAstExtractor


@pytest.fixture
def extractor() -> PythonAstExtractor:
    return PythonAstExtractor()


# ── Symbol extraction ─────────────────────────────────────────────


class TestFunctions:
    def test_top_level_function(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            def greet(name: str) -> str:
                return f"hello {name}"
        """))
        data = extractor.extract_file(str(src))
        syms = data["symbols"]
        assert len(syms) == 1
        s = syms[0]
        assert s["name"] == "greet"
        assert s["qualified_name"] == "greet"
        assert s["kind"] == "function"
        assert s["visibility"] == "public"
        assert "name: str" in s["signature"]
        assert "-> str" in s["signature"]
        assert s["start_line"] == 1

    def test_async_function(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            async def fetch(url: str) -> bytes:
                pass
        """))
        data = extractor.extract_file(str(src))
        s = data["symbols"][0]
        assert s["kind"] == "function"
        assert s["signature"].startswith("async def")

    def test_decorated_function(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            import functools

            @functools.cache
            def expensive(n: int) -> int:
                return n * n
        """))
        data = extractor.extract_file(str(src))
        fn = [s for s in data["symbols"] if s["kind"] == "function"][0]
        assert "functools.cache" in fn["decorators"]


class TestClasses:
    def test_class_with_methods(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class Dog:
                breed: str

                def __init__(self, name: str):
                    self.name = name

                def bark(self) -> str:
                    return "woof"

                @staticmethod
                def species() -> str:
                    return "canis"

                @classmethod
                def from_shelter(cls, id: int) -> "Dog":
                    pass

                @property
                def tag(self) -> str:
                    return self.name
        """))
        data = extractor.extract_file(str(src))
        names = {s["qualified_name"]: s for s in data["symbols"]}

        assert "Dog" in names
        assert names["Dog"]["kind"] == "class"

        assert "Dog.__init__" in names
        assert names["Dog.__init__"]["kind"] == "method"
        assert names["Dog.__init__"]["visibility"] == "dunder"

        assert "Dog.bark" in names
        assert names["Dog.bark"]["kind"] == "method"

        assert "Dog.species" in names
        assert names["Dog.species"]["kind"] == "staticmethod"

        assert "Dog.from_shelter" in names
        assert names["Dog.from_shelter"]["kind"] == "classmethod"

        assert "Dog.tag" in names
        assert names["Dog.tag"]["kind"] == "property"

    def test_class_bases(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class MyError(ValueError, RuntimeError):
                pass
        """))
        data = extractor.extract_file(str(src))
        cls = data["symbols"][0]
        assert cls["bases"] == ["ValueError", "RuntimeError"]

    def test_class_attributes(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class Config:
                MAX_RETRIES: int = 3
                timeout: float = 30.0
        """))
        data = extractor.extract_file(str(src))
        # UPPER_CASE class vars are "constant", lowercase are "attribute"
        non_class = [s for s in data["symbols"] if s["kind"] != "class"]
        names = {s["name"]: s["kind"] for s in non_class}
        assert names["MAX_RETRIES"] == "constant"
        assert names["timeout"] == "attribute"


class TestVisibility:
    def test_private_method(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class Foo:
                def __secret(self):
                    pass
        """))
        data = extractor.extract_file(str(src))
        method = [s for s in data["symbols"] if s["name"] == "__secret"][0]
        assert method["visibility"] == "private"

    def test_protected_method(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class Foo:
                def _internal(self):
                    pass
        """))
        data = extractor.extract_file(str(src))
        method = [s for s in data["symbols"] if s["name"] == "_internal"][0]
        assert method["visibility"] == "protected"


class TestConstants:
    def test_module_level_constants(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            MAX_SIZE = 1024
            DEFAULT_NAME = "unknown"
            _internal_var = True
        """))
        data = extractor.extract_file(str(src))
        names = {s["name"] for s in data["symbols"]}
        assert "MAX_SIZE" in names
        assert "DEFAULT_NAME" in names
        # lowercase module-level vars should NOT be captured
        assert "_internal_var" not in names


class TestLocalVariables:
    def test_local_vars_not_captured(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            def process():
                result = []
                count = 0
                MAX_LOCAL = 100
                for item in result:
                    count += 1
                return count
        """))
        data = extractor.extract_file(str(src))
        # Only the function itself should be a symbol, not its local vars
        assert len(data["symbols"]) == 1
        assert data["symbols"][0]["name"] == "process"


class TestCallExtraction:
    def test_simple_calls(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            def outer():
                inner()
                print("hello")
        """))
        data = extractor.extract_file(str(src))
        fn = data["symbols"][0]
        call_names = {c["name"] for c in fn.get("calls", [])}
        assert "inner" in call_names
        assert "print" in call_names

    def test_method_calls(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text(textwrap.dedent("""\
            class Foo:
                def bar(self):
                    self.baz()
                    other.method()
        """))
        data = extractor.extract_file(str(src))
        method = [s for s in data["symbols"] if s["name"] == "bar"][0]
        call_names = {c["name"] for c in method.get("calls", [])}
        assert "baz" in call_names
        assert "method" in call_names


# ── Import extraction ─────────────────────────────────────────────


class TestImports:
    def test_plain_import(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("import os\nimport sys\n")
        data = extractor.extract_file(str(src))
        imports = data["imports"]
        modules = {i["module"] for i in imports}
        assert "os" in modules
        assert "sys" in modules
        assert all(i["type"] == "import" for i in imports)

    def test_from_import(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("from pathlib import Path, PurePath\n")
        data = extractor.extract_file(str(src))
        imports = data["imports"]
        assert len(imports) == 2
        assert all(i["module"] == "pathlib" for i in imports)
        names = {i["name"] for i in imports}
        assert names == {"Path", "PurePath"}
        assert all(i["type"] == "from_import" for i in imports)

    def test_relative_import(self, extractor, tmp_path):
        src = tmp_path / "mod.py"
        src.write_text("from . import sibling\nfrom ..parent import Thing\n")
        data = extractor.extract_file(str(src))
        imports = data["imports"]
        assert len(imports) == 2
        # Relative imports should have dots prepended to module
        rel1 = [i for i in imports if i["name"] == "sibling"][0]
        assert rel1["module"] == "."
        assert rel1["level"] == 1
        rel2 = [i for i in imports if i["name"] == "Thing"][0]
        assert rel2["module"] == "..parent"
        assert rel2["level"] == 2

    def test_import_format_matches_resolver(self, extractor, tmp_path):
        """Imports must have 'module' and 'name' keys for ImportResolver."""
        src = tmp_path / "mod.py"
        src.write_text("from core.database.keys import file_key\n")
        data = extractor.extract_file(str(src))
        imp = data["imports"][0]
        assert "module" in imp
        assert "name" in imp
        assert imp["module"] == "core.database.keys"
        assert imp["name"] == "file_key"


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_syntax_error_returns_empty(self, extractor, tmp_path):
        src = tmp_path / "bad.py"
        src.write_text("def broken(\n")
        data = extractor.extract_file(str(src))
        assert data["symbols"] == []
        assert data["imports"] == []

    def test_missing_file_returns_empty(self, extractor):
        data = extractor.extract_file("/nonexistent/file.py")
        assert data["symbols"] == []

    def test_empty_file(self, extractor, tmp_path):
        src = tmp_path / "empty.py"
        src.write_text("")
        data = extractor.extract_file(str(src))
        assert data["symbols"] == []
        assert data["imports"] == []
        assert "analyzed_at" in data
