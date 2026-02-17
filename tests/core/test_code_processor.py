"""Tests for codebase knowledge graph: code processor, import resolver, keys, collections."""

import pytest

from core.database.keys import file_key

# ── file_key tests ──────────────────────────────────────────────


class TestFileKey:
    def test_basic_path(self):
        assert file_key("core/persephone/models.py") == "core_persephone_models_py"

    def test_root_file(self):
        assert file_key("setup.py") == "setup_py"

    def test_deeply_nested(self):
        assert file_key("core/cli/commands/codebase.py") == "core_cli_commands_codebase_py"

    def test_init_file(self):
        assert file_key("core/__init__.py") == "core___init___py"


# ── ImportResolver tests ────────────────────────────────────────


class TestImportResolver:
    @pytest.fixture()
    def resolver(self):
        from core.database.import_resolver import ImportResolver

        known = {
            "core/persephone/models.py",
            "core/persephone/tasks.py",
            "core/persephone/sessions.py",
            "core/persephone/__init__.py",
            "core/database/keys.py",
            "core/cli/main.py",
        }
        return ImportResolver("/repo", known)

    def test_resolve_absolute_from_import(self, resolver):
        imp = {"module": "core.persephone.models", "name": "TaskCreate", "type": "from_import"}
        edge = resolver.resolve_import(imp, "core/persephone/tasks.py")
        assert edge is not None
        assert edge["_to_key"] == "core_persephone_models_py"
        assert edge["type"] == "imports"

    def test_resolve_absolute_import(self, resolver):
        imp = {"module": "", "name": "core.database.keys", "type": "import"}
        edge = resolver.resolve_import(imp, "core/cli/main.py")
        assert edge is not None
        assert edge["_to_key"] == "core_database_keys_py"

    def test_external_import_returns_none(self, resolver):
        imp = {"module": "pydantic", "name": "BaseModel", "type": "from_import"}
        edge = resolver.resolve_import(imp, "core/persephone/models.py")
        assert edge is None

    def test_self_import_returns_none(self, resolver):
        imp = {"module": "core.persephone.models", "name": "X", "type": "from_import"}
        edge = resolver.resolve_import(imp, "core/persephone/models.py")
        assert edge is None

    def test_relative_import(self, resolver):
        imp = {"module": ".models", "name": "TaskCreate", "type": "from_import"}
        edge = resolver.resolve_import(imp, "core/persephone/tasks.py")
        assert edge is not None
        assert edge["_to_key"] == "core_persephone_models_py"

    def test_resolve_all_deduplicates(self, resolver):
        files = [
            {
                "rel_path": "core/persephone/tasks.py",
                "symbols": {
                    "imports": [
                        {"module": "core.persephone.models", "name": "TaskCreate", "type": "from_import"},
                        {"module": "core.persephone.models", "name": "TaskUpdate", "type": "from_import"},
                    ]
                },
            },
        ]
        edges = resolver.resolve_all(files)
        # Same source→target should produce only 1 edge
        assert len(edges) == 1

    def test_package_import(self, resolver):
        imp = {"module": "core.persephone", "name": "", "type": "import"}
        edge = resolver.resolve_import(imp, "core/cli/main.py")
        assert edge is not None
        assert edge["_to_key"] == "core_persephone___init___py"


# ── CodeProcessor tests ─────────────────────────────────────────


class TestCodeProcessor:
    @pytest.fixture()
    def processor(self):
        from core.processors.code_processor import CodeProcessor

        return CodeProcessor(embedder=None)

    def test_chunk_by_ast_no_structure(self, processor):
        """Files without AST structure get a single module chunk."""
        chunks = processor._chunk_by_ast("x = 1\ny = 2\n", {}, "test.py")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "module"
        assert "# file: test.py" in chunks[0].text

    def test_chunk_by_ast_with_structure(self, processor):
        """Files with function/class structure get multiple chunks."""
        code = "import os\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
        metadata = {
            "code_structure": {
                "type": "module",
                "children": [
                    {"type": "function", "line": 3, "end_line": 4},
                    {"type": "class", "line": 6, "end_line": 7},
                ],
            }
        }
        chunks = processor._chunk_by_ast(code, metadata, "test.py")
        # Should have: module chunk (import os), function chunk, class chunk
        assert len(chunks) >= 2
        types = [c.chunk_type for c in chunks]
        assert "function" in types
        assert "class" in types

    def test_preamble_added(self, processor):
        text = processor._with_preamble("def foo(): pass", "core/test.py")
        assert text.startswith("# file: core/test.py\n")

    def test_empty_text_no_chunks(self, processor):
        chunks = processor._chunk_by_ast("", {}, "empty.py")
        assert chunks == []


# ── CodebaseCollections tests ───────────────────────────────────


class TestCodebaseCollections:
    def test_collection_names(self):
        from core.database.codebase_collections import CODEBASE_COLLECTIONS

        assert CODEBASE_COLLECTIONS.files == "codebase_files"
        assert CODEBASE_COLLECTIONS.chunks == "codebase_chunks"
        assert CODEBASE_COLLECTIONS.embeddings == "codebase_embeddings"
        assert CODEBASE_COLLECTIONS.edges == "codebase_edges"

    def test_frozen(self):
        from core.database.codebase_collections import CODEBASE_COLLECTIONS

        with pytest.raises(AttributeError):
            CODEBASE_COLLECTIONS.files = "other"  # type: ignore[misc]
