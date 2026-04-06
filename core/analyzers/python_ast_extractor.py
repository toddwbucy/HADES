"""Python AST symbol extraction via the standard ``ast`` module.

Extracts rich symbol data from Python files and structures it for storage
as the ``python_ast`` attribute on codebase_files nodes.  This is the
Python counterpart of ``RustSymbolExtractor`` (which uses rust-analyzer).

The ``ast`` module gives us a *semantic* AST — scopes, name resolution,
decorators, base classes — which tree-sitter cannot provide.  Zero
external dependencies.

Usage:
    extractor = PythonAstExtractor()
    file_data = extractor.extract_file("core/database/pool.py", repo_root)
    # file_data is the dict to store as file_node["python_ast"]
"""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _visibility(name: str) -> str:
    """Derive Python visibility from naming convention.

    - ``__name__`` → dunder (magic / special)
    - ``__name``   → private (name-mangled)
    - ``_name``    → protected (conventional)
    - ``name``     → public
    """
    if name.startswith("__") and name.endswith("__"):
        return "dunder"
    if name.startswith("__"):
        return "private"
    if name.startswith("_"):
        return "protected"
    return "public"


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names as strings."""
    names: list[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(ast.unparse(dec))
        elif isinstance(dec, ast.Call):
            # e.g. @decorator(args)
            names.append(ast.unparse(dec.func))
        else:
            names.append(ast.unparse(dec))
    return names


def _base_names(node: ast.ClassDef) -> list[str]:
    """Extract base class names."""
    return [ast.unparse(b) for b in node.bases]


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a human-readable signature string."""
    args = node.args
    parts: list[str] = []

    # Regular positional args
    for arg in args.args:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")

    # *args
    if args.vararg:
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")

    # keyword-only args
    for arg in args.kwonlyargs:
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}")

    # **kwargs
    if args.kwarg:
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)}){ret}"


def _extract_calls(body: list[ast.stmt]) -> list[dict[str, str]]:
    """Walk a function body and extract call targets.

    Only collects calls at the current scope — does NOT descend into
    nested function/class definitions (those belong to the nested symbol).

    Returns a list of dicts with ``name`` and optionally ``qualified_name``
    (for attribute calls like ``self.foo()`` or ``module.func()``).
    """
    calls: list[dict[str, str]] = []
    seen: set[str] = set()

    def _visit(nodes: list[ast.AST]) -> None:
        for node in nodes:
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                    if name not in seen:
                        seen.add(name)
                        calls.append({"name": name, "qualified_name": name})
                elif isinstance(func, ast.Attribute):
                    attr = func.attr
                    try:
                        full = ast.unparse(func)
                    except Exception:
                        full = attr
                    if full not in seen:
                        seen.add(full)
                        calls.append({"name": attr, "qualified_name": full})
            # Skip nested definitions — their calls belong to their own symbol
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # Recurse into child nodes
            for child in ast.iter_child_nodes(node):
                _visit([child])

    _visit(body)
    return calls


class PythonAstExtractor:
    """Extract structured symbol data from Python files via ``ast``.

    Produces a dict suitable for storing as the ``python_ast`` attribute
    on a codebase_files document — structurally parallel to the
    ``rust_analyzer`` attribute produced by ``RustSymbolExtractor``.
    """

    def extract_file(
        self,
        file_path: str | Path,
        repo_root: str | Path | None = None,
        file_content: str | None = None,
    ) -> dict[str, Any]:
        """Extract all symbol data for a single Python file.

        Args:
            file_path: Path to the .py file (absolute or relative).
            repo_root: Repository root for computing relative paths.
            file_content: Optional file content; read from disk if omitted.

        Returns:
            Dict suitable for ``file_node["python_ast"]``::

                {
                    "symbols": [...],
                    "imports": [...],
                    "analyzed_at": "..."
                }
        """
        path = Path(file_path)
        if repo_root and not path.is_absolute():
            path = Path(repo_root) / path

        if file_content is None:
            try:
                file_content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Cannot read %s", path)
                return self._empty_result()

        try:
            tree = ast.parse(file_content, filename=str(path))
        except SyntaxError as exc:
            logger.warning("Syntax error in %s: %s", path, exc)
            return self._empty_result()

        symbols = self._walk_module(tree, file_content)
        imports = self._extract_imports(tree)

        return {
            "symbols": symbols,
            "imports": imports,
            "analyzed_at": datetime.now(UTC).isoformat(),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _walk_module(
        self,
        tree: ast.Module,
        source: str,
    ) -> list[dict[str, Any]]:
        """Walk the AST and extract all symbol definitions."""
        symbols: list[dict[str, Any]] = []
        self._walk_body(tree.body, parent_name=None, symbols=symbols, source=source, in_function=False)
        return symbols

    def _walk_body(
        self,
        body: list[ast.stmt],
        parent_name: str | None,
        symbols: list[dict[str, Any]],
        source: str,
        *,
        in_function: bool = False,
    ) -> None:
        """Recursively walk a body (module, class, function) for definitions.

        Assignments are only extracted at module and class scope — local
        variables inside functions are not symbols.
        """
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(node, parent_name, symbols, source)

            elif isinstance(node, ast.ClassDef):
                self._process_class(node, parent_name, symbols, source)

            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not in_function:
                self._process_assignment(node, parent_name, symbols)

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_name: str | None,
        symbols: list[dict[str, Any]],
        source: str,
    ) -> None:
        name = node.name
        qualified = f"{parent_name}.{name}" if parent_name else name
        kind = "method" if parent_name else "function"

        # Detect static/class methods
        decs = _decorator_names(node)
        if "staticmethod" in decs:
            kind = "staticmethod"
        elif "classmethod" in decs:
            kind = "classmethod"
        elif "property" in decs:
            kind = "property"

        sym: dict[str, Any] = {
            "name": name,
            "qualified_name": qualified,
            "kind": kind,
            "visibility": _visibility(name),
            "signature": _function_signature(node),
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "parent_symbol": parent_name,
            "decorators": decs,
            "calls": _extract_calls(node.body),
        }
        symbols.append(sym)

        # Walk nested definitions (closures, inner classes)
        self._walk_body(node.body, qualified, symbols, source, in_function=True)

    def _process_class(
        self,
        node: ast.ClassDef,
        parent_name: str | None,
        symbols: list[dict[str, Any]],
        source: str,
    ) -> None:
        name = node.name
        qualified = f"{parent_name}.{name}" if parent_name else name
        decs = _decorator_names(node)
        bases = _base_names(node)

        sym: dict[str, Any] = {
            "name": name,
            "qualified_name": qualified,
            "kind": "class",
            "visibility": _visibility(name),
            "signature": f"class {name}({', '.join(bases)})" if bases else f"class {name}",
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "parent_symbol": parent_name,
            "decorators": decs,
            "bases": bases,
        }
        symbols.append(sym)

        # Walk class body for methods, nested classes, class variables
        self._walk_body(node.body, qualified, symbols, source, in_function=False)

    def _process_assignment(
        self,
        node: ast.Assign | ast.AnnAssign,
        parent_name: str | None,
        symbols: list[dict[str, Any]],
    ) -> None:
        """Extract module-level constants and class-level attributes."""
        targets: list[ast.expr] = []
        if isinstance(node, ast.AnnAssign) and node.target:
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets

        for target in targets:
            if not isinstance(target, ast.Name):
                continue

            name = target.id
            # Only capture UPPER_CASE (constants) at module level,
            # or all assignments at class level
            if parent_name is None and not name.isupper():
                continue

            qualified = f"{parent_name}.{name}" if parent_name else name
            kind = "constant" if name.isupper() else "attribute"

            # Build a signature from annotation if available
            sig = name
            if isinstance(node, ast.AnnAssign) and node.annotation:
                sig = f"{name}: {ast.unparse(node.annotation)}"

            sym: dict[str, Any] = {
                "name": name,
                "qualified_name": qualified,
                "kind": kind,
                "visibility": _visibility(name),
                "signature": sig,
                "start_line": node.lineno,
                "end_line": node.end_lineno or node.lineno,
                "parent_symbol": parent_name,
            }
            symbols.append(sym)

    def _extract_imports(self, tree: ast.Module) -> list[dict[str, Any]]:
        """Extract structured import information.

        Returns dicts compatible with ``ImportResolver.resolve_import()``:
        ``{"module": "...", "name": "...", "type": "...", "line": N, "level": N}``
        """
        imports: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "name": alias.asname or alias.name,
                        "type": "import",
                        "line": node.lineno,
                        "level": 0,
                    })

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0

                # For relative imports, prepend dots to module
                if level > 0:
                    module = "." * level + module

                for alias in node.names:
                    imports.append({
                        "module": module,
                        "name": alias.name,
                        "type": "from_import",
                        "line": node.lineno,
                        "level": level,
                    })

        return imports

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "symbols": [],
            "imports": [],
            "analyzed_at": datetime.now(UTC).isoformat(),
        }
