"""Python import resolution for the codebase knowledge graph.

Resolves Python import statements (extracted by TreeSitter) into
file-path-based edges. Only internal imports (files in the repo)
produce edges; external packages are silently skipped.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

from core.database.keys import file_key

logger = logging.getLogger(__name__)


class ImportResolver:
    """Resolve Python imports to repo-internal file paths.

    Args:
        repo_root: Absolute path to the repository root.
        known_rel_paths: Set of relative file paths that exist in the repo.
            Used to distinguish internal from external imports.
    """

    def __init__(self, repo_root: str, known_rel_paths: set[str]) -> None:
        self._repo_root = repo_root
        self._known = known_rel_paths
        # Pre-build a lookup from dotted module path → rel_path
        self._module_index: dict[str, str] = {}
        for rel in known_rel_paths:
            if rel.endswith(".py"):
                # core/persephone/models.py → core.persephone.models
                dotted = rel[:-3].replace("/", ".")
                self._module_index[dotted] = rel
                # Also handle __init__.py → package name
                if rel.endswith("/__init__.py"):
                    pkg_dotted = rel[: -len("/__init__.py")].replace("/", ".")
                    self._module_index[pkg_dotted] = rel

    def resolve_import(
        self,
        import_dict: dict[str, Any],
        source_rel_path: str,
    ) -> dict[str, Any] | None:
        """Resolve a single import to an edge dict.

        Args:
            import_dict: Import info from TreeSitter symbols.imports.
                Expected keys: "module" (str), optionally "name" (str),
                "type" ("import" | "from_import").
            source_rel_path: Relative path of the importing file.

        Returns:
            Edge dict with _from, _to, type, module fields,
            or None if the import is external.
        """
        module = import_dict.get("module", "")
        if not module:
            # Plain "import X" — module is in "name"
            module = import_dict.get("name", "")
        if not module:
            return None

        # Handle relative imports: "from .models import X"
        if module.startswith("."):
            module = self._resolve_relative(module, source_rel_path)
            if not module:
                return None

        # Try direct module match
        target_rel = self._module_index.get(module)

        if not target_rel:
            # Try parent module (e.g., "from core.persephone.models import TaskCreate"
            # where module is "core.persephone.models")
            parts = module.split(".")
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                target_rel = self._module_index.get(candidate)
                if target_rel:
                    break

        if not target_rel or target_rel == source_rel_path:
            return None

        source_fk = file_key(source_rel_path)
        target_fk = file_key(target_rel)

        return {
            "_from_key": source_fk,
            "_to_key": target_fk,
            "type": "imports",
            "module": module,
            "source": source_rel_path,
            "target": target_rel,
        }

    def _resolve_relative(self, module: str, source_rel_path: str) -> str | None:
        """Convert a relative import to an absolute dotted module path.

        Example: source_rel_path="core/persephone/tasks.py", module=".models"
        → "core.persephone.models"
        """
        # Count leading dots
        dots = 0
        for ch in module:
            if ch == ".":
                dots += 1
            else:
                break
        remainder = module[dots:]

        source_parts = PurePosixPath(source_rel_path).parts
        # Drop filename → package parts
        pkg_parts = list(source_parts[:-1])

        # Each dot beyond the first goes up one package level
        levels_up = dots - 1
        if levels_up > 0:
            if levels_up > len(pkg_parts):
                return None
            pkg_parts = pkg_parts[:-levels_up]

        if remainder:
            pkg_parts.append(remainder.replace(".", "/"))

        absolute = ".".join(str(p) for p in pkg_parts)
        return absolute if absolute else None

    def resolve_all(
        self,
        files_metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve all imports across all files.

        Args:
            files_metadata: List of dicts, each with "rel_path" and
                "symbols" (containing "imports" list from TreeSitter).

        Returns:
            List of edge dicts suitable for batch insertion.
        """
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for file_meta in files_metadata:
            rel_path = file_meta.get("rel_path", "")
            symbols = file_meta.get("symbols", {})
            imports = symbols.get("imports", [])

            for imp in imports:
                edge = self.resolve_import(imp, rel_path)
                if edge is None:
                    continue

                pair = (edge["_from_key"], edge["_to_key"])
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append(edge)

        logger.info("Resolved %d internal import edges from %d files", len(edges), len(files_metadata))
        return edges
