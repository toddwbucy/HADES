"""LaTeX extraction backend — pure text parsing, no ML.

Handles .tex files and .tar.gz arXiv source packages.
Extracts equations, tables, citations, and section structure.
"""

from __future__ import annotations

import gzip
import logging
import re
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from .docling_backend import ExtractionResult

logger = logging.getLogger(__name__)


class LaTeXExtractor:
    """Extract structured content from LaTeX source files."""

    SUPPORTED_EXTENSIONS = {".tex", ".gz", ".tar.gz"}

    # Tar bomb limits
    MAX_TAR_MEMBERS = 500
    MAX_TAR_EXPANDED_BYTES = 200 * 1024 * 1024  # 200 MiB of extracted files
    MAX_SOURCE_BYTES = 200 * 1024 * 1024  # decoded only after this byte check
    MAX_TAR_STREAM_BYTES = MAX_TAR_EXPANDED_BYTES + 1024 * 1024  # headers/padding included

    MAX_TAR_HEADER_BYTES = 1024 * 1024
    MAX_TAR_METADATA_BYTES = 4 * 1024 * 1024

    def _bounded_tar_info(self) -> type[tarfile.TarInfo]:
        # A fresh class gives each archive its own counters. _proc_member runs before
        # tarfile processes PAX/GNU payloads, including on older Python releases.
        extractor = self

        class BoundedTarInfo(tarfile.TarInfo):
            header_count = 0
            metadata_bytes = 0

            def _proc_member(member, archive):
                cls = type(member)
                cls.header_count += 1
                if cls.header_count > extractor.MAX_TAR_MEMBERS:
                    raise ValueError(f"Archive member count exceeds limit of {extractor.MAX_TAR_MEMBERS}")
                if member.size < 0:
                    raise ValueError("Archive contains a negative member size")
                if member.type in (
                    tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE,
                    tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                ):
                    if member.size > extractor.MAX_TAR_HEADER_BYTES:
                        raise ValueError("Archive extended header exceeds byte limit")
                    cls.metadata_bytes += member.size
                    if cls.metadata_bytes > extractor.MAX_TAR_METADATA_BYTES:
                        raise ValueError("Archive metadata exceeds byte limit")
                return super()._proc_member(archive)

        return BoundedTarInfo

    @staticmethod
    def _read_source(stream: Any, limit: int) -> str:
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f"Source expanded size exceeds limit of {limit} bytes")
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _copy_archive(source: Any, destination: Any, limit: int) -> None:
        total = 0
        while block := source.read(min(64 * 1024, limit - total + 1)):
            total += len(block)
            if total > limit:
                raise ValueError(f"Archive expanded stream exceeds limit of {limit} bytes")
            destination.write(block)

    def extract(
        self,
        file_path: str | Path,
        *,
        extract_tables: bool = True,
        extract_equations: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        path = Path(file_path)
        start = time.time()

        if not path.exists():
            return ExtractionResult(error=f"File not found: {path}")

        suffix = path.suffix.lower()
        name = path.name.lower()

        try:
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                return self._extract_tar_gz(path, start, extract_tables, extract_equations)
            elif suffix == ".gz":
                return self._extract_plain_gz(path, start, extract_tables, extract_equations)
            elif suffix == ".tex":
                return self._extract_tex(path, start, extract_tables, extract_equations)
            else:
                return ExtractionResult(error=f"Unsupported LaTeX format: {suffix}")
        except Exception as e:
            return ExtractionResult(
                error=f"LaTeX extraction failed: {e}",
                processing_time=time.time() - start,
            )

    def _extract_tar_gz(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from arXiv .tar.gz source package."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Bound decompression on disk, then validate each raw header before
            # tarfile reads archive-controlled metadata lengths.
            archive = tmp_path / "source.tar"
            with gzip.open(path, "rb") as source, archive.open("wb") as destination:
                self._copy_archive(source, destination, self.MAX_TAR_STREAM_BYTES)
            extracted = tmp_path / "files"
            with tarfile.open(archive, "r:", tarinfo=self._bounded_tar_info()) as tar:
                safe = []
                file_paths = set()
                total_bytes = 0
                for count, member in enumerate(tar, 1):
                    # Count unsafe entries too: filtering is not an admission limit.
                    if count > self.MAX_TAR_MEMBERS:
                        raise ValueError(f"Archive member count exceeds limit of {self.MAX_TAR_MEMBERS}")
                    if not (member.isreg() or member.isdir()) or not self._is_safe(member, extracted):
                        continue
                    if member.isreg():
                        canonical = (extracted / member.name).resolve()
                        if canonical in file_paths:
                            raise ValueError("Archive contains duplicate source paths")
                        file_paths.add(canonical)
                    total_bytes += member.size if member.isreg() else 0
                    if total_bytes > self.MAX_TAR_EXPANDED_BYTES:
                        raise ValueError(f"Archive expanded size exceeds limit of {self.MAX_TAR_EXPANDED_BYTES} bytes")
                    safe.append(member)
                # Only the largest regular .tex member is used. Read it directly
                # instead of writing archive-controlled paths or permissions.
                tex_files = [member for member in safe if member.isreg() and member.name.endswith(".tex")]
                if not tex_files:
                    return ExtractionResult(
                        error="No .tex files found in archive",
                        processing_time=time.time() - start,
                    )
                main_tex = max(tex_files, key=lambda member: member.size)
                source = tar.extractfile(main_tex)
                if source is None:
                    raise ValueError("Archive main source is not readable")
                with source:
                    latex = self._read_source(source, self.MAX_SOURCE_BYTES)

            return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _extract_plain_gz(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from gzipped .tex file."""
        with gzip.open(path, "rb") as f:
            latex = self._read_source(f, self.MAX_SOURCE_BYTES)
        return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _extract_tex(self, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Extract from plain .tex file."""
        with path.open("rb") as source:
            latex = self._read_source(source, self.MAX_SOURCE_BYTES)
        return self._build_result(latex, path, start, extract_tables, extract_equations)

    def _build_result(self, latex: str, path: Path, start: float, extract_tables: bool = True, extract_equations: bool = True) -> ExtractionResult:
        """Build structured result from LaTeX source."""
        # Strip LaTeX commands for plain text (rough)
        text = self._strip_commands(latex)

        equations = self._extract_equations(latex) if extract_equations else []
        tables = self._extract_tables(latex) if extract_tables else []
        sections = self._extract_sections(latex)

        metadata = {
            "source": str(path),
            "format": "latex",
            "extractor": "latex_native",
            "num_equations": str(len(equations)),
            "num_tables": str(len(tables)),
            "num_sections": str(len(sections)),
        }

        return ExtractionResult(
            text=text,
            equations=equations,
            tables=tables,
            metadata=metadata,
            processing_time=time.time() - start,
        )

    @staticmethod
    def _is_safe(member: tarfile.TarInfo, target: Path) -> bool:
        """Reject symlinks, absolute paths, path traversal."""
        if member.issym() or member.islnk():
            return False
        if member.name.startswith("/") or ":" in member.name[:3]:
            return False
        target_resolved = target.resolve()
        resolved = (target / member.name).resolve()
        try:
            resolved.relative_to(target_resolved)
            return True
        except ValueError:
            return False

    @staticmethod
    def _strip_commands(latex: str) -> str:
        """Rough LaTeX → plain text (for full_text field)."""
        text = re.sub(r"\\begin\{.*?\}", "", latex)
        text = re.sub(r"\\end\{.*?\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[.*?\])?(?:\{.*?\})?", "", text)
        text = re.sub(r"[{}]", "", text)
        text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_equations(latex: str) -> list[dict[str, Any]]:
        """Extract equations from LaTeX source."""
        equations: list[dict[str, Any]] = []
        idx = 0

        # Display equations: \[ ... \], equation, align, etc.
        for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
            for m in re.finditer(
                rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
                latex,
                re.DOTALL,
            ):
                equations.append({
                    "latex": m.group(1).strip(),
                    "text": "",
                    "index": idx,
                    "is_inline": False,
                })
                idx += 1

        # Bracket display math: \[ ... \]
        for m in re.finditer(r"\\\[(.*?)\\\]", latex, re.DOTALL):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": False,
            })
            idx += 1

        # Inline math: \( ... \)
        for m in re.finditer(r"\\\((.*?)\\\)", latex, re.DOTALL):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": True,
            })
            idx += 1

        # Inline math: $...$ (but not $$...$$, and not escaped \$)
        for m in re.finditer(r"(?<!\$)(?<!\\)\$((?!\$).+?)\$(?!\$)", latex):
            equations.append({
                "latex": m.group(1).strip(),
                "text": "",
                "index": idx,
                "is_inline": True,
            })
            idx += 1

        return equations

    @classmethod
    def _extract_tables(cls, latex: str) -> list[dict[str, Any]]:
        """Extract tables from LaTeX source."""
        tables: list[dict[str, Any]] = []

        for i, m in enumerate(
            re.finditer(
                r"\\begin\{table\}(.*?)\\end\{table\}",
                latex,
                re.DOTALL,
            )
        ):
            content = m.group(1)
            caption = cls._extract_braced(content, r"\caption")

            tables.append({
                "content": content.strip(),
                "caption": caption,
                "index": i,
            })

        return tables

    @classmethod
    def _extract_sections(cls, latex: str) -> list[dict[str, Any]]:
        """Extract section headings."""
        sections: list[dict[str, Any]] = []
        level_map = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}

        for m in re.finditer(
            r"\\(section|subsection|subsubsection|paragraph)\*?\{",
            latex,
        ):
            cmd = m.group(1)
            title = cls._extract_braced(latex[m.start():], f"\\{cmd}")
            if not title:
                title = cls._extract_braced(latex[m.start():], f"\\{cmd}*")
            sections.append({
                "level": level_map.get(cmd, 0),
                "title": title,
            })

        return sections

    @staticmethod
    def _extract_braced(text: str, command: str) -> str:
        """Extract brace-delimited argument from a LaTeX command, handling nesting."""
        idx = text.find(command)
        if idx == -1:
            return ""
        # Find the opening brace after the command
        brace_start = text.find("{", idx + len(command))
        if brace_start == -1:
            return ""
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : i]
        return ""
