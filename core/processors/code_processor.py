"""Code file processor for the codebase knowledge graph.

Lighter than DocumentProcessor (no Docling, no PDF assumptions).
Composes the existing CodeExtractor + TreeSitter for extraction,
then chunks by AST boundaries and optionally embeds with Jina V4
task="code" LoRA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.database.keys import file_key

logger = logging.getLogger(__name__)

# Maximum tokens per chunk before splitting (approximate, char-based)
_MAX_CHUNK_CHARS = 8000


@dataclass
class CodeChunk:
    """A single AST-aligned chunk of source code."""

    index: int
    text: str
    chunk_type: str  # "module", "function", "class"
    start_line: int
    end_line: int


@dataclass
class CodeFileResult:
    """Result of processing a single code file."""

    file_key: str
    rel_path: str
    text: str
    metadata: dict[str, Any]
    chunks: list[CodeChunk]
    symbol_hash: str | None = None
    embedding_vectors: list[list[float]] = field(default_factory=list)


class CodeProcessor:
    """Extract, chunk, and optionally embed code files.

    Uses CodeExtractor+TreeSitter for extraction, then chunks by AST
    boundaries (top-level functions and classes). Module-level code
    (imports, constants, etc.) is grouped into chunk 0.

    Args:
        embedder: Optional embedder instance with an embed_batch method.
            If None, embedding is skipped.
    """

    def __init__(self, embedder: Any = None) -> None:
        self._extractor: Any = None
        self._embedder = embedder

    def _get_extractor(self) -> Any:
        """Lazy-init the CodeExtractor to avoid import overhead."""
        if self._extractor is None:
            from core.extractors.extractors_code import CodeExtractor

            self._extractor = CodeExtractor(use_tree_sitter=True)
        return self._extractor

    def process_file(self, file_path: str | Path, repo_root: str | Path) -> CodeFileResult:
        """Process a single code file: extract, chunk by AST, optionally embed.

        Args:
            file_path: Absolute or relative path to the file.
            repo_root: Repository root for computing relative paths.

        Returns:
            CodeFileResult with text, metadata, AST-aligned chunks.
        """
        file_path = Path(file_path).resolve()
        repo_root = Path(repo_root).resolve()
        rel_path = str(file_path.relative_to(repo_root))

        extractor = self._get_extractor()
        result = extractor.extract(file_path)

        if result.error:
            logger.warning("Extraction failed for %s: %s", rel_path, result.error)
            return CodeFileResult(
                file_key=file_key(rel_path),
                rel_path=rel_path,
                text="",
                metadata={"error": result.error},
                chunks=[],
            )

        metadata = result.metadata or {}
        text = result.text or ""

        # Build AST-aligned chunks from structure
        chunks = self._chunk_by_ast(text, metadata, rel_path)

        return CodeFileResult(
            file_key=file_key(rel_path),
            rel_path=rel_path,
            text=text,
            metadata=metadata,
            chunks=chunks,
            symbol_hash=metadata.get("symbol_hash"),
        )

    def _chunk_by_ast(
        self,
        text: str,
        metadata: dict[str, Any],
        rel_path: str,
    ) -> list[CodeChunk]:
        """Split code into AST-aligned chunks.

        Strategy:
        1. Use structure.children (from TreeSitter) to find function/class
           boundaries.
        2. Module-level code (everything before the first definition, plus
           inter-definition gaps) goes into a "module" chunk.
        3. Each function/class is its own chunk.
        4. Oversized chunks are split at line boundaries.
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        structure = metadata.get("code_structure", {})
        children = structure.get("children", [])

        chunks: list[CodeChunk] = []

        if not children:
            # No AST info — single chunk for the whole file
            chunks.append(CodeChunk(
                index=0,
                text=self._with_preamble(text, rel_path),
                chunk_type="module",
                start_line=1,
                end_line=len(lines),
            ))
            return self._split_oversized(chunks, rel_path)

        # Sort children by line number
        children = sorted(children, key=lambda c: c.get("line", 0))

        # Collect module-level lines (everything not inside a function/class)
        covered = set()
        for child in children:
            start = child.get("line", 1)
            end = child.get("end_line", start)
            for i in range(start, end + 1):
                covered.add(i)

        # Module chunk = all uncovered lines
        module_lines = []
        for i, line in enumerate(lines, start=1):
            if i not in covered:
                module_lines.append(line)

        module_text = "".join(module_lines).strip()
        if module_text:
            chunks.append(CodeChunk(
                index=0,
                text=self._with_preamble(module_text, rel_path),
                chunk_type="module",
                start_line=1,
                end_line=len(lines),
            ))

        # Function/class chunks
        for child in children:
            start = child.get("line", 1)
            end = child.get("end_line", start)
            chunk_lines = lines[start - 1 : end]
            chunk_text = "".join(chunk_lines).strip()
            if chunk_text:
                idx = len(chunks)
                chunks.append(CodeChunk(
                    index=idx,
                    text=self._with_preamble(chunk_text, rel_path),
                    chunk_type=child.get("type", "unknown"),
                    start_line=start,
                    end_line=end,
                ))

        # Re-index
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return self._split_oversized(chunks, rel_path)

    def _with_preamble(self, text: str, rel_path: str) -> str:
        """Add a file-path preamble comment for context."""
        return f"# file: {rel_path}\n{text}"

    def _split_oversized(
        self,
        chunks: list[CodeChunk],
        rel_path: str,
    ) -> list[CodeChunk]:
        """Split chunks that exceed _MAX_CHUNK_CHARS."""
        result: list[CodeChunk] = []
        for chunk in chunks:
            if len(chunk.text) <= _MAX_CHUNK_CHARS:
                result.append(chunk)
                continue

            # Split at line boundaries
            lines = chunk.text.splitlines(keepends=True)
            current: list[str] = []
            current_len = 0
            line_offset = 0

            for line in lines:
                if current_len + len(line) > _MAX_CHUNK_CHARS and current:
                    result.append(CodeChunk(
                        index=len(result),
                        text="".join(current),
                        chunk_type=chunk.chunk_type,
                        start_line=chunk.start_line + line_offset - len(current),
                        end_line=chunk.start_line + line_offset - 1,
                    ))
                    current = []
                    current_len = 0

                current.append(line)
                current_len += len(line)
                line_offset += 1

            if current:
                result.append(CodeChunk(
                    index=len(result),
                    text="".join(current),
                    chunk_type=chunk.chunk_type,
                    start_line=chunk.start_line + line_offset - len(current),
                    end_line=chunk.start_line + line_offset - 1,
                ))

        # Re-index
        for i, chunk in enumerate(result):
            chunk.index = i

        return result

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed a list of code chunks using the configured embedder.

        Args:
            chunks: Chunks to embed.

        Returns:
            List of embedding vectors (same order as chunks).
            Empty list if no embedder configured.
        """
        if not self._embedder or not chunks:
            return []

        texts = [c.text for c in chunks]
        try:
            return self._embedder.embed_batch(texts, task="code")
        except Exception:
            logger.exception("Embedding failed for %d chunks", len(chunks))
            return []

    def process_batch(
        self,
        file_paths: list[Path],
        repo_root: str | Path,
    ) -> list[CodeFileResult]:
        """Process multiple files: extract all, then embed all.

        This pattern keeps GPU memory usage efficient — extraction
        doesn't need the GPU, so we batch extraction first, then
        batch-embed all chunks together.
        """
        results: list[CodeFileResult] = []
        all_chunks: list[CodeChunk] = []
        chunk_offsets: list[tuple[int, int]] = []  # (start_idx, count) per file

        # Phase 1: Extract + chunk
        for fp in file_paths:
            r = self.process_file(fp, repo_root)
            start = len(all_chunks)
            all_chunks.extend(r.chunks)
            chunk_offsets.append((start, len(r.chunks)))
            results.append(r)

        # Phase 2: Embed all at once
        if self._embedder and all_chunks:
            vectors = self.embed_chunks(all_chunks)
            if vectors and len(vectors) == len(all_chunks):
                for i, (start, count) in enumerate(chunk_offsets):
                    results[i].embedding_vectors = vectors[start : start + count]

        return results
