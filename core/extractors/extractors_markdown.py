#!/usr/bin/env python3
"""Markdown and HTML Extractor.

Extracts structured content from Markdown and HTML documents,
including headers, code blocks, tables, and links.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

from .extractors_base import ExtractionResult, ExtractorBase, ExtractorConfig

logger = logging.getLogger(__name__)


class MarkdownExtractor(ExtractorBase):
    """Extractor for Markdown and HTML documents.

    Parses markdown/HTML structure to extract:
    - Plain text content
    - Code blocks with language detection
    - Tables
    - Links/references
    - Headers for document structure

    Example:
        extractor = MarkdownExtractor()
        result = extractor.extract("README.md")
        print(result.text)
        print(result.code_blocks)
    """

    def __init__(
        self,
        config: ExtractorConfig | None = None,
        extract_html: bool = True,
        **kwargs,
    ):
        """Initialize the Markdown extractor.

        Args:
            config: Extractor configuration
            extract_html: Whether to also support HTML files
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(config)
        self.extract_html = extract_html

    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        formats = [".md", ".markdown", ".mdown", ".mkd"]
        if self.extract_html:
            formats.extend([".html", ".htm"])
        return formats

    def extract(self, file_path: str | Path, **kwargs) -> ExtractionResult:
        """Extract content from a Markdown or HTML file.

        Args:
            file_path: Path to the document
            **kwargs: Additional extraction options

        Returns:
            ExtractionResult with extracted content
        """
        start_time = time.time()
        path = Path(file_path)

        if not self.validate_file(path):
            return ExtractionResult(
                text="",
                error=f"Invalid file: {path}",
                processing_time=time.time() - start_time,
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                return ExtractionResult(
                    text="",
                    error=f"Failed to read file: {e}",
                    processing_time=time.time() - start_time,
                )

        ext = path.suffix.lower()
        if ext in [".html", ".htm"]:
            if not self.extract_html:
                return ExtractionResult(
                    text="",
                    error=f"HTML extraction disabled for file: {path}",
                    processing_time=time.time() - start_time,
                )
            return self._extract_html(content, path, start_time)
        else:
            return self._extract_markdown(content, path, start_time)

    def extract_batch(
        self, file_paths: list[str | Path], **kwargs
    ) -> list[ExtractionResult]:
        """Extract content from multiple documents.

        Args:
            file_paths: List of document paths
            **kwargs: Additional extraction options

        Returns:
            List of ExtractionResult objects
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.extract(file_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {file_path}: {e}")
                results.append(
                    ExtractionResult(
                        text="",
                        error=str(e),
                        metadata={"file_path": str(file_path)},
                    )
                )
        return results

    def _extract_markdown(
        self, content: str, path: Path, start_time: float
    ) -> ExtractionResult:
        """Extract content from Markdown.

        Args:
            content: Raw markdown content
            path: File path for metadata
            start_time: Start time for processing duration

        Returns:
            ExtractionResult with structured content
        """
        # Extract code blocks before processing
        code_blocks = self._extract_code_blocks(content)

        # Extract tables
        tables = self._extract_tables(content) if self.config.extract_tables else []

        # Extract references/links
        references = (
            self._extract_links(content) if self.config.extract_references else []
        )

        # Extract headers for structure
        headers = self._extract_headers(content)

        # Convert to plain text (remove markdown syntax)
        plain_text = self._markdown_to_plain_text(content)

        # Build metadata
        metadata = {
            "file_path": str(path),
            "file_name": path.name,
            "format": "markdown",
            "header_count": len(headers),
            "code_block_count": len(code_blocks),
            "table_count": len(tables),
            "link_count": len(references),
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
        }

        return ExtractionResult(
            text=plain_text,
            metadata=metadata,
            code_blocks=code_blocks if self.config.extract_code else [],
            tables=tables,
            references=references,
            processing_time=time.time() - start_time,
        )

    def _extract_html(
        self, content: str, path: Path, start_time: float
    ) -> ExtractionResult:
        """Extract content from HTML.

        Args:
            content: Raw HTML content
            path: File path for metadata
            start_time: Start time for processing duration

        Returns:
            ExtractionResult with structured content
        """
        # Extract code blocks from <pre><code> and <code> tags
        code_blocks = self._extract_html_code_blocks(content)

        # Extract tables from <table> tags
        tables = (
            self._extract_html_tables(content) if self.config.extract_tables else []
        )

        # Extract links from <a> tags
        references = (
            self._extract_html_links(content) if self.config.extract_references else []
        )

        # Convert to plain text (strip HTML tags)
        plain_text = self._html_to_plain_text(content)

        # Build metadata
        metadata = {
            "file_path": str(path),
            "file_name": path.name,
            "format": "html",
            "code_block_count": len(code_blocks),
            "table_count": len(tables),
            "link_count": len(references),
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
        }

        return ExtractionResult(
            text=plain_text,
            metadata=metadata,
            code_blocks=code_blocks if self.config.extract_code else [],
            tables=tables,
            references=references,
            processing_time=time.time() - start_time,
        )

    def _extract_code_blocks(self, content: str) -> list[dict[str, Any]]:
        """Extract fenced code blocks from Markdown.

        Args:
            content: Markdown content

        Returns:
            List of code block dictionaries
        """
        code_blocks = []

        # Match fenced code blocks: ```language\ncode\n```
        # Use [^\n`]* to support languages like c++, c#, objective-c
        pattern = r"```([^\n`]*)\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, (language, code) in enumerate(matches):
            code_blocks.append(
                {
                    "index": i,
                    "language": language.strip() or "text",
                    "code": code.strip(),
                    "line_count": code.count("\n") + 1,
                }
            )

        # Also match indented code blocks (4 spaces or 1 tab)
        # Only if not inside a fenced block
        indented_pattern = r"(?:^|\n\n)((?:(?:    |\t).+\n?)+)"
        indented_matches = re.findall(indented_pattern, content)

        for code in indented_matches:
            # Remove indentation
            lines = code.split("\n")
            dedented = "\n".join(
                line[4:] if line.startswith("    ") else line[1:]
                for line in lines
                if line.strip()
            )
            if dedented.strip():
                code_blocks.append(
                    {
                        "index": len(code_blocks),
                        "language": "text",
                        "code": dedented.strip(),
                        "line_count": dedented.count("\n") + 1,
                    }
                )

        return code_blocks

    def _extract_tables(self, content: str) -> list[dict[str, Any]]:
        """Extract tables from Markdown.

        Args:
            content: Markdown content

        Returns:
            List of table dictionaries
        """
        tables = []

        # Match markdown tables (pipe-delimited)
        # Header row | separator row | data rows
        table_pattern = r"(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)+)"
        matches = re.findall(table_pattern, content)

        for i, (header_row, _separator, body_rows) in enumerate(matches):
            # Parse header - preserve empty cells but strip leading/trailing from split
            header_parts = [cell.strip() for cell in header_row.split("|")]
            # Remove empty strings from leading/trailing pipes (e.g., "|A|B|" -> ['', 'A', 'B', ''])
            headers = header_parts[1:-1] if header_parts[0] == "" and header_parts[-1] == "" else header_parts

            # Parse body rows - preserve empty cells for correct column alignment
            rows = []
            for row in body_rows.strip().split("\n"):
                cell_parts = [cell.strip() for cell in row.split("|")]
                # Remove empty strings from leading/trailing pipes
                cells = cell_parts[1:-1] if cell_parts[0] == "" and cell_parts[-1] == "" else cell_parts
                # Keep rows with at least one non-empty cell (skip fully-blank rows)
                if any(cell != "" for cell in cells):
                    rows.append(cells)

            tables.append(
                {
                    "index": i,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "column_count": len(headers),
                }
            )

        return tables

    def _extract_links(self, content: str) -> list[dict[str, Any]]:
        """Extract links from Markdown.

        Args:
            content: Markdown content

        Returns:
            List of link dictionaries
        """
        links = []

        # Match inline links: [text](url)
        inline_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for i, (text, url) in enumerate(re.findall(inline_pattern, content)):
            links.append(
                {
                    "index": i,
                    "text": text,
                    "url": url,
                    "type": "inline",
                }
            )

        # Match reference links: [text][ref] and [ref]: url
        # Reference labels are case-insensitive in Markdown
        ref_def_pattern = r"^\[([^\]]+)\]:\s*(.+)$"
        ref_defs = {
            k.lower(): v for k, v in re.findall(ref_def_pattern, content, re.MULTILINE)
        }

        ref_use_pattern = r"\[([^\]]+)\]\[([^\]]*)\]"
        for text, ref in re.findall(ref_use_pattern, content):
            ref_key = (ref or text).lower()
            if ref_key in ref_defs:
                links.append(
                    {
                        "index": len(links),
                        "text": text,
                        "url": ref_defs[ref_key],
                        "type": "reference",
                    }
                )

        # Match autolinks: <url>
        autolink_pattern = r"<(https?://[^>]+)>"
        for url in re.findall(autolink_pattern, content):
            links.append(
                {
                    "index": len(links),
                    "text": url,
                    "url": url,
                    "type": "autolink",
                }
            )

        return links

    def _extract_headers(self, content: str) -> list[dict[str, Any]]:
        """Extract headers from Markdown.

        Args:
            content: Markdown content

        Returns:
            List of header dictionaries
        """
        headers = []

        # Match ATX headers: # Header
        atx_pattern = r"^(#{1,6})\s+(.+)$"
        for match in re.finditer(atx_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append(
                {
                    "level": level,
                    "text": text,
                    "position": match.start(),
                }
            )

        # Match Setext headers (underlined)
        setext_pattern = r"^(.+)\n([=-]+)$"
        for match in re.finditer(setext_pattern, content, re.MULTILINE):
            text = match.group(1).strip()
            underline = match.group(2)
            level = 1 if underline[0] == "=" else 2
            headers.append(
                {
                    "level": level,
                    "text": text,
                    "position": match.start(),
                }
            )

        # Sort by position
        headers.sort(key=lambda h: h["position"])

        return headers

    def _markdown_to_plain_text(self, content: str) -> str:
        """Convert Markdown to plain text.

        Args:
            content: Markdown content

        Returns:
            Plain text content
        """
        text = content

        # Remove fenced code blocks (keep the code content)
        # Use [^\n`]* to match languages like c++, c#, objective-c
        text = re.sub(r"```[^\n`]*\n", "", text)
        text = re.sub(r"```", "", text)

        # Remove inline code backticks
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove headers markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Convert links to just text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)

        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

        # Remove blockquote markers
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_html_code_blocks(self, content: str) -> list[dict[str, Any]]:
        """Extract code blocks from HTML.

        Args:
            content: HTML content

        Returns:
            List of code block dictionaries
        """
        code_blocks = []

        # Match <pre><code> blocks - capture the whole code tag to extract class
        pre_code_pattern = r"<pre[^>]*><code([^>]*)>(.*?)</code></pre>"
        for i, (attrs, code) in enumerate(
            re.findall(pre_code_pattern, content, re.DOTALL | re.IGNORECASE)
        ):
            # Extract language from class attribute (supports c++, c#, etc.)
            language = "text"
            class_match = re.search(r'language-([A-Za-z0-9_+#-]+)', attrs, re.IGNORECASE)
            if class_match:
                language = class_match.group(1)
            # Unescape HTML entities
            code = self._unescape_html(code)
            code_blocks.append(
                {
                    "index": i,
                    "language": language,
                    "code": code.strip(),
                    "line_count": code.count("\n") + 1,
                }
            )

        # Match standalone <code> blocks, excluding those already in <pre><code>
        # First, remove <pre><code>...</code></pre> blocks to avoid double-counting
        content_no_pre = re.sub(pre_code_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        code_pattern = r"<code([^>]*)>(.*?)</code>"
        for attrs, code in re.findall(code_pattern, content_no_pre, re.DOTALL | re.IGNORECASE):
            # Extract language from class attribute (supports c++, c#, etc.)
            language = "text"
            class_match = re.search(r'language-([A-Za-z0-9_+#-]+)', attrs, re.IGNORECASE)
            if class_match:
                language = class_match.group(1)
            code = self._unescape_html(code)
            if code.strip() and len(code) > 50:  # Only substantial code blocks
                code_blocks.append(
                    {
                        "index": len(code_blocks),
                        "language": language or "text",
                        "code": code.strip(),
                        "line_count": code.count("\n") + 1,
                    }
                )

        return code_blocks

    def _extract_html_tables(self, content: str) -> list[dict[str, Any]]:
        """Extract tables from HTML.

        Args:
            content: HTML content

        Returns:
            List of table dictionaries
        """
        tables = []

        # Match <table> elements
        table_pattern = r"<table[^>]*>(.*?)</table>"
        for i, table_content in enumerate(
            re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
        ):
            # Extract header cells
            headers = []
            th_pattern = r"<th[^>]*>(.*?)</th>"
            for th in re.findall(th_pattern, table_content, re.DOTALL | re.IGNORECASE):
                headers.append(self._html_to_plain_text(th).strip())

            # Extract body rows
            rows = []
            tr_pattern = r"<tr[^>]*>(.*?)</tr>"
            for tr in re.findall(tr_pattern, table_content, re.DOTALL | re.IGNORECASE):
                td_pattern = r"<td[^>]*>(.*?)</td>"
                cells = [
                    self._html_to_plain_text(td).strip()
                    for td in re.findall(td_pattern, tr, re.DOTALL | re.IGNORECASE)
                ]
                if cells:
                    rows.append(cells)

            if headers or rows:
                tables.append(
                    {
                        "index": i,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers) or (len(rows[0]) if rows else 0),
                    }
                )

        return tables

    def _extract_html_links(self, content: str) -> list[dict[str, Any]]:
        """Extract links from HTML.

        Args:
            content: HTML content

        Returns:
            List of link dictionaries
        """
        links = []

        # Match <a> tags
        link_pattern = r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
        for i, (url, text) in enumerate(
            re.findall(link_pattern, content, re.DOTALL | re.IGNORECASE)
        ):
            text = self._html_to_plain_text(text).strip()
            links.append(
                {
                    "index": i,
                    "text": text or url,
                    "url": url,
                    "type": "anchor",
                }
            )

        return links

    def _html_to_plain_text(self, content: str) -> str:
        """Convert HTML to plain text.

        Args:
            content: HTML content

        Returns:
            Plain text content
        """
        text = content

        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Replace block elements with newlines
        text = re.sub(r"<br[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Unescape HTML entities
        text = self._unescape_html(text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _unescape_html(self, text: str) -> str:
        """Unescape HTML entities.

        Args:
            text: Text with HTML entities

        Returns:
            Unescaped text
        """
        entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
            "&nbsp;": " ",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "…",
            "&copy;": "©",
            "&reg;": "®",
            "&trade;": "™",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Handle numeric entities with safe conversion
        def safe_chr(codepoint: int) -> str:
            """Convert codepoint to character, returning replacement char on error."""
            try:
                return chr(codepoint)
            except (ValueError, OverflowError):
                # Out-of-range codepoints (e.g., &#999999999; or &#xFFFFFFFF;)
                return "\uFFFD"  # Unicode replacement character

        text = re.sub(
            r"&#(\d+);", lambda m: safe_chr(int(m.group(1))), text
        )
        text = re.sub(
            r"&#x([0-9a-fA-F]+);", lambda m: safe_chr(int(m.group(1), 16)), text
        )

        return text
