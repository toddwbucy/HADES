"""Unit tests for MarkdownExtractor.

Tests for:
- Markdown extraction (code blocks, tables, links, headers)
- HTML extraction (code, tables, links)
- Plain text conversion
- Batch extraction
- Error handling
"""


class TestMarkdownExtractorInit:
    """Tests for MarkdownExtractor initialization."""

    def test_default_config(self):
        """Should initialize with default configuration."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        extractor = MarkdownExtractor()

        assert extractor.config is not None
        assert extractor.extract_html is True

    def test_supported_formats_with_html(self):
        """Should include HTML formats when extract_html=True."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        extractor = MarkdownExtractor(extract_html=True)

        formats = extractor.supported_formats
        assert ".md" in formats
        assert ".markdown" in formats
        assert ".html" in formats
        assert ".htm" in formats

    def test_supported_formats_without_html(self):
        """Should exclude HTML formats when extract_html=False."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        extractor = MarkdownExtractor(extract_html=False)

        formats = extractor.supported_formats
        assert ".md" in formats
        assert ".markdown" in formats
        assert ".html" not in formats
        assert ".htm" not in formats


class TestMarkdownExtraction:
    """Tests for Markdown content extraction."""

    def test_extracts_plain_text(self, tmp_path):
        """Should extract plain text from markdown."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello World\n\nThis is a paragraph.")

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert result.error is None
        assert "Hello World" in result.text
        assert "This is a paragraph" in result.text

    def test_extracts_code_blocks(self, tmp_path):
        """Should extract fenced code blocks."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Code Example\n\n```python\nprint('hello')\n```\n"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "python"
        assert "print('hello')" in result.code_blocks[0]["code"]

    def test_extracts_tables(self, tmp_path):
        """Should extract markdown tables."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "| Name | Age |\n"
            "|------|-----|\n"
            "| Alice | 30 |\n"
            "| Bob | 25 |\n"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.tables) == 1
        assert result.tables[0]["headers"] == ["Name", "Age"]
        assert len(result.tables[0]["rows"]) == 2

    def test_extracts_links(self, tmp_path):
        """Should extract markdown links."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "Check out [GitHub](https://github.com) for code.\n"
            "Also see <https://example.com> for more."
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.references) >= 2
        urls = [ref["url"] for ref in result.references]
        assert "https://github.com" in urls
        assert "https://example.com" in urls

    def test_extracts_headers(self, tmp_path):
        """Should extract headers for document structure."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Title\n\n## Section 1\n\n### Subsection\n\n## Section 2\n"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert result.metadata["header_count"] == 4

    def test_removes_markdown_syntax(self, tmp_path):
        """Should remove markdown syntax in plain text output."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# Header\n\n"
            "**Bold** and *italic* text.\n"
            "[Link](http://example.com)\n"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        # Should not contain markdown syntax
        assert "**" not in result.text
        assert "*italic*" not in result.text
        assert "](http" not in result.text
        # Should contain the actual text
        assert "Bold" in result.text
        assert "italic" in result.text
        assert "Link" in result.text

    def test_metadata_includes_counts(self, tmp_path):
        """Should include various counts in metadata."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nContent here.")

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert "file_path" in result.metadata
        assert "file_name" in result.metadata
        assert "format" in result.metadata
        assert result.metadata["format"] == "markdown"
        assert "char_count" in result.metadata
        assert "line_count" in result.metadata


class TestHTMLExtraction:
    """Tests for HTML content extraction."""

    def test_extracts_plain_text_from_html(self, tmp_path):
        """Should extract plain text from HTML."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>Hello</h1><p>World</p></body></html>"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert result.error is None
        assert "Hello" in result.text
        assert "World" in result.text
        assert "<html>" not in result.text

    def test_extracts_code_from_html(self, tmp_path):
        """Should extract code blocks from HTML."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            '<pre><code class="language-python">print("hello")</code></pre>'
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert len(result.code_blocks) >= 1
        assert result.code_blocks[0]["language"] == "python"

    def test_extracts_tables_from_html(self, tmp_path):
        """Should extract tables from HTML."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<table>"
            "<tr><th>Name</th><th>Age</th></tr>"
            "<tr><td>Alice</td><td>30</td></tr>"
            "</table>"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert len(result.tables) == 1
        assert result.tables[0]["headers"] == ["Name", "Age"]

    def test_extracts_links_from_html(self, tmp_path):
        """Should extract links from HTML."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            '<a href="https://example.com">Example</a>'
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert len(result.references) == 1
        assert result.references[0]["url"] == "https://example.com"
        assert result.references[0]["text"] == "Example"

    def test_unescapes_html_entities(self, tmp_path):
        """Should unescape HTML entities."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text("<p>&lt;tag&gt; &amp; &quot;quotes&quot;</p>")

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert "<tag>" in result.text
        assert "&" in result.text
        assert '"quotes"' in result.text

    def test_removes_script_and_style(self, tmp_path):
        """Should remove script and style elements."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert('hi')</script>Content</body></html>"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert "color:red" not in result.text
        assert "alert" not in result.text
        assert "Content" in result.text


class TestBatchExtraction:
    """Tests for batch extraction."""

    def test_extracts_multiple_files(self, tmp_path):
        """Should extract multiple files."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        file1 = tmp_path / "file1.md"
        file1.write_text("# File 1")
        file2 = tmp_path / "file2.md"
        file2.write_text("# File 2")

        extractor = MarkdownExtractor()
        results = extractor.extract_batch([file1, file2])

        assert len(results) == 2
        assert "File 1" in results[0].text
        assert "File 2" in results[1].text

    def test_handles_errors_in_batch(self, tmp_path):
        """Should handle errors gracefully in batch mode."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        valid_file = tmp_path / "valid.md"
        valid_file.write_text("# Valid")
        invalid_file = tmp_path / "nonexistent.md"

        extractor = MarkdownExtractor()
        results = extractor.extract_batch([valid_file, invalid_file])

        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error is not None


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_nonexistent_file(self, tmp_path):
        """Should handle nonexistent files gracefully."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        extractor = MarkdownExtractor()
        result = extractor.extract(tmp_path / "nonexistent.md")

        assert result.error is not None
        assert result.text == ""

    def test_handles_empty_file(self, tmp_path):
        """Should handle empty files gracefully."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")

        extractor = MarkdownExtractor()
        result = extractor.extract(empty_file)

        # Empty file should fail validation
        assert result.error is not None

    def test_handles_encoding_issues(self, tmp_path):
        """Should handle encoding issues with fallback."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        # Write a file with latin-1 encoding
        latin_file = tmp_path / "latin.md"
        latin_file.write_bytes(b"# Caf\xe9")  # é in latin-1

        extractor = MarkdownExtractor()
        result = extractor.extract(latin_file)

        assert result.error is None
        assert "Caf" in result.text


class TestExtractorFactory:
    """Tests for factory integration."""

    def test_factory_creates_markdown_extractor(self):
        """Should create MarkdownExtractor via factory."""
        from core.extractors.extractor_factory import ExtractorFactory
        from core.extractors.extractors_markdown import MarkdownExtractor

        extractor = ExtractorFactory.create("markdown")

        assert isinstance(extractor, MarkdownExtractor)

    def test_factory_for_mdown_file(self, tmp_path):
        """Should return MarkdownExtractor for .mdown files (not claimed by docling)."""
        from core.extractors.extractor_factory import ExtractorFactory
        from core.extractors.extractors_markdown import MarkdownExtractor

        mdown_file = tmp_path / "test.mdown"
        mdown_file.write_text("# Test")

        extractor = ExtractorFactory.for_file(mdown_file)

        assert isinstance(extractor, MarkdownExtractor)

    def test_factory_for_html_file(self, tmp_path):
        """Should return MarkdownExtractor for .html files."""
        from core.extractors.extractor_factory import ExtractorFactory
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        extractor = ExtractorFactory.for_file(html_file)

        assert isinstance(extractor, MarkdownExtractor)

    def test_factory_lists_markdown(self):
        """Should list markdown in available extractors."""
        from core.extractors.extractor_factory import ExtractorFactory

        available = ExtractorFactory.list_available()

        assert "markdown" in available


class TestCodeRabbitReviewFixes:
    """Tests for issues identified in CodeRabbit review."""

    def test_extract_html_disabled_returns_error(self, tmp_path):
        """Should return error when extract_html=False and given HTML file."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>")

        extractor = MarkdownExtractor(extract_html=False)
        result = extractor.extract(html_file)

        assert result.error is not None
        assert "HTML extraction disabled" in result.error

    def test_case_insensitive_reference_links(self, tmp_path):
        """Should match reference links case-insensitively."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "[Click HERE][Example]\n\n"
            "[example]: https://example.com"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.references) == 1
        assert result.references[0]["url"] == "https://example.com"

    def test_extracts_cpp_language_code_block(self, tmp_path):
        """Should extract code blocks with non-word language tags like c++."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "```c++\n"
            "int main() { return 0; }\n"
            "```"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "c++"

    def test_extracts_csharp_language_code_block(self, tmp_path):
        """Should extract code blocks with c# language tag."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        md_file = tmp_path / "test.md"
        md_file.write_text(
            "```c#\n"
            "class Program { }\n"
            "```"
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(md_file)

        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "c#"

    def test_html_code_with_cpp_language(self, tmp_path):
        """Should extract HTML code blocks with c++ language class."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        html_file.write_text(
            '<pre><code class="language-c++">int main() {}</code></pre>'
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "c++"

    def test_no_double_counting_pre_code_blocks(self, tmp_path):
        """Should not double-count code in <pre><code> blocks."""
        from core.extractors.extractors_markdown import MarkdownExtractor

        html_file = tmp_path / "test.html"
        # This code appears in both <pre><code> and standalone <code>
        html_file.write_text(
            '<pre><code class="language-python">'
            'def hello():\n    print("world")\n'
            '</code></pre>'
        )

        extractor = MarkdownExtractor()
        result = extractor.extract(html_file)

        # Should only count once
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0]["language"] == "python"
