"""Pinned Docling adapter contracts with real document types and no models."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.docling_backend import DoclingExtractor


def extract(tmp_path, document, status="success"):
    """Run the adapter with a supplied conversion result and a private source."""
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"%PDF-fixture")

    class Converter:
        def convert(self, source):
            """Return the document while checking the adapter uses the source path."""
            assert source == str(path)
            return SimpleNamespace(document=document, status=status)

    backend = DoclingExtractor()
    backend._converter = Converter()
    return backend.extract(path)


def test_v2_document_exports_text_pages_and_linked_captions(tmp_path):
    import docling_core.types.doc as types
    DocItemLabel, DoclingDocument = types.DocItemLabel, types.DoclingDocument
    Size, TableData, TableCell = types.Size, types.TableData, types.TableCell
    doc = DoclingDocument(name="fixture")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    doc.add_text(label=DocItemLabel.TEXT, text="The human selects a source passage.")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="The selected passage.")
    doc.add_picture(caption=caption)
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=column,
            end_col_offset_idx=column + 1,
            column_header=(row == 0),
        )
        for row, values in enumerate((("Subject", "Status"), ("Source passage", "Reviewed")))
        for column, text in enumerate(values)
    ]
    doc.add_table(
        data=TableData(num_rows=2, num_cols=2, table_cells=cells),
        caption=caption,
    )

    result = extract(tmp_path, doc)
    assert result.error is None
    assert "The human selects a source passage." in result.text
    assert result.metadata["extractor"] == "docling"
    assert result.metadata["num_pages"] == "1"
    assert result.images[0]["caption"] == "The selected passage."
    assert result.tables[0]["caption"] == "The selected passage."
    markdown = result.tables[0]["content"]
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in markdown.splitlines() if line.startswith("|")
    ]
    assert len(rows) == 3
    assert rows[0] == ["Subject", "Status"]
    assert len(rows[1]) == 2 and all(set(cell) <= {"-", ":"} and "-" in cell for cell in rows[1])
    assert rows[2] == ["Source passage", "Reviewed"]
    assert markdown in result.text


def test_adapter_passes_document_to_table_export(tmp_path):
    class CaptionedItem:
        def caption_text(self, document):
            assert document is expected_doc
            return "Source caption"

        def export_to_markdown(self, *, doc):
            assert doc is expected_doc
            return "| Source table |"

    item = CaptionedItem()
    expected_doc = SimpleNamespace(
        export_to_markdown=lambda: "Source content",
        num_pages=lambda: 2,
        texts=[], tables=[item], pictures=[item],
    )
    result = extract(tmp_path, expected_doc, SimpleNamespace(value="success"))
    assert result.error is None
    assert result.text == "Source content"
    assert result.metadata["num_pages"] == "2"
    assert result.tables[0]["content"] == "| Source table |"
    assert result.tables[0]["caption"] == "Source caption"
    assert result.images[0]["caption"] == "Source caption"


@pytest.mark.parametrize("status", ["failure", "partial_success"])
def test_incomplete_conversion_is_not_success(tmp_path, status):
    result = extract(tmp_path, SimpleNamespace(), status)
    assert "did not complete" in result.error
    assert result.text == ""


def test_success_without_document_is_not_success(tmp_path):
    result = extract(tmp_path, None)
    assert "no document" in result.error


def test_formula_labels_and_enrichment_metadata(tmp_path):
    from docling_core.types.doc import DocItemLabel, DoclingDocument
    doc = DoclingDocument(name="formulas")
    doc.add_text(label=DocItemLabel.FORMULA, text="x = 2")
    result = extract(tmp_path, doc)
    assert result.error is None
    assert result.equations == [{"text": "x = 2", "latex": "", "index": 0, "is_inline": False}]
    assert result.metadata["formula_enrichment"] == "false"


def test_converter_failure_is_never_a_text_fallback(tmp_path):
    class Broken:
        def convert(self, source):
            raise RuntimeError("test converter failure")
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-fixture")
    backend = DoclingExtractor()
    backend._converter = Broken()
    result = backend.extract(path)
    assert result.error == "Docling extraction failed: test converter failure"
    assert not result.text and not result.metadata


def test_old_fallback_configuration_fails_explicitly(monkeypatch):
    from extraction.config import ExtractionConfig
    monkeypatch.setenv("HADES_EXTRACTOR_FALLBACK", "true")
    with pytest.raises(ValueError, match="no longer supported"):
        ExtractionConfig.from_env()


def test_pinned_registry_and_rust_routing_agree():
    import importlib.metadata
    import re
    from docling.datamodel.base_models import FormatToExtensions
    assert importlib.metadata.version("docling") == "2.126.0"
    assert importlib.metadata.version("docling-core") == "2.96.0"
    registry = {"." + e.lower() for values in FormatToExtensions.values() for e in values}
    assert DoclingExtractor.SUPPORTED_EXTENSIONS == registry | {".markdown"}
    routing = (Path(__file__).resolve().parents[2] / "crates/hades-core/src/ingest_routing.rs").read_text()
    table = routing.split('const DOCUMENT_EXTENSIONS: &[&str] = &[', 1)[1].split('];', 1)[0]
    declared = {"." + e for e in re.findall(r'"([^"]+)"', table)}
    assert declared == registry | {".markdown", ".rst", ".gz", ".tgz"}


def test_service_policy_controls_pdf_and_image_models():
    from docling.datamodel.base_models import InputFormat
    backend = DoclingExtractor(formula_enrichment=True, use_ocr=True, device="cpu")
    for kind in (InputFormat.PDF, InputFormat.IMAGE):
        options = backend.converter.format_to_options[kind].pipeline_options
        assert options.do_formula_enrichment and options.do_ocr
        assert options.accelerator_options.device == "cpu"


def test_pinned_archive_probe_distinguishes_mets_from_arxiv(tmp_path):
    import io
    import tarfile
    for name, text, expected in [
        ("mets.tar.gz", '<mets xmlns="http://www.loc.gov/METS/"/>', True),
        ("arxiv.tar.gz", '<article>Native archive</article>', False),
    ]:
        path = tmp_path / name
        content = text.encode()
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo("metadata.xml")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        assert DoclingExtractor.is_mets_archive(path) is expected


def test_default_device_is_portable(monkeypatch):
    from extraction.config import ExtractionConfig
    monkeypatch.delenv("HADES_EXTRACTOR_DEVICE", raising=False)
    monkeypatch.delenv("HADES_EXTRACTOR_FALLBACK", raising=False)
    assert ExtractionConfig.from_env().device == "auto"


def test_reading_document_keeps_structure_and_reports_unavailable_figures():
    import json
    from docling_core.types.doc import DoclingDocument, DocItemLabel, TableData, TableCell
    from extraction.docling_backend import reading_document
    doc = DoclingDocument(name="reading fixture")
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="A heading")
    doc.add_text(label=DocItemLabel.TEXT, text="The exact source passage.")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="An unavailable picture.")
    doc.add_picture(caption=caption)
    doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[
        TableCell(text="A cell",start_row_offset_idx=0,end_row_offset_idx=1,
                  start_col_offset_idx=0,end_col_offset_idx=1)]))
    result = json.loads(reading_document(doc))
    assert result["format"] == "docling-blocks-v1"
    assert any(b["kind"] == "section_header" and b["text"] == "A heading" for b in result["blocks"])
    assert next(b for b in result["blocks"] if b["kind"] == "table")["rows"] == [["A cell"]]
    picture = next(b for b in result["blocks"] if b["kind"] == "picture")
    assert picture["caption"] == "An unavailable picture."
    assert picture["image_available"] is False
    assert any("figures" in w for w in result["warnings"])
