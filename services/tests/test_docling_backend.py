"""Docling v2 adapter contracts without models; optional real document-type check."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.docling_backend import DoclingExtractor


def extract(tmp_path, document, status="success"):
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"%PDF-fixture")

    class Converter:
        def convert(self, source):
            assert source == str(path)
            return SimpleNamespace(document=document, status=status)

    backend = DoclingExtractor(use_fallback=False)
    backend._converter = Converter()
    return backend.extract(path)


def test_v2_document_exports_text_pages_and_linked_captions(tmp_path):
    types = pytest.importorskip("docling_core.types.doc", reason="real document types need Docling")
    DocItemLabel, DoclingDocument = types.DocItemLabel, types.DoclingDocument
    Size, TableData = types.Size, types.TableData
    doc = DoclingDocument(name="fixture")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    doc.add_text(label=DocItemLabel.TEXT, text="The human selects a source passage.")
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="The selected passage.")
    doc.add_picture(caption=caption)
    doc.add_table(data=TableData(num_rows=0, num_cols=0, table_cells=[]), caption=caption)

    result = extract(tmp_path, doc)
    assert result.error is None
    assert "The human selects a source passage." in result.text
    assert result.metadata["extractor"] == "docling"
    assert result.metadata["num_pages"] == "1"
    assert result.images[0]["caption"] == "The selected passage."
    assert result.tables[0]["caption"] == "The selected passage."


def test_v2_api_contract_without_optional_docling_package(tmp_path):
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
