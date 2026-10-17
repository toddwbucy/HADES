"""Docling v2 adapter contracts, using real document types without loading models."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, Size, TableData
from docling.datamodel.base_models import ConversionStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.docling_backend import DoclingExtractor


def extract(tmp_path, document, status=ConversionStatus.SUCCESS):
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


@pytest.mark.parametrize("status", [ConversionStatus.FAILURE, ConversionStatus.PARTIAL_SUCCESS])
def test_incomplete_conversion_is_not_success(tmp_path, status):
    result = extract(tmp_path, DoclingDocument(name="incomplete"), status)
    assert "did not complete" in result.error
    assert result.text == ""


def test_success_without_document_is_not_success(tmp_path):
    result = extract(tmp_path, None)
    assert "no document" in result.error
