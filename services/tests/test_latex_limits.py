"""Small synthetic fixtures for untrusted LaTeX/archive resource limits."""
import gzip
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.latex_backend import LaTeXExtractor


@pytest.mark.parametrize("compressed", [False, True])
def test_source_byte_limit_is_checked_before_decode(tmp_path, compressed):
    extractor = LaTeXExtractor()
    extractor.MAX_SOURCE_BYTES = 16
    path = tmp_path / ("source.tex.gz" if compressed else "source.tex")
    for size in [16, 17]:
        raw = b"x" * size
        path.write_bytes(gzip.compress(raw) if compressed else raw)
        result = extractor.extract(path)
        if size == 16:
            assert result.error is None
            assert result.text == "x" * 16
        else:
            assert "expanded size exceeds limit" in result.error
            assert not result.text


def make_tar(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_tar_member_limit_counts_rejected_paths(tmp_path):
    path = tmp_path / "source.tar.gz"
    make_tar(path, [("../../bad.tex", b"x"), ("good.tex", b"valid"), ("extra.tex", b"x")])
    extractor = LaTeXExtractor()
    extractor.MAX_TAR_MEMBERS = 2
    result = extractor.extract(path)
    assert "member count exceeds limit" in result.error
    assert not result.text


def test_tar_expanded_stream_limit_precedes_tar_parsing(tmp_path):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(gzip.compress(b"x" * 4096))
    extractor = LaTeXExtractor()
    extractor.MAX_TAR_STREAM_BYTES = 128
    result = extractor.extract(path)
    assert "expanded stream exceeds limit" in result.error
    assert not result.text


def test_tar_payload_budget_and_safe_extraction(tmp_path):
    path = tmp_path / "source.tar.gz"
    make_tar(path, [("nested/main.tex", b"hello world"), ("../outside.tex", b"bad")])
    extractor = LaTeXExtractor()
    result = extractor.extract(path)
    assert result.error is None
    assert result.text == "hello world"
    assert not (tmp_path / "outside.tex").exists()
    extractor.MAX_TAR_EXPANDED_BYTES = 10
    result = extractor.extract(path)
    assert "expanded size exceeds limit" in result.error
    assert not result.text


def test_concatenated_gzip_members_share_source_budget(tmp_path):
    path = tmp_path / "source.tex.gz"
    path.write_bytes(gzip.compress(b"x" * 8) + gzip.compress(b"y" * 9))
    extractor = LaTeXExtractor()
    extractor.MAX_SOURCE_BYTES = 16
    result = extractor.extract(path)
    assert "expanded size exceeds limit" in result.error
    assert not result.text


def test_duplicate_normalized_tar_paths_fail_closed(tmp_path):
    path = tmp_path / "source.tar.gz"
    make_tar(path, [("main.tex", b"first"), ("./main.tex", b"second")])
    result = LaTeXExtractor().extract(path)
    assert "duplicate source paths" in result.error
    assert not result.text
