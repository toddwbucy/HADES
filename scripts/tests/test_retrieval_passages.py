"""Synthetic snapshot contracts; never read private paper drafts."""
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("passages", Path(__file__).parents[1] / "prepare_retrieval_passages.py")
passages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(passages)


class PassageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "documents").mkdir()

    def snapshot(self, raw):
        (self.root / "documents/draft.md").write_bytes(raw)
        manifest = {"documents": [{"path":"draft.md", "bytes":len(raw),
            "sha256":passages.digest(raw)}]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_multibyte_long_lines_and_newlines_round_trip_with_exact_offsets(self):
        raw = "# Draft\r\n\nA😀é長 line without a newline\nLast".encode()
        self.snapshot(raw)
        result = passages.prepare(self.root, 9)
        rebuilt = b""
        for item in result["documents"]:
            text = item["text"].encode()
            start, end = item["start_byte"], item["end_byte"]
            self.assertEqual(start, len(rebuilt))
            self.assertEqual(raw[start:end], text)
            self.assertLessEqual(len(text), 9)
            self.assertEqual(item["start_line"], raw[:start].count(b"\n") + 1)
            self.assertEqual(item["end_line"], raw[:end].count(b"\n") + int(not raw[:end].endswith(b"\n")))
            rebuilt += text
        self.assertEqual(rebuilt, raw)
        self.assertEqual(result, passages.prepare(self.root, 9))
        self.assertEqual(result["queries"], [])

    def test_changed_source_is_rejected(self):
        self.snapshot(b"original")
        (self.root / "documents/draft.md").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "size/hash"):
            passages.prepare(self.root)

    def test_invalid_utf8_is_not_silently_replaced(self):
        self.snapshot(b"invalid\xff")
        with self.assertRaises(UnicodeDecodeError):
            passages.prepare(self.root)

    def test_duplicate_and_escaping_paths_are_rejected(self):
        manifest = self.snapshot(b"x")
        manifest["documents"] *= 2
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            passages.prepare(self.root)
        manifest["documents"] = [dict(manifest["documents"][0], path="../draft.md")]
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "normalized and relative"):
            passages.prepare(self.root)

    def test_symlink_outside_snapshot_is_rejected(self):
        self.snapshot(b"x")
        source = self.root / "documents/draft.md"
        source.unlink()
        (self.root / "outside.md").write_bytes(b"x")
        source.symlink_to(self.root / "outside.md")
        with self.assertRaisesRegex(ValueError, "escapes snapshot"):
            passages.prepare(self.root)

    def test_declared_oversize_is_rejected_before_file_read(self):
        manifest = self.snapshot(b"x")
        manifest["documents"][0]["bytes"] = passages.MAX_DOCUMENT_BYTES + 1
        (self.root / "documents/draft.md").unlink()
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "document exceeds"):
            passages.prepare(self.root)

    def test_private_output_is_exclusive(self):
        output = self.root / "passages.json"
        passages.write_private(output, {"test":"é"})
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            passages.write_private(output, {})
        self.assertEqual(json.loads(output.read_text()), {"test":"é"})

    def test_aggregate_byte_and_passage_budgets_reject_excess(self):
        self.snapshot(b"123456789")
        with patch.object(passages, "MAX_CORPUS_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "corpus exceeds byte"):
                passages.prepare(self.root)
        with patch.object(passages, "MAX_PASSAGES", 1):
            with self.assertRaisesRegex(ValueError, "passage limit"):
                passages.prepare(self.root, 4)

    def test_bounds_and_empty_corpus_are_rejected(self):
        self.snapshot(b"")
        with self.assertRaisesRegex(ValueError, "no passages"):
            passages.prepare(self.root)
        for cap in [0, 3, 65537, True]:
            with self.assertRaisesRegex(ValueError, "max_bytes"):
                passages.prepare(self.root, cap)


if __name__ == "__main__":
    unittest.main()
