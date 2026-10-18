"""Backup integrity checks must fail before a restore process is started."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_workspace_backup import verified_files


class BackupIntegrityTests(unittest.TestCase):
    def test_valid_and_corrupted_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dump").mkdir()
            source = root / "dump/data"
            source.write_bytes(b"source")
            (root / "manifest.json").write_text("{}")
            manifest = {"files": {"dump/data": hashlib.sha256(b"source").hexdigest()}}
            self.assertEqual(verified_files(root, manifest), {"dump/data": source})
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verified_files(root, manifest)

    def test_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../outside", "/tmp/outside"):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    verified_files(root, {"files": {name: "unused"}})
            (root / "link").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "symlink"):
                verified_files(root, {"files": {"link": "unused"}})

    def test_unmanifested_dump_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.data.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                verified_files(root, {"files": {}})

    def test_unverified_provider_transcript_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "provider transcript"):
                verified_files(root, {"files": {}, "provider_transcripts": ["codex/sessions/missing.jsonl"]})


if __name__ == "__main__":
    unittest.main()
