"""Probe preflight contracts without Docling, model downloads, or source corpora."""
import builtins
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "probe_pdf_layout.py"
SPEC = importlib.util.spec_from_file_location("probe_pdf_layout", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ModelBoundaryReached(Exception):
    """Stop a valid invocation before importing any model dependencies."""


class ProbePreflight(unittest.TestCase):
    """All inputs must pass validation before output or model initialization."""

    def setUp(self):
        """Create synthetic paths in a private, disposable directory."""
        temp = tempfile.TemporaryDirectory(prefix="hades-layout-preflight-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pdf = self.root / "source.pdf"
        self.pdf.write_bytes(b"%PDF-synthetic-preflight-only")
        self.output = self.root / "output"
        self.stderr = io.StringIO()

    def invoke(self, cases):
        """Run argument parsing with a hard stop at the first Docling import."""
        argv = [str(SCRIPT), "--output", str(self.output)]
        for label, path, page in cases:
            argv.extend(["--case", label, str(path), str(page)])
        original_import = builtins.__import__

        def guard_import(name, *args, **kwargs):
            """Keep preflight tests independent of installed ML dependencies."""
            if name.split(".")[0] == "docling":
                raise ModelBoundaryReached()
            return original_import(name, *args, **kwargs)

        with patch.object(sys, "argv", argv), patch("builtins.__import__", guard_import):
            with redirect_stderr(self.stderr):
                PROBE.main()

    def test_invalid_later_case_never_initializes_models_or_creates_output(self):
        """Reject invalid paths and pages even after an otherwise valid case."""
        non_pdf = self.root / "source.txt"
        non_pdf.write_text("synthetic")
        directory = self.root / "directory.pdf"
        directory.mkdir()
        invalid = [
            (self.pdf, "not-an-integer"), (self.pdf, 0), (self.pdf, -1),
            (self.root / "missing.pdf", 1), (directory, 1), (non_pdf, 1),
        ]
        for path, page in invalid:
            with self.subTest(path=path.name, page=page):
                with self.assertRaises(SystemExit) as exit_status:
                    self.invoke([("valid", self.pdf, 1), ("invalid", path, page)])
                self.assertEqual(exit_status.exception.code, 2)
                self.assertFalse(self.output.exists())
                self.assertIn("invalid: invalid case:", self.stderr.getvalue())
        self.assertEqual(self.pdf.read_bytes(), b"%PDF-synthetic-preflight-only")

    def test_corrected_retry_can_reuse_output_name(self):
        """An invalid page must not reserve the output directory for a retry."""
        with self.assertRaises(SystemExit):
            self.invoke([("case", self.pdf, 0)])
        with self.assertRaises(ModelBoundaryReached):
            self.invoke([("case", self.pdf, 1)])
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_output_is_preserved_and_reported_before_model_load(self):
        """A prior run's artifacts cannot be overwritten by a repeated invocation."""
        self.output.mkdir()
        marker = self.output / "manifest.json"
        marker.write_text("preserve")
        with self.assertRaises(SystemExit) as exit_status:
            self.invoke([("case", self.pdf, 1)])
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("cannot create output directory:", self.stderr.getvalue())
        self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
