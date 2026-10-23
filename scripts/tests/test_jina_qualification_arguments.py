"""Qualification opt-in arguments must fail before starting any database or model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class QualificationArguments(unittest.TestCase):
    """Exercise actual command-line refusal without ML dependencies or sockets."""

    def runner(self, *arguments):
        """Use an existing harmless executable; rejected arguments must never launch it."""
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/test_isolated_database.py"),
             "--arangod", sys.executable, *map(str, arguments)],
            text=True, capture_output=True, timeout=10,
        )

    def test_context_and_report_must_be_explicit_and_new(self):
        """Reject missing pairs, missing context, and any existing evidence file."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text("{}")
            report = root / "report.json"
            for args in [
                ["--verified-jina", context],
                ["--verified-jina-report", report],
                ["--verified-jina", root / "missing", "--verified-jina-report", report],
            ]:
                with self.subTest(args=args):
                    result = self.runner(*args)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertNotIn("Isolated test artifacts", result.stdout)
            report.write_text("keep existing evidence")
            result = self.runner("--verified-jina", context, "--verified-jina-report", report)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(report.read_text(), "keep existing evidence")
            self.assertNotIn("Isolated test artifacts", result.stdout)

    def test_cli_help_needs_no_model_or_ml_runtime(self):
        """Keep the opt-in command discoverable before installing its ML dependencies."""
        result = subprocess.run([sys.executable, str(ROOT / "scripts/qualify_verified_jina.py"), "--help"],
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--registry", result.stdout)

    def test_cli_requires_explicit_offline_environment_before_outputs(self):
        """Refuse accidental online loading before creating evidence or a registry."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            env = {k: v for k, v in os.environ.items()
                   if k not in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}}
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/qualify_verified_jina.py"),
                "--registry", str(root / "registry"), "--provider", "fixture",
                "--device", "cpu", "--output-dir", str(output), "--arangod", sys.executable,
            ], env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("HF_HUB_OFFLINE=1", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse((root / "registry").exists())


if __name__ == "__main__":
    unittest.main()
