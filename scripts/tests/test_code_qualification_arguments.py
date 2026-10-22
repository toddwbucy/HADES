"""Code qualification must be explicitly configured before touching assets or outputs."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "qualify_verified_code.py"


class CodeQualificationArguments(unittest.TestCase):
    def test_help_without_ml_runtime(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--code-max-input-tokens", result.stdout)

    def test_offline_device_and_distinct_bounded_ceilings_precede_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            for offline, device, document, code in [
                (False, "cpu", "1024", "512"), (True, "cuda", "1024", "512"),
                (True, "cpu", "1024", "0"), (True, "cpu", "1024", "8193"),
                (True, "cpu", "512", "512"), (True, "cpu", "1024", "8.5"),
            ]:
                with self.subTest(offline=offline, device=device, document=document, code=code):
                    env = {k: v for k, v in os.environ.items()
                           if k not in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}}
                    if offline:
                        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    result = subprocess.run([
                        sys.executable, str(SCRIPT), "--registry", str(root / "registry"),
                        "--provider", "fixture", "--device", device, "--output-dir", str(output),
                        "--max-input-tokens", document, "--code-max-input-tokens", code,
                    ], env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertFalse(output.exists())
                    self.assertFalse((root / "registry").exists())


if __name__ == "__main__":
    unittest.main()
