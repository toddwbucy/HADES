"""Reading runner deadlines own and stop every process in their private group."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("isolated_reading_runner", Path(__file__).resolve().parents[1] / "test_isolated_database.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReadingCleanup(unittest.TestCase):
    def test_timeout_and_exited_leader_reap_descendants(self):
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out), tempfile.TemporaryDirectory() as directory:
                pidfile = Path(directory) / "group.json"
                code = '''import json, os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
with open(sys.argv[1], 'w') as stream:
    json.dump({'group':os.getpgrp(), 'child':child.pid}, stream)
if sys.argv[2] == 'timeout': time.sleep(120)
'''
                argv = [sys.executable, "-c", code, str(pidfile), "timeout" if timed_out else "exit"]
                results = {}
                if timed_out:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        runner.run_reading_e2e(argv, dict(os.environ), 1, results)
                    self.assertFalse(results["reading-e2e"]["passed"])
                    self.assertIn("error", results["reading-e2e"])
                else:
                    runner.run_reading_e2e(argv, dict(os.environ), 5, results)
                    self.assertTrue(results["reading-e2e"]["passed"])
                self.assertEqual(results["reading-e2e"]["command"], argv)
                self.assertFalse(runner.group_running(json.loads(pidfile.read_text())["group"]))

    def test_nonzero_exit_keeps_failure_receipt(self):
        results = {}
        with self.assertRaisesRegex(RuntimeError, "end-to-end test failed"):
            runner.run_reading_e2e([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), 5, results)
        self.assertEqual(results["reading-e2e"]["exit_code"], 7)
        self.assertFalse(results["reading-e2e"]["passed"])
