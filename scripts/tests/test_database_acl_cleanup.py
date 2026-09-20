"""The private ACL probe must clean up setup/startup failures in the right order."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_database_acl as probe


class AclCleanupTests(unittest.TestCase):
    def run_failure(self, *, before_start=False, stop_fails=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary = base / 'arangod'
            binary.write_text('never executed')
            owned = base / 'owned'
            owned.mkdir(mode=0o700)
            child = Mock()
            child.poll.return_value = 1
            stopped = []
            def stop(value):
                self.assertIs(value, child)
                self.assertTrue(owned.is_dir())
                stopped.append(True)
                if stop_fails:
                    raise RuntimeError('injected stop failure')
            with patch.object(sys, 'argv', ['probe', '--bin-dir', str(base)]), \
                 patch.object(probe.tempfile, 'mkdtemp', return_value=str(owned)), \
                 patch.object(probe.isolation, 'lower_priority', side_effect=RuntimeError('setup failed') if before_start else None), \
                 patch.object(probe.isolation, 'stop_group', side_effect=stop), \
                 patch.object(probe.subprocess, 'Popen', return_value=child) as launch, \
                 patch.object(probe.signal, 'signal'), \
                 patch.object(probe.os, 'umask'), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(RuntimeError):
                    probe.main()
            self.assertNotIn('"status": "passed"', output.getvalue())
            self.assertEqual(owned.exists(), stop_fails)
            self.assertEqual(bool(stopped), not before_start)
            if before_start:
                launch.assert_not_called()

    def test_setup_failure_removes_temporary_directory(self):
        self.run_failure(before_start=True)

    def test_startup_failure_stops_child_before_removing_directory(self):
        self.run_failure()

    def test_failed_stop_preserves_directory_and_does_not_certify_success(self):
        self.run_failure(stop_fails=True)


if __name__ == '__main__':
    unittest.main()
