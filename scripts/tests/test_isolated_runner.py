"""Process cleanup contracts; only subprocess groups created by these tests."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import signal

spec = importlib.util.spec_from_file_location("runner", Path(__file__).parents[1] / "test_isolated_database.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CleanupTests(unittest.TestCase):
    def test_exited_leader_does_not_skip_live_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            program = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); open(sys.argv[1],'w').close()"
            child = subprocess.Popen([sys.executable, "-c", program, str(ready)], start_new_session=True)
            try:
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertTrue(ready.exists())
                self.assertTrue(runner.group_running(child.pid))
                runner.stop_group(child)
                self.assertFalse(runner.group_running(child.pid))
            finally:
                runner.stop_group(child)

    def test_live_group_escalates_after_leader_exits(self):
        child = mock.Mock(pid=12345)
        child.poll.return_value = 0
        with mock.patch.object(runner.os, "killpg") as kill, \
             mock.patch.object(runner, "group_running", side_effect=[True, False]), \
             mock.patch.object(runner.time, "monotonic", side_effect=[0, 0, 21, 21, 21]), \
             mock.patch.object(runner.time, "sleep"):
            runner.stop_group(child)
        self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])

    def test_already_gone_group_is_clean(self):
        child = mock.Mock(pid=12345)
        with mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError):
            runner.stop_group(child)
        child.wait.assert_called_once_with(timeout=1)


if __name__ == "__main__":
    unittest.main()
