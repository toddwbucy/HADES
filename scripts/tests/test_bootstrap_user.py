"""Installer contracts use synthetic credentials and a private Unix HTTP server."""
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "install/setup-arangodb-user.sh"
ROOT_PASSWORD = 'fixture-root"\\\n密碼'
USER_PASSWORD = 'fixture-user"\\\n密碼'


@contextmanager
def fixture(statuses, *, block=False, disconnect=False):
    requests = []
    arrived, resume = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.command, self.path, json.loads(body), self.headers["Authorization"]))
            arrived.set()
            if block:
                resume.wait(35)
            if disconnect:
                self.close_connection = True
                return
            payload = USER_PASSWORD.encode()
            try:
                self.send_response(statuses[min(len(requests)-1, len(statuses)-1)])
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
        do_PUT = do_POST

    with tempfile.TemporaryDirectory(prefix="hades-bootstrap-test-") as directory:
        root = Path(directory)
        scratch = root / "scratch"
        scratch.mkdir()
        server = socketserver.UnixStreamServer(str(root / "api.sock"), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        env = {**os.environ, "ARANGO_ROOT_PASSWORD": ROOT_PASSWORD,
               "HADES_PASSWORD": USER_PASSWORD, "ARANGO_SOCKET": str(root / "api.sock"),
               "TMPDIR": str(scratch)}
        try:
            yield env, requests, arrived, resume, scratch
        finally:
            resume.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=6)
            assert not thread.is_alive()


class BootstrapTests(unittest.TestCase):
    def run_script(self, env):
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=5)

    def assert_private(self, result, scratch):
        self.assertNotIn(USER_PASSWORD, result.stdout + result.stderr)
        self.assertNotIn(ROOT_PASSWORD, result.stdout + result.stderr)
        self.assertEqual(list(scratch.iterdir()), [])

    def test_special_passwords_and_default_grants(self):
        with fixture([201, 200, 200]) as (env, requests, _, _, scratch):
            result = self.run_script(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(requests[0][2], {"user": "hades", "passwd": USER_PASSWORD, "active": True})
            self.assertEqual([r[1] for r in requests], ["/_api/user", "/_api/user/hades/database/_system", "/_api/user/hades/database/%2A"])
            expected = "Basic " + base64.b64encode(("root:" + ROOT_PASSWORD).encode()).decode()
            self.assertTrue(all(r[3] == expected for r in requests))
            self.assert_private(result, scratch)

    def test_existing_user_grants_are_untouched(self):
        with fixture([409]) as (env, requests, _, _, scratch):
            result = self.run_script(env)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(requests), 1)
            self.assertIn("Leaving grants untouched", result.stdout)
            self.assert_private(result, scratch)

    def test_http_failures_are_sanitized(self):
        for statuses, count in [([400], 1), ([201, 403], 2)]:
            with self.subTest(statuses=statuses), fixture(statuses) as (env, requests, _, _, scratch):
                result = self.run_script(env)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(len(requests), count)
                self.assertIn(f"HTTP {statuses[-1]}", result.stderr)
                self.assert_private(result, scratch)

    def test_transport_failure_is_sanitized(self):
        with fixture([], disconnect=True) as (env, requests, _, _, scratch):
            result = self.run_script(env)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len(requests), 1)
            self.assertIn("request failed", result.stderr)
            self.assert_private(result, scratch)

    def test_stalled_response_hits_total_request_deadline(self):
        with fixture([201], block=True) as (env, requests, _, _, scratch):
            start = time.monotonic()
            result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=34)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len(requests), 1)
            self.assertIn("TimeoutError", result.stderr)
            self.assertLess(time.monotonic() - start, 34)
            self.assert_private(result, scratch)

    def test_credentials_absent_from_argv_and_interrupt_leaves_no_files(self):
        with fixture([201], block=True) as (env, _, arrived, resume, scratch):
            child = subprocess.Popen(["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(arrived.wait(3))
                command = Path(f"/proc/{child.pid}/cmdline").read_bytes()
                self.assertNotIn(ROOT_PASSWORD.encode(), command)
                self.assertNotIn(USER_PASSWORD.encode(), command)
                child.send_signal(signal.SIGINT)
                stdout, stderr = child.communicate(timeout=3)
                self.assertEqual(child.returncode, 130, stderr)
                self.assert_private(subprocess.CompletedProcess([], child.returncode, stdout, stderr), scratch)
            finally:
                resume.set()
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
