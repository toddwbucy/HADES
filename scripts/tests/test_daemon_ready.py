"""Private Unix peers exercise readiness framing and deadline failures."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location('ready', Path(__file__).resolve().parents[1] / 'check_daemon_ready.py')
ready = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ready)
GOOD = {'request_id':'install-readiness','success':True,'error':None,'error_code':None,
        'data':{'database':'fixture','arangodb':{'reader_ok':True,'writer_ok':True,'status':'healthy'}}}


class Readiness(unittest.TestCase):
    def run_peer(self, response, succeeds=False, stall=False):
        with tempfile.TemporaryDirectory(prefix='hades-ready-') as root:
            path = str(Path(root) / 'peer.sock')
            failures = []
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
                server.listen(1)
                server.settimeout(1)
                def serve():
                    try:
                        with server.accept()[0] as peer:
                            peer.settimeout(1)
                            def receive(n):
                                value = b''
                                while len(value) < n:
                                    part = peer.recv(n-len(value))
                                    if not part:
                                        raise AssertionError('incomplete request')
                                    value += part
                                return value
                            length = struct.unpack('!I', receive(4))[0]
                            request = json.loads(receive(length))
                            assert request['command'] == 'db.health'
                            assert request['session'] == 'admin'
                            if stall:
                                time.sleep(.3)
                            else:
                                peer.sendall(response)
                    except Exception as error:
                        failures.append(error)
                thread = threading.Thread(target=serve)
                thread.start()
                start = time.monotonic()
                try:
                    if succeeds:
                        ready.wait_ready(path, 'fixture', .2)
                    else:
                        with self.assertRaises(RuntimeError):
                            ready.wait_ready(path, 'fixture', .12)
                    self.assertLess(time.monotonic()-start, .8)
                finally:
                    thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])

    def test_healthy_response(self):
        payload = json.dumps(GOOD).encode()
        self.run_peer(struct.pack('!I', len(payload))+payload, succeeds=True)

    def test_invalid_frames(self):
        for frame in [b'', b'\0\0', struct.pack('!I', ready.MAX_RESPONSE+1),
                      struct.pack('!I', 0), struct.pack('!I', 10)+b'{}',
                      struct.pack('!I', 1)+b'{', struct.pack('!I', 2)+b'[]']:
            with self.subTest(frame=frame):self.run_peer(frame)

    def test_unhealthy_or_wrong_response(self):
        for change in ['success', 'request_id', 'database', 'reader_ok', 'writer_ok', 'status']:
            result = copy.deepcopy(GOOD)
            if change in ['success','request_id']:result[change] = False
            elif change == 'database':result['data'][change] = 'wrong'
            else:result['data']['arangodb'][change] = False
            payload = json.dumps(result).encode()
            with self.subTest(change=change):self.run_peer(struct.pack('!I', len(payload))+payload)

    def test_accepting_but_stalled_process_is_not_ready(self):
        self.run_peer(b'', stall=True)

    def test_missing_socket(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                ready.wait_ready(str(Path(root)/'missing.sock'), 'fixture', .02)

    def test_invalid_deadlines(self):
        for value in [0, -1, 121, float('inf'), float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ready.wait_ready('/unused', 'fixture', value)


if __name__ == '__main__':
    unittest.main()
