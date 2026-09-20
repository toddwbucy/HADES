"""Redirects must not escape the adapter endpoint policy, with or without auth."""
import contextlib
import http.server
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as writer


@contextlib.contextmanager
def server(handler):
    peer = http.server.HTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=lambda: peer.serve_forever(poll_interval=.01))
    thread.start()
    try:
        yield peer
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize('status', [301, 302, 303, 307, 308])
@pytest.mark.parametrize('authenticated', [False, True])
@pytest.mark.parametrize('method', ['GET', 'POST'])
def test_redirect_rejected_without_destination_traffic(monkeypatch, status, authenticated, method):
    hits = []
    class Target(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.headers.get('Authorization'))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"error":false}')
        do_POST = do_GET
        def log_message(self, *args):
            pass
    with server(Target) as destination:
        class Origin(Target):
            def do_GET(self):
                self.send_response(status)
                self.send_header('Location', f'http://localhost:{destination.server_port}/sink')
                self.end_headers()
                # A redirect body must never turn a refusal into success.
                self.wfile.write(b'{"error":false}')
            do_POST = do_GET
        with server(Origin) as origin:
            monkeypatch.setenv('ARANGO_HOST', '127.0.0.1')
            monkeypatch.setenv('ARANGO_PORT', str(origin.server_port))
            monkeypatch.setenv('ARANGO_USERNAME', 'fixture')
            if authenticated:
                monkeypatch.setenv('ARANGO_PASSWORD', 'synthetic-only')
            else:
                monkeypatch.delenv('ARANGO_PASSWORD', raising=False)
            result = writer.arango('fixture', 'version', method=method)
            assert result['error'] is True
            assert result['code'] == status
            assert hits == []


def test_direct_response_preserved(monkeypatch):
    class Direct(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version":"fixture"}')
        def log_message(self, *args):
            pass
    with server(Direct) as peer:
        monkeypatch.setenv('ARANGO_HOST', '127.0.0.1')
        monkeypatch.setenv('ARANGO_PORT', str(peer.server_port))
        monkeypatch.delenv('ARANGO_PASSWORD', raising=False)
        assert writer.arango('fixture', 'version', method='GET') == {'version':'fixture'}
