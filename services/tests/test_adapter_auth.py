"""The adapter never authenticates as a user nobody named, and says who failed (#199)."""
import http.server
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from test_adapter_redirects import server

PASSWORD = 'synthetic-secret-199'


class Recorder(http.server.BaseHTTPRequestHandler):
    """Answers every request with one canned status and body, recording headers."""
    status = 200
    body = b'{"error":false,"result":[],"hasMore":false}'
    hits = []

    def do_GET(self):
        type(self).hits.append(self.headers.get('Authorization'))
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(self.body)
    do_POST = do_PUT = do_GET

    def log_message(self, *args):
        pass


def recorder(status, body):
    return type('R', (Recorder,), {'status': status, 'body': body, 'hits': []})


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv('ARANGO_HOST', '127.0.0.1')
    monkeypatch.delenv('ARANGO_USERNAME', raising=False)
    monkeypatch.delenv('ARANGO_PASSWORD', raising=False)
    return monkeypatch


@pytest.mark.parametrize('username', [None, ''])
def test_password_without_username_refused_before_any_request(env, username):
    handler = recorder(200, b'{}')
    env.setenv('ARANGO_PASSWORD', PASSWORD)
    if username is not None:
        env.setenv('ARANGO_USERNAME', username)
    with server(handler) as peer:
        env.setenv('ARANGO_PORT', str(peer.server_port))
        with pytest.raises(SystemExit) as refused:
            w.arango('fixture', 'version', method='GET')
    message = str(refused.value)
    assert 'ARANGO_USERNAME' in message
    assert PASSWORD not in message
    assert 'root' not in message
    assert handler.hits == []


def test_no_password_sends_no_auth_header(env):
    handler = recorder(200, b'{"version":"fixture"}')
    with server(handler) as peer:
        env.setenv('ARANGO_PORT', str(peer.server_port))
        assert w.arango('fixture', 'version', method='GET') == {'version': 'fixture'}
    assert handler.hits == [None]


def test_named_user_is_the_one_sent(env):
    import base64
    handler = recorder(200, b'{}')
    env.setenv('ARANGO_PASSWORD', PASSWORD)
    env.setenv('ARANGO_USERNAME', 'hades')
    with server(handler) as peer:
        env.setenv('ARANGO_PORT', str(peer.server_port))
        w.arango('fixture', 'version', method='GET')
    [header] = handler.hits
    assert base64.b64decode(header.split()[1]).decode() == f'hades:{PASSWORD}'


def test_401_names_user_code_and_errornum_but_not_password(env, monkeypatch, capsys):
    body = b'{"error":true,"code":401,"errorNum":11,"errorMessage":"not authorized to execute this request"}'
    handler = recorder(401, body)
    env.setenv('ARANGO_PASSWORD', PASSWORD)
    env.setenv('ARANGO_USERNAME', 'hades')
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', '/unused', '--dry-run'])
    with server(handler) as peer:
        env.setenv('ARANGO_PORT', str(peer.server_port))
        with pytest.raises(w.AdapterError) as failed:
            w.document_keys_by_rel('fixture')
        # The whole CLI, too: the cause has to reach stderr, not stop at the exception.
        assert w.main() == 1
    message = str(failed.value)
    assert "user 'hades'" in message and 'HTTP 401' in message and 'errorNum 11' in message
    output = capsys.readouterr()
    assert "authentication failed for user 'hades'" in output.err
    assert 'HTTP 401' in output.err
    for text in (message, output.err, output.out):
        assert PASSWORD not in text


def test_non_json_401_keeps_code_and_user(env):
    handler = recorder(401, b'<html>no</html>')
    env.setenv('ARANGO_PASSWORD', PASSWORD)
    env.setenv('ARANGO_USERNAME', 'hades')
    with server(handler) as peer:
        env.setenv('ARANGO_PORT', str(peer.server_port))
        response = w.arango('fixture', 'cursor', {'query': 'RETURN 1'})
    assert response == {'error': True, 'code': 401}
    with pytest.raises(w.AdapterError) as failed:
        w.checked(response, 'scope read')
    message = str(failed.value)
    assert 'HTTP 401' in message and "user 'hades'" in message
    assert PASSWORD not in message


def test_other_failures_carry_codes_but_not_backend_text():
    quoted = 'FOR secret IN collection'
    with pytest.raises(w.AdapterError) as failed:
        w.checked({'error': True, 'code': 400, 'errorNum': 1501, 'errorMessage': quoted}, 'scope read')
    message = str(failed.value)
    assert message == 'scope read failed (HTTP 400, errorNum 1501)'
    assert quoted not in message


@pytest.mark.parametrize('response', [None, [], {'error': True, 'code': 'x', 'errorNum': [1]}])
def test_malformed_failures_stay_terse(response):
    with pytest.raises(w.AdapterError) as failed:
        w.checked(response, 'scope read')
    assert str(failed.value) == 'scope read failed'
