"""The deployment check must fail before unsafe reuse and reject false success."""
import importlib.util
import io
import json
import os
from email.message import Message
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('mcp_check', Path(__file__).resolve().parents[1] / 'check_mcp_workflow.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class Peer:
    def __init__(self, missing=(), failure=None):
        """Configure a deterministic MCP peer with optional failure injection."""
        self.missing, self.failure, self.calls = missing, failure, []

    def connect(self):
        """Advertise the tool set for a current or deliberately old server."""
        return check.REQUIRED - set(self.missing)

    def call(self, name, args, expect_error=False):
        """Record each call and supply the selected workflow outcome."""
        self.calls.append((name, args))
        if name == 'create_database':
            if self.failure == 'existing':
                raise RuntimeError('database already exists')
            return {'created': True, 'name': args['name']}
        if expect_error:
            return {'success': False, 'error_code': 'CONFLICT', 'error': 'duplicate key'}
        if name == 'db_get':
            return {'title': 'Wrong' if self.failure == 'overwrite' else 'Human note'}
        if name == 'ingest_start':
            return {'job_id': '1234567890abcdef'}
        if name == 'ingest_status':
            return {'status': 'failed' if self.failure == 'job' else 'completed',
                    'result': {'success': True, 'data': {'failed': int(self.failure == 'degraded'),
                               'enrichment_degraded': self.failure == 'degraded'}}}
        if name == 'db_query':
            return {'results': [] if self.failure == 'retrieval' else
                    [{'parent_key': 'mcp_fixture', 'text': check.FIXTURE}]}
        if name == 'graph_traverse':
            return {'results': [] if self.failure == 'traversal' else
                    [{'vertex': {'_id': v}} for v in ['research_nodes/agent', 'documents/mcp_fixture']]}
        if name == 'db_count':
            count = 2 if args['collection'].startswith('research_') else 1
            if self.failure == 'embeddings' and args['collection'] == 'embeddings':
                count = 0
            return {'count': count}
        return {}


class Workflow(unittest.TestCase):
    def exercise(self, peer):
        """Run the fake-peer workflow with an isolated local fixture root."""
        report = {'success': False}
        with tempfile.TemporaryDirectory() as root:
            check.exercise(peer, 'example_smoke', Path(root), 1, report)
        return report

    def test_complete_workflow_scopes_all_calls(self):
        """Every database operation stays scoped and no cleanup deletes data."""
        peer = Peer()
        self.assertTrue(self.exercise(peer)['success'])
        for name, args in peer.calls:
            if name != 'create_database':
                self.assertEqual(args['db'], 'example_smoke')
        self.assertFalse(any('delete' in name or 'drop' in name for name, _ in peer.calls))

    def test_old_server_is_detected_before_database_creation(self):
        """Missing authoring tools must stop the run before writes."""
        peer = Peer(missing=['kg_create'])
        with self.assertRaisesRegex(RuntimeError, 'missing tools'):
            self.exercise(peer)
        self.assertEqual(peer.calls, [])

    def test_existing_database_is_never_reused(self):
        """A database conflict must not lead to schema or graph changes."""
        peer = Peer(failure='existing')
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            self.exercise(peer)
        self.assertEqual([name for name, _ in peer.calls], ['create_database'])

    def test_partial_success_never_passes(self):
        """Failures in any required workflow stage prevent a success report."""
        for failure in ['overwrite', 'job', 'degraded', 'retrieval', 'traversal', 'embeddings']:
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.exercise(Peer(failure=failure))

    def test_invalid_timeout_causes_no_calls(self):
        """Reject invalid polling budgets before contacting the server."""
        for timeout in [0, -1, float('inf'), float('nan')]:
            peer = Peer()
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                check.exercise(peer, 'example_smoke', Path(root), timeout, {})
            self.assertEqual(peer.calls, [])

    def test_fixture_is_daemon_readable_under_restrictive_umask(self):
        """A separate daemon account can read the new fixture, even with umask 0077."""
        peer = Peer()
        original_call = peer.call
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            root_path.chmod(0o755)
            root_mode = stat.S_IMODE(root_path.stat().st_mode)

            def inspect_fixture(name, args, **kwargs):
                """Check permissions at the moment ingestion receives the path."""
                if name == 'ingest_start':
                    fixture = Path(args['path'])
                    self.assertEqual(fixture.read_text(encoding='utf-8'), check.FIXTURE)
                    self.assertEqual(stat.S_IMODE(fixture.stat().st_mode), 0o644)
                    self.assertEqual(stat.S_IMODE(fixture.parent.stat().st_mode), 0o755)
                    self.assertEqual(fixture.parent.parent, root_path)
                return original_call(name, args, **kwargs)

            previous_umask = os.umask(0o077)
            try:
                with patch.object(peer, 'call', side_effect=inspect_fixture):
                    check.exercise(peer, 'example_smoke', root_path, 1, {})
            finally:
                os.umask(previous_umask)
            self.assertEqual(stat.S_IMODE(root_path.stat().st_mode), root_mode)


class Response(io.BytesIO):
    def __init__(self, data, content_type):
        """Provide the HTTP response interface used by the MCP parser."""
        super().__init__(data)
        self.headers = Message()
        self.headers['Content-Type'] = content_type
        self.status = 200


class Opener:
    def __init__(self, response):
        """Hold one response without opening a real network connection."""
        self.response = response

    def open(self, request, timeout):
        """Return the predetermined HTTP response."""
        return self.response


class Framing(unittest.TestCase):
    def request(self, data, content_type='text/event-stream'):
        """Parse one supplied HTTP body as a tools/list response."""
        client = check.McpClient('http://localhost/mcp', 'test-token')
        client.opener = Opener(Response(data, content_type))
        return client.request('tools/list', {})

    def test_empty_rmcp_retry_event_and_multiline_response(self):
        """Ignore empty retry events and join multi-line SSE data fields."""
        data = (b'data: \nid: 0\nretry: 3000\n\n'
                b'data: {"jsonrpc":"2.0",\n'
                b'data: "id":1,"result":{"tools":[]}}\n\n')
        self.assertEqual(self.request(data), {'tools': []})

    def test_json_response(self):
        """Accept ordinary JSON as well as the streaming transport."""
        data = json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {'tools': []}}).encode()
        self.assertEqual(self.request(data, 'application/json'), {'tools': []})

    def test_truncated_stream_is_not_success(self):
        """An SSE stream without the requested response must fail."""
        with self.assertRaisesRegex(RuntimeError, 'ended without'):
            self.request(b'data: \n\n')

    def test_wrong_response_id_is_rejected(self):
        """Do not accept another request's JSON response."""
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.request(b'{"id":99,"result":{}}', 'application/json')

    def test_redirects_are_never_followed_with_credentials(self):
        """A redirect must fail before any second request can carry the token."""
        class Request:
            full_url = 'http://localhost/mcp'
        with self.assertRaises(HTTPError) as caught:
            check.NoRedirect().redirect_request(Request(), None, 307, 'redirect', {}, 'https://elsewhere/mcp')
        caught.exception.close()

    def test_non_object_payloads_produce_failure_reports(self):
        """Malformed JSON and SSE shapes must still produce the CLI JSON report."""
        for value in [None, [], 'text', 7, True, {'id': 1, 'result': []}]:
            for content_type in ['application/json', 'text/event-stream']:
                with self.subTest(value=value, content_type=content_type), tempfile.TemporaryDirectory() as root:
                    data = json.dumps(value).encode()
                    if content_type == 'text/event-stream':
                        data = b'data: ' + data + b'\n\n'
                    token = Path(root) / 'token'
                    token.write_text('test-token')
                    output = Path(root) / 'report.json'
                    argv = ['check_mcp_workflow.py', '--url', 'http://localhost/mcp',
                            '--token-file', str(token), '--database', 'example_smoke',
                            '--fixture-root', root, '--output', str(output)]
                    opener = Opener(Response(data, content_type))
                    stdout = io.StringIO()
                    with patch('sys.argv', argv), patch('sys.stdout', stdout), \
                            patch.object(check, 'build_opener', return_value=opener):
                        self.assertEqual(check.main(), 1)
                    report = json.loads(output.read_text())
                    self.assertFalse(report['success'])
                    self.assertIn('must be a JSON object', report['error'])
                    self.assertEqual(json.loads(stdout.getvalue()), report)
                    self.assertNotIn('test-token', output.read_text())


class Discovery(unittest.TestCase):
    def test_malformed_tools_pages_produce_failure_reports_before_writes(self):
        """Invalid tool arrays, names, and cursors fail with a saved JSON report."""
        pages = [{}, {'tools': None}, {'tools': {}}, {'tools': 'tools'},
                 {'tools': [None]}, {'tools': ['name']}, {'tools': [{}]},
                 {'tools': [{'name': []}]}, {'tools': [{'name': 1}]},
                 {'tools': [{'name': ''}]}, {'tools': [], 'nextCursor': []}]
        for page in pages:
            with self.subTest(page=page), tempfile.TemporaryDirectory() as root:
                token = Path(root) / 'token'
                token.write_text('test-token')
                output = Path(root) / 'report.json'
                argv = ['check_mcp_workflow.py', '--url', 'http://localhost/mcp',
                        '--token-file', str(token), '--database', 'example_smoke',
                        '--fixture-root', root, '--output', str(output)]
                responses = [{'protocolVersion': '2025-03-26'}, None, page]
                stdout = io.StringIO()
                with patch('sys.argv', argv), patch('sys.stdout', stdout), \
                        patch.object(check.McpClient, 'request', side_effect=responses) as request:
                    self.assertEqual(check.main(), 1)
                self.assertEqual([call.args[0] for call in request.call_args_list],
                                 ['initialize', 'notifications/initialized', 'tools/list'])
                report = json.loads(output.read_text())
                self.assertFalse(report['success'])
                self.assertIn('MCP tools page', report['error'])
                self.assertEqual(json.loads(stdout.getvalue()), report)
                self.assertNotIn('test-token', output.read_text())

    def test_valid_paginated_tools_are_collected(self):
        """Valid pages preserve tool discovery and send the server's cursor back."""
        client = check.McpClient('http://localhost/mcp', 'test-token')
        responses = [{'protocolVersion': '2025-03-26'}, None,
                     {'tools': [{'name': 'kg_create'}], 'nextCursor': 'page-2'},
                     {'tools': [{'name': 'db_query'}]}]
        with patch.object(client, 'request', side_effect=responses) as request:
            self.assertEqual(client.connect(), {'kg_create', 'db_query'})
        self.assertEqual(request.call_args.args, ('tools/list', {'cursor': 'page-2'}))


class Transport(unittest.TestCase):
    def test_nonlocal_plain_http_is_rejected_before_network_setup(self):
        """Reject non-loopback HTTP and preserve URL credential validation."""
        for url in ['http://192.0.2.1/mcp', 'http://mcp.example/mcp',
                    'http://[2001:db8::1]/mcp', 'ftp://localhost/mcp',
                    'https:///mcp', 'https://user:secret@mcp.example/mcp',
                    'http://user:secret@localhost/mcp']:
            with self.subTest(url=url), patch.object(check, 'build_opener') as opener:
                with self.assertRaises(ValueError):
                    check.McpClient(url, 'test-token')
                opener.assert_not_called()

    def test_https_and_local_http_are_accepted(self):
        """Permit encrypted endpoints and local or tunneled HTTP endpoints."""
        for url in ['https://mcp.example/mcp', 'http://localhost/mcp',
                    'http://127.0.0.1:11443/mcp', 'http://[::1]/mcp']:
            with self.subTest(url=url):
                self.assertEqual(check.McpClient(url, 'test-token').url, url)

    def test_local_http_disables_environment_proxies(self):
        """A proxy must not move a local cleartext credential onto the network."""
        with patch.object(check, 'build_opener') as opener:
            check.McpClient('http://localhost/mcp', 'test-token')
        handlers = opener.call_args.args
        proxies = [handler.proxies for handler in handlers if isinstance(handler, check.ProxyHandler)]
        self.assertEqual(proxies, [{}])


if __name__ == '__main__':
    unittest.main()
