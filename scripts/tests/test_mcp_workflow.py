"""The deployment check must fail before unsafe reuse and reject false success."""
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location('mcp_check', Path(__file__).resolve().parents[1] / 'check_mcp_workflow.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class Peer:
    def __init__(self, missing=(), failure=None):
        self.missing, self.failure, self.calls = missing, failure, []

    def connect(self):
        return check.REQUIRED - set(self.missing)

    def call(self, name, args, expect_error=False):
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
        report = {'success': False}
        with tempfile.TemporaryDirectory() as root:
            check.exercise(peer, 'example_smoke', Path(root), 1, report)
        return report

    def test_complete_workflow_scopes_all_calls(self):
        peer = Peer()
        self.assertTrue(self.exercise(peer)['success'])
        for name, args in peer.calls:
            if name != 'create_database':
                self.assertEqual(args['db'], 'example_smoke')
        self.assertFalse(any('delete' in name or 'drop' in name for name, _ in peer.calls))

    def test_old_server_is_detected_before_database_creation(self):
        peer = Peer(missing=['kg_create'])
        with self.assertRaisesRegex(RuntimeError, 'missing tools'):
            self.exercise(peer)
        self.assertEqual(peer.calls, [])

    def test_existing_database_is_never_reused(self):
        peer = Peer(failure='existing')
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            self.exercise(peer)
        self.assertEqual([name for name, _ in peer.calls], ['create_database'])

    def test_partial_success_never_passes(self):
        for failure in ['overwrite', 'job', 'degraded', 'retrieval', 'traversal', 'embeddings']:
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.exercise(Peer(failure=failure))

    def test_invalid_timeout_causes_no_calls(self):
        for timeout in [0, -1, float('inf'), float('nan')]:
            peer = Peer()
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                check.exercise(peer, 'example_smoke', Path(root), timeout, {})
            self.assertEqual(peer.calls, [])


class Response(io.BytesIO):
    def __init__(self, data, content_type):
        super().__init__(data)
        self.headers = Message()
        self.headers['Content-Type'] = content_type
        self.status = 200


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        return self.response


class Framing(unittest.TestCase):
    def request(self, data, content_type='text/event-stream'):
        client = check.McpClient('http://localhost/mcp', 'test-token')
        client.opener = Opener(Response(data, content_type))
        return client.request('tools/list', {})

    def test_empty_rmcp_retry_event_and_multiline_response(self):
        data = (b'data: \nid: 0\nretry: 3000\n\n'
                b'data: {"jsonrpc":"2.0",\n'
                b'data: "id":1,"result":{"tools":[]}}\n\n')
        self.assertEqual(self.request(data), {'tools': []})

    def test_json_response(self):
        data = json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': {'tools': []}}).encode()
        self.assertEqual(self.request(data, 'application/json'), {'tools': []})

    def test_truncated_stream_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, 'ended without'):
            self.request(b'data: \n\n')

    def test_wrong_response_id_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.request(b'{"id":99,"result":{}}', 'application/json')

    def test_redirects_are_never_followed_with_credentials(self):
        class Request:
            full_url = 'http://localhost/mcp'
        with self.assertRaises(HTTPError) as caught:
            check.NoRedirect().redirect_request(Request(), None, 307, 'redirect', {}, 'https://elsewhere/mcp')
        caught.exception.close()


if __name__ == '__main__':
    unittest.main()
