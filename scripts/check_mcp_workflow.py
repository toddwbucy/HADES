#!/usr/bin/env python3
"""Exercise a deployed MCP endpoint using a NEW database and synthetic document.

Requires explicit endpoint, token file, database name, and host-side fixture root.
Leaves its isolated database and fixture in place for inspection. Never drops data.
"""
import argparse
import json
import math
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_RESPONSE = 4 * 1024 * 1024
REQUIRED = {'create_database', 'db_schema_init', 'kg_create', 'kg_add_node',
            'kg_add_edge', 'db_get', 'db_count', 'graph_traverse', 'ingest_start',
            'ingest_status', 'db_query'}
FIXTURE = ('The amber archive stores research notes. Every note has a stable source '
           'identifier. The archive preserves source revisions when a note changes.\n')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Prevent a redirect from forwarding the bearer credential elsewhere."""
        raise HTTPError(req.full_url, code, 'MCP redirects are refused', headers, fp)


class McpClient:
    def __init__(self, url, token, timeout=30):
        """Accept HTTPS or local HTTP, with a bounded socket wait per request."""
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('MCP URL must be HTTP(S), with no embedded credentials')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('plain HTTP is allowed only for loopback endpoints; use HTTPS or an SSH tunnel')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('request timeout must be positive and finite')
        self.url, self.token, self.timeout = url, token, timeout
        self.session, self.sequence = None, 0
        self.protocol = '2025-03-26'
        # Local HTTP must not send its credential through an environment proxy.
        handlers = [NoRedirect()]
        if parsed.scheme == 'http':
            handlers.append(ProxyHandler({}))
        self.opener = build_opener(*handlers)

    def request(self, method, params, notification=False):
        """Exchange one JSON-RPC message and decode a bounded JSON or SSE reply."""
        self.sequence += 1
        payload = {'jsonrpc': '2.0', 'method': method, 'params': params}
        if not notification:
            payload['id'] = self.sequence
        headers = {'Authorization': 'Bearer ' + self.token,
                   'Content-Type': 'application/json',
                   'Accept': 'application/json, text/event-stream'}
        if self.session:
            headers['Mcp-Session-Id'] = self.session
            headers['MCP-Protocol-Version'] = self.protocol
        req = Request(self.url, data=json.dumps(payload).encode(), headers=headers, method='POST')
        with self.opener.open(req, timeout=self.timeout) as response:
            if response.headers.get('mcp-session-id'):
                self.session = response.headers['mcp-session-id']
            if notification:
                if response.status != 202:
                    raise RuntimeError('MCP notification was not accepted')
                return None
            if response.headers.get_content_type() == 'text/event-stream':
                size, data = 0, []
                result = None
                while True:
                    line = response.readline(MAX_RESPONSE + 1)
                    size += len(line)
                    if size > MAX_RESPONSE:
                        raise RuntimeError('MCP response exceeds size limit')
                    if not line:
                        break
                    line = line.rstrip(b'\r\n')
                    if line.startswith(b'data:'):
                        data.append(line[5:].lstrip(b' '))
                    elif not line and data:
                        event_data = b'\n'.join(data)
                        data = []
                        if not event_data.strip():
                            continue  # rmcp opens streams with an empty retry event.
                        event = json.loads(event_data)
                        if not isinstance(event, dict):
                            raise RuntimeError('MCP SSE event must be a JSON object')
                        if event.get('id') == payload['id']:
                            result = event
                            break
                if result is None:
                    raise RuntimeError('MCP stream ended without the requested response')
            else:
                raw = response.read(MAX_RESPONSE + 1)
                if len(raw) > MAX_RESPONSE:
                    raise RuntimeError('MCP response exceeds size limit')
                result = json.loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError('MCP response must be a JSON object')
        if result.get('id') != payload['id'] or 'error' in result or 'result' not in result:
            raise RuntimeError('MCP JSON-RPC request failed: ' + str(result))
        if not isinstance(result['result'], dict):
            raise RuntimeError('MCP result must be a JSON object')
        return result['result']

    def connect(self):
        """Initialize the MCP session and collect all advertised tool names."""
        result = self.request('initialize', {'protocolVersion': self.protocol,
                              'capabilities': {}, 'clientInfo': {'name': 'hades-mcp-check', 'version': '1'}})
        self.protocol = result['protocolVersion']
        self.request('notifications/initialized', {}, notification=True)
        tools, cursor = set(), None
        while True:
            page = self.request('tools/list', {'cursor': cursor} if cursor else {})
            entries = page.get('tools')
            if not isinstance(entries, list) or any(
                not isinstance(tool, dict)
                or not isinstance(tool.get('name'), str)
                or not tool['name']
                for tool in entries
            ):
                raise RuntimeError('MCP tools page must contain an array of objects with nonempty string names')
            tools.update(tool['name'] for tool in entries)
            cursor = page.get('nextCursor')
            if cursor is not None and not isinstance(cursor, str):
                raise RuntimeError('MCP tools page nextCursor must be a string')
            if not cursor:
                return tools

    def call(self, name, arguments, expect_error=False):
        """Unwrap a HADES tool response, requiring success or an expected rejection."""
        result = self.request('tools/call', {'name': name, 'arguments': arguments})
        texts = [c['text'] for c in result.get('content', []) if c.get('type') == 'text']
        if len(texts) != 1:
            raise RuntimeError(name + ': missing HADES response envelope')
        try:
            envelope = json.loads(texts[0])
        except ValueError:
            raise RuntimeError(name + ': ' + texts[0]) from None
        if not isinstance(envelope, dict):
            raise RuntimeError(name + ': HADES response envelope must be a JSON object')
        failed = result.get('isError') is True or envelope.get('success') is not True
        if expect_error:
            if not failed:
                raise RuntimeError(name + ': expected rejection, but operation succeeded')
            return envelope
        if failed:
            raise RuntimeError(name + ': ' + str(envelope))
        return envelope['data']


def exercise(client, database, fixture_root, timeout, report):
    """Fail on old tool surfaces or existing databases before any corpus writes."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', database):
        raise ValueError('database must be an explicit valid new database name')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('ingestion timeout must be positive and finite')
    if not fixture_root.is_dir():
        raise ValueError('fixture root must exist on this host and be permitted by the MCP server')
    available = client.connect()
    if REQUIRED - available:
        raise RuntimeError('Endpoint missing tools: ' + ', '.join(sorted(REQUIRED - available)))
    report['tools'] = sorted(available)
    created = client.call('create_database', {'name': database})
    if created.get('created') is not True or created.get('name') != database:
        raise RuntimeError('Endpoint did not confirm creation of the requested new database')
    report['created_database'] = database
    scoped = lambda **values: {'db': database, **values}
    client.call('db_schema_init', scoped(seed='empty'))
    client.call('kg_create', scoped(name='research'))
    for key, title in [('human', 'Human note'), ('agent', 'Agent note')]:
        client.call('kg_add_node', scoped(graph='research', key=key, data={'title': title}))
    client.call('kg_add_edge', scoped(graph='research', key='related', **{
        'from': 'research_nodes/human', 'to': 'research_nodes/agent', 'data': {'relation': 'related_to'}}))
    rejected = client.call('kg_add_node', scoped(graph='research', key='human', data={'title': 'Must not replace'}), expect_error=True)
    diagnostic = str(rejected.get('error', '')).lower()
    if rejected.get('error_code') != 'CONFLICT' and not any(word in diagnostic for word in ('duplicate', 'unique constraint')):
        raise RuntimeError('Duplicate check failed for a reason other than a key conflict')
    original = client.call('db_get', scoped(collection='research_nodes', key='human', fields=['_id', 'title']))
    if original.get('title') != 'Human note':
        raise RuntimeError('Duplicate insertion changed the existing node')
    report['authored_writes_and_duplicate_protection'] = True
    folder = Path(tempfile.mkdtemp(prefix='hades-mcp-check-', dir=fixture_root))
    fixture = folder / 'mcp_fixture.txt'
    fixture.write_text(FIXTURE, encoding='utf-8')
    # Only this newly created, non-sensitive fixture is made daemon-readable.
    # Explicit modes also work when the invoking operator has umask 0077.
    fixture.chmod(0o644)
    folder.chmod(0o755)
    report['fixture_path'] = str(fixture)
    job = client.call('ingest_start', scoped(path=str(fixture)))
    report['job_id'] = job['job_id']
    deadline = time.monotonic() + timeout
    while True:
        status = client.call('ingest_status', scoped(job_id=job['job_id']))
        report['ingest_status'] = status.get('status')
        if status.get('status') == 'completed':
            result = status.get('result', {})
            data = result.get('data', {})
            if result.get('success') is not True or data.get('failed') != 0 or data.get('enrichment_degraded') is not False:
                raise RuntimeError('Ingestion completed with failures or degraded enrichment')
            break
        if status.get('status') not in ('starting', 'running'):
            raise RuntimeError('Ingestion did not succeed: ' + str(status))
        if time.monotonic() >= deadline:
            raise RuntimeError('Ingestion wait timed out; job may still run; inspect recorded job_id')
        time.sleep(min(.5, max(0, deadline - time.monotonic())))
    found = client.call('db_query', scoped(text='How does the amber archive preserve source revisions?', collection='default', limit=3))
    hits = found.get('results', [])
    if not any(h.get('parent_key') == 'mcp_fixture' and 'amber archive' in h.get('text', '') for h in hits):
        raise RuntimeError('Semantic retrieval did not return the ingested fixture')
    report['semantic_retrieval'] = True
    client.call('kg_add_edge', scoped(graph='research', key='supported', **{
        'from': 'research_nodes/agent', 'to': 'documents/mcp_fixture', 'data': {'relation': 'supported_by'}}))
    traversed = client.call('graph_traverse', scoped(graph='research', start='research_nodes/human', max_depth=2, fields=['_id', 'title']))
    ids = {r.get('vertex', {}).get('_id') for r in traversed.get('results', [])}
    if not {'research_nodes/agent', 'documents/mcp_fixture'} <= ids:
        raise RuntimeError('Traversal did not connect authored nodes to the ingested document')
    report['traversal_to_ingested_document'] = True
    report['counts'] = {col: client.call('db_count', scoped(collection=col)) for col in
                        ['research_nodes', 'research_edges', 'documents', 'chunks', 'embeddings']}
    counts = {col: row.get('count') for col, row in report['counts'].items()}
    if counts['research_nodes'] != 2 or counts['research_edges'] != 2 or counts['documents'] != 1:
        raise RuntimeError('Unexpected authored or document row counts')
    if not isinstance(counts['chunks'], int) or counts['chunks'] < 1 or counts['embeddings'] != counts['chunks']:
        raise RuntimeError('Stored chunks and embeddings are missing or inconsistent')
    report['success'] = True


def main():
    """Run the explicit deployment check and emit a JSON success or failure report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--token-file', required=True, type=Path)
    parser.add_argument('--database', required=True, help='NEW database in the configured provisioning namespace')
    parser.add_argument('--fixture-root', required=True, type=Path, help='existing local directory permitted for server ingestion')
    parser.add_argument('--timeout', type=float, default=120, help='ingestion polling budget in seconds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'success': False, 'database': args.database}
    try:
        tokens = [line.strip() for line in args.token_file.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
        if not tokens:
            raise ValueError('token file contains no tokens')
        exercise(McpClient(args.url, tokens[0]), args.database, args.fixture_root.resolve(), args.timeout, report)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report['error'] = str(error)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end='')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
