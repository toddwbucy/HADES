"""Read-only backup input, bounded private restore; no production endpoints."""
import argparse
import gzip
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dump-dir', type=Path, required=True)
parser.add_argument('--bin-dir', type=Path, required=True)
args = parser.parse_args()
SOURCE = args.dump_dir.resolve()
BIN = args.bin_dir.resolve()
HELPER = Path(__file__).resolve().parents[3] / 'scripts/test_isolated_database.py'
for name in ('arangod', 'arangorestore'):
    if not (BIN / name).is_file():
        parser.error('missing existing binary: ' + name)
spec = importlib.util.spec_from_file_location('isolation', HELPER)
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)
os.umask(0o077)
root = Path(tempfile.mkdtemp(prefix='hades-backup-restore-', dir='/tmp'))
print(root, flush=True)
isolation.lower_priority()
copy = root / 'dump'
copy.mkdir()
files = sorted(SOURCE.iterdir())
assert len(files) <= 100
hashes = {}
for source in files:
    info = source.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_size <= 1024 * 1024
    with source.open('rb') as stream:
        data = stream.read(1024 * 1024 + 1)
    assert len(data) == info.st_size
    hashes[source.name] = hashlib.sha256(data).hexdigest()
    (copy / source.name).write_bytes(data)
assert sum(p.stat().st_size for p in copy.iterdir()) <= 8 * 1024 * 1024
manifest = json.loads((copy / 'dump.json').read_text())
expected = {}
expanded = 0
for path in sorted(copy.glob('*.structure.json')):
    structure = json.loads(path.read_text())
    name = structure['parameters']['name']
    assert (name.startswith('wt_') or name == 'hades_schema') and name.replace('_', '').isalnum()
    assert name not in expected
    with gzip.open(path.with_name(path.name.replace('.structure.json', '.data.json.gz')), 'rb') as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    assert len(data) <= 16 * 1024 * 1024
    expanded += len(data)
    assert expanded <= 64 * 1024 * 1024
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert len(rows) <= 10000 and all(isinstance(row, dict) and '_key' in row for row in rows)
    assert len({row['_key'] for row in rows}) == len(rows)
    expected[name] = (structure, rows)
assert len(expected) == 20
endpoint = root / 'arango.sock'
env = {'PATH': os.environ['PATH'], 'HOME': str(root), 'LANG': 'C', 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
children = []
report = {'scope': 'One historical WeaverTools logical dump restored privately; not current production recovery certification',
          'source_created_at': manifest.get('createdAt'), 'source_files': len(hashes),
          'source_sha256': hashes, 'expanded_data_bytes': expanded}


def interrupted(signum, frame):
    raise RuntimeError('private restore interrupted')


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)


def api(method, path, body=None, status=200):
    connection = http.client.HTTPConnection('localhost', timeout=10)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.settimeout(10)
    connection.sock.connect(str(endpoint))
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read(32 * 1024 * 1024 + 1)
        assert len(raw) <= 32 * 1024 * 1024
        assert response.status == status, f'private API returned status {response.status}'
        return json.loads(raw) if raw else None
    finally:
        connection.close()


def db(method, path, body=None, status=200):
    return api(method, '/_db/backup_restore_fixture' + path, body, status)


def canonical(rows):
    return [{k: v for k, v in row.items() if k != '_rev'} for row in sorted(rows, key=lambda row: row['_key'])]


def indexes(values):
    return sorted(json.dumps({k: value[k] for k in ('type', 'fields', 'unique', 'sparse') if k in value}, sort_keys=True)
                  for value in values if value['type'] not in ('primary', 'edge'))


try:
    flags = ['--configuration', 'none', '--database.directory', str(root / 'data'),
             '--server.endpoint', 'unix://' + str(endpoint), '--server.authentication', 'false',
             '--javascript.enabled', 'false', '--foxx.queues', 'false', '--server.statistics', 'false',
             '--server.minimal-threads', '4', '--server.maximal-threads', '8', '--server.io-threads', '1',
             '--rocksdb.block-cache-size', '67108864', '--rocksdb.total-write-buffer-size', '67108864',
             '--rocksdb.write-buffer-size', '16777216', '--rocksdb.max-background-jobs', '2',
             '--arangosearch.threads', '1', '--arangosearch.threads-limit', '1', '--log.output', '-']
    with (root / 'server.log').open('w') as output:
        server = subprocess.Popen([str(BIN / 'arangod'), *flags], env=env, cwd=root, stdout=output,
                                  stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=isolation.bounded_process)
    children.append(server)
    deadline = time.monotonic() + 60
    while True:
        assert server.poll() is None, 'private server exited'
        try:
            report['server_version'] = api('GET', '/_api/version')['version']
            break
        except (OSError, http.client.HTTPException):
            if time.monotonic() > deadline:
                raise RuntimeError('private startup deadline')
            time.sleep(.1)
    api('POST', '/_api/database', {'name': 'backup_restore_fixture'}, status=201)
    command = [str(BIN / 'arangorestore'), '--configuration', 'none', '--server.endpoint', 'unix://' + str(endpoint),
               '--server.authentication', 'false', '--server.database', 'backup_restore_fixture', '--threads', '1',
               '--include-system-collections', 'false', '--input-directory', str(copy)]
    started = time.monotonic()
    with (root / 'restore.log').open('w') as output:
        client = subprocess.Popen(command, env=env, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                  start_new_session=True, preexec_fn=isolation.bounded_process)
    children.append(client)
    assert client.wait(timeout=120) == 0, 'private restore failed; inspect protected log'
    report['restore_seconds'] = time.monotonic() - started
    counts = {'collections': 0, 'documents': 0, 'edge_collections': 0, 'edges': 0, 'dangling_edges': 0, 'secondary_indexes': 0}
    for name, (structure, rows) in expected.items():
        result = db('POST', '/_api/cursor', {'query': 'FOR d IN @@collection RETURN d',
                    'bindVars': {'@collection': name}, 'batchSize': 10000}, status=201)
        assert result['hasMore'] is False
        assert canonical(result['result']) == canonical(rows), 'restored document content mismatch'
        properties = db('GET', '/_api/collection/' + name + '/properties')
        assert properties['type'] == structure['parameters']['type']
        assert properties.get('schema') == structure['parameters'].get('schema')
        actual_indexes = db('GET', '/_api/index?collection=' + name)['indexes']
        assert indexes(actual_indexes) == indexes(structure['indexes'])
        counts['secondary_indexes'] += len(indexes(actual_indexes))
        counts['collections'] += 1
        counts['documents'] += len(rows)
        if properties['type'] == 3:
            counts['edge_collections'] += 1
            counts['edges'] += len(rows)
            result = db('POST', '/_api/cursor', {'query': 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO n RETURN n',
                        'bindVars': {'@collection': name}}, status=201)
            assert result['hasMore'] is False
            counts['dangling_edges'] += result['result'][0]
    # Recheck source bytes; this probe never writes to the source tree.
    for name, digest in hashes.items():
        with (SOURCE / name).open('rb') as stream:
            assert hashlib.sha256(stream.read(1024 * 1024 + 1)).hexdigest() == digest
    report['checks'] = ['all document fields except server revision match dump', 'all collection types and schemas match',
                        'secondary index definitions match', 'source file hashes unchanged']
    report['counts'] = counts
    report['binary_sha256'] = {}
    for name in ('arangod', 'arangorestore'):
        with (BIN / name).open('rb') as stream:
            report['binary_sha256'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['isolation_helper_sha256'] = hashlib.sha256(HELPER.read_bytes()).hexdigest()
    for child in reversed(children):
        isolation.stop_group(child)
    children.clear()
    report['owned_processes_stopped'] = True
    report['status'] = 'passed'
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'server_version', 'source_created_at', 'restore_seconds', 'counts', 'checks')}), flush=True)
finally:
    for child in reversed(children):
        isolation.stop_group(child)
