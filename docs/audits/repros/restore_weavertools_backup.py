"""Read-only backup input, bounded private restore; no production endpoints."""
import argparse
import gzip
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import socket
import stat
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dump-dir', type=Path, required=True)
parser.add_argument('--bin-dir', type=Path, required=True)
parser.add_argument('--staged-root', type=Path, help=argparse.SUPPRESS)
args = parser.parse_args()
SOURCE = args.dump_dir.resolve()
BIN = args.bin_dir.resolve()
os.umask(0o077)


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if args.staged_root is None:
    root = Path(tempfile.mkdtemp(prefix='hades-backup-restore-', dir='/tmp'))
    print(root, flush=True)
    execution = root / 'execution'
    execution.mkdir(mode=0o700)
    sources = {'probe.py': Path(__file__).resolve(),
               'helper.py': Path(__file__).resolve().parents[3] / 'scripts/test_isolated_database.py',
               'arangod': BIN / 'arangod', 'arangorestore': BIN / 'arangorestore',
               'icudtl.dat': BIN / 'icudtl.dat', 'icudtl_legacy.dat': BIN / 'icudtl_legacy.dat'}
    timezone_files = sorted((BIN / 'tzdata').iterdir())
    if not timezone_files or len(timezone_files) > 100:
        parser.error('missing or oversized timezone data set')
    sources.update({'tzdata/' + source.name: source for source in timezone_files})
    execution_hashes = {}
    for name, source in sources.items():
        if not source.is_file() or source.stat().st_size > 512 * 1024 * 1024:
            parser.error('missing or oversized execution input: ' + name)
        target = execution / name
        target.parent.mkdir(mode=0o700, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o500 if name.startswith('arango') else 0o400)
        execution_hashes[name] = file_hash(target)
    (root / 'execution-hashes.json').write_text(json.dumps(execution_hashes) + '\n')
    optimize = ['-OO' if sys.flags.optimize > 1 else '-O'] if sys.flags.optimize else []
    command = [sys.executable, *optimize, str(execution / 'probe.py'),
               '--dump-dir', str(SOURCE), '--bin-dir', str(execution), '--staged-root', str(root)]
    os.execve(sys.executable, command, {'PATH': os.environ['PATH'], 'HOME': str(root),
              'LANG': 'C', 'PYTHONDONTWRITEBYTECODE': '1'})

root = args.staged_root.resolve()
execution = root / 'execution'
if Path(__file__).resolve() != execution / 'probe.py' or BIN != execution:
    raise RuntimeError('staged execution paths differ from private manifest')
execution_hashes = json.loads((root / 'execution-hashes.json').read_text())
required = {'probe.py', 'helper.py', 'arangod', 'arangorestore', 'icudtl.dat', 'icudtl_legacy.dat'}
if not required < set(execution_hashes) or any(
    name not in required and (len(Path(name).parts) != 2 or Path(name).parts[0] != 'tzdata' or '..' in Path(name).parts)
    for name in execution_hashes
):
    raise RuntimeError('invalid execution manifest')
for name, expected_hash in execution_hashes.items():
    if file_hash(execution / name) != expected_hash:
        raise RuntimeError('execution input hash mismatch')
HELPER = execution / 'helper.py'
spec = importlib.util.spec_from_file_location('isolation', HELPER)
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)
isolation.lower_priority()
copy = root / 'dump'
copy.mkdir()
files = sorted(SOURCE.iterdir())
if not (len(files) <= 100):
    raise RuntimeError('private restore validation failed at check 75')
hashes = {}
for source in files:
    info = source.lstat()
    if not (stat.S_ISREG(info.st_mode) and info.st_size <= 1024 * 1024):
        raise RuntimeError('private restore validation failed at check 79')
    with source.open('rb') as stream:
        data = stream.read(1024 * 1024 + 1)
    if not (len(data) == info.st_size):
        raise RuntimeError('private restore validation failed at check 82')
    hashes[source.name] = hashlib.sha256(data).hexdigest()
    (copy / source.name).write_bytes(data)
if not (sum(p.stat().st_size for p in copy.iterdir()) <= 8 * 1024 * 1024):
    raise RuntimeError('private restore validation failed at check 85')
manifest = json.loads((copy / 'dump.json').read_text())
expected = {}
expanded = 0
for path in sorted(copy.glob('*.structure.json')):
    structure = json.loads(path.read_text())
    name = structure['parameters']['name']
    if not ((name.startswith('wt_') or name == 'hades_schema') and name.replace('_', '').isalnum()):
        raise RuntimeError('private restore validation failed at check 92')
    if not (name not in expected):
        raise RuntimeError('private restore validation failed at check 93')
    with gzip.open(path.with_name(path.name.replace('.structure.json', '.data.json.gz')), 'rb') as stream:
        data = stream.read(16 * 1024 * 1024 + 1)
    if not (len(data) <= 16 * 1024 * 1024):
        raise RuntimeError('private restore validation failed at check 96')
    expanded += len(data)
    if not (expanded <= 64 * 1024 * 1024):
        raise RuntimeError('private restore validation failed at check 98')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not (len(rows) <= 10000 and all(isinstance(row, dict) and '_key' in row for row in rows)):
        raise RuntimeError('private restore validation failed at check 100')
    if not (len({row['_key'] for row in rows}) == len(rows)):
        raise RuntimeError('private restore validation failed at check 101')
    expected[name] = (structure, rows)
if not (len(expected) == 20):
    raise RuntimeError('private restore validation failed at check 103')
endpoint = root / 'arango.sock'
env = {'PATH': os.environ['PATH'], 'HOME': str(root), 'LANG': 'C', 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'ICU_DATA': str(execution), 'ICU_DATA_LEGACY': str(execution), 'TZ_DATA': str(execution / 'tzdata')}
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
        if not (len(raw) <= 32 * 1024 * 1024):
            raise RuntimeError('private restore validation failed at check 130')
        if not (response.status == status):
            raise RuntimeError(f'private API returned status {response.status}')
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
        if not (server.poll() is None):
            raise RuntimeError('private server exited')
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
    restore_exit = client.wait(timeout=120)
    if not (restore_exit == 0):
        raise RuntimeError('private restore failed; inspect protected log')
    report['restore_seconds'] = time.monotonic() - started
    counts = {'collections': 0, 'documents': 0, 'edge_collections': 0, 'edges': 0, 'dangling_edges': 0, 'secondary_indexes': 0}
    for name, (structure, rows) in expected.items():
        result = db('POST', '/_api/cursor', {'query': 'FOR d IN @@collection RETURN d',
                    'bindVars': {'@collection': name}, 'batchSize': 10000}, status=201)
        if not (result['hasMore'] is False):
            raise RuntimeError('private restore validation failed at check 188')
        if not (canonical(result['result']) == canonical(rows)):
            raise RuntimeError('restored document content mismatch')
        properties = db('GET', '/_api/collection/' + name + '/properties')
        if not (properties['type'] == structure['parameters']['type']):
            raise RuntimeError('private restore validation failed at check 191')
        if not (properties.get('schema') == structure['parameters'].get('schema')):
            raise RuntimeError('private restore validation failed at check 192')
        actual_indexes = db('GET', '/_api/index?collection=' + name)['indexes']
        if not (indexes(actual_indexes) == indexes(structure['indexes'])):
            raise RuntimeError('private restore validation failed at check 194')
        counts['secondary_indexes'] += len(indexes(actual_indexes))
        counts['collections'] += 1
        counts['documents'] += len(rows)
        if properties['type'] == 3:
            counts['edge_collections'] += 1
            counts['edges'] += len(rows)
            result = db('POST', '/_api/cursor', {'query': 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO n RETURN n',
                        'bindVars': {'@collection': name}}, status=201)
            if not (result['hasMore'] is False):
                raise RuntimeError('private restore validation failed at check 203')
            counts['dangling_edges'] += result['result'][0]
    # Recheck source bytes; this probe never writes to the source tree.
    for name, digest in hashes.items():
        with (SOURCE / name).open('rb') as stream:
            if not (hashlib.sha256(stream.read(1024 * 1024 + 1)).hexdigest() == digest):
                raise RuntimeError('private restore validation failed at check 208')
    report['checks'] = ['all document fields except server revision match dump', 'all collection types and schemas match',
                        'secondary index definitions match', 'source file hashes unchanged']
    report['counts'] = counts
    for name, expected_hash in execution_hashes.items():
        if file_hash(execution / name) != expected_hash:
            raise RuntimeError('execution input changed during restore')
    report['binary_sha256'] = {name: execution_hashes[name] for name in ('arangod', 'arangorestore')}
    report['runtime_data_sha256'] = {name: execution_hashes[name] for name in execution_hashes if name not in {'probe.py', 'helper.py', 'arangod', 'arangorestore'}}
    report['probe_sha256'] = execution_hashes['probe.py']
    report['isolation_helper_sha256'] = execution_hashes['helper.py']
    report['execution_provenance'] = 'Pre-hashed private copies; probe re-executed and helper/binaries loaded from copies; hashes rechecked after validation'
    report['python_optimization'] = sys.flags.optimize
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
