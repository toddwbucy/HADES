#!/usr/bin/env python3
"""Verify synthetic ArangoDB ACLs on an owned Unix-only server, never a live URL."""
import argparse
import base64
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time

import test_isolated_database as isolation


def b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args()
    binary = args.bin_dir.resolve() / 'arangod'
    if not binary.is_file():
        parser.error('existing arangod binary required')
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix='hades-acl-', dir='/tmp'))
    print(root, flush=True)
    isolation.lower_priority()
    endpoint = root / 'arango.sock'
    secret = secrets.token_hex(32)
    secret_file = root / 'jwt.key'
    secret_file.write_text(secret)
    secret_file.chmod(0o400)
    now = int(time.time())
    header = b64(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode())
    claims = b64(json.dumps({'iss': 'arangodb', 'preferred_username': 'root', 'iat': now, 'exp': now + 300}).encode())
    signing = header + '.' + claims
    admin = 'Bearer ' + signing + '.' + b64(hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest())
    password = secrets.token_urlsafe(24)
    credentials = {name: 'Basic ' + base64.b64encode((name + ':' + password).encode()).decode()
                   for name in ('fixture_reader', 'fixture_writer')}
    cases = []

    def request(method, path, body=None, authorization=admin, expected=200):
        c = http.client.HTTPConnection('localhost', timeout=5)
        c.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.sock.settimeout(5)
        c.sock.connect(str(endpoint))
        try:
            headers = {'Content-Type': 'application/json'}
            if authorization is not None:
                headers['Authorization'] = authorization
            c.request(method, path, json.dumps(body) if body is not None else None, headers)
            response = c.getresponse()
            raw = response.read(65537)
            if len(raw) > 65536:
                raise RuntimeError('private ACL response exceeded limit')
            if response.status != expected:
                raise RuntimeError(f'private ACL request expected {expected}, received {response.status}')
            return json.loads(raw) if raw else None
        finally:
            c.close()

    def check(name, method, path, body=None, authorization=None, expected=200):
        request(method, path, body, authorization, expected)
        cases.append({'case': name, 'status': expected})

    def interrupted(signum, frame):
        raise RuntimeError('private ACL probe interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = {'PATH': os.environ['PATH'], 'HOME': str(root), 'LANG': 'C', 'OMP_NUM_THREADS': '1',
           'CUDA_VISIBLE_DEVICES': '', 'ICU_DATA': str(binary.parent), 'ICU_DATA_LEGACY': str(binary.parent),
           'TZ_DATA': str(binary.parent / 'tzdata')}
    flags = ['--configuration', 'none', '--database.directory', str(root / 'data'),
             '--server.endpoint', 'unix://' + str(endpoint), '--server.authentication', 'true',
             '--server.authentication-unix-sockets', 'true', '--server.jwt-secret-keyfile', str(secret_file),
             '--javascript.enabled', 'false', '--foxx.queues', 'false', '--server.statistics', 'false',
             '--server.minimal-threads', '4', '--server.maximal-threads', '8', '--server.io-threads', '1',
             '--rocksdb.block-cache-size', '67108864', '--rocksdb.total-write-buffer-size', '67108864',
             '--rocksdb.write-buffer-size', '16777216', '--rocksdb.max-background-jobs', '2',
             '--arangosearch.threads', '1', '--arangosearch.threads-limit', '1', '--log.output', '-']
    child = None
    try:
        with (root / 'server.log').open('w') as output:
            child = subprocess.Popen([str(binary), *flags], env=env, cwd=root, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     preexec_fn=isolation.bounded_process)
        deadline = time.monotonic() + 60
        while True:
            if child.poll() is not None:
                raise RuntimeError('private ACL server exited')
            try:
                version = request('GET', '/_api/version')['version']
                break
            except (OSError, http.client.HTTPException):
                if time.monotonic() > deadline:
                    raise RuntimeError('private ACL startup timeout')
                time.sleep(.1)
        for db in ('acl_allowed', 'acl_denied'):
            request('POST', '/_api/database', {'name': db}, expected=201)
            for collection in ('records', 'protected'):
                request('POST', f'/_db/{db}/_api/collection', {'name': collection}, expected=200)
                request('POST', f'/_db/{db}/_api/document/{collection}', {'_key': 'seed', 'value': 1}, expected=202)
        for user, level in [('fixture_reader', 'ro'), ('fixture_writer', 'rw')]:
            request('POST', '/_api/user', {'user': user, 'passwd': password, 'active': True}, expected=201)
            for db, grant in [('%2A', 'none'), ('_system', 'none'), ('acl_denied', 'none'), ('acl_allowed', level)]:
                request('PUT', f'/_api/user/{user}/database/{db}', {'grant': grant})
        request('PUT', '/_api/user/fixture_writer/database/acl_allowed/protected', {'grant': 'ro'})
        read = '/_db/acl_allowed/_api/document/records/seed'
        write = '/_db/acl_allowed/_api/document/records'
        check('missing_credentials_rejected', 'GET', read, expected=401)
        check('wrong_credentials_rejected', 'GET', read, authorization='Basic ' + base64.b64encode(b'fixture_reader:wrong').decode(), expected=401)
        check('reader_read_allowed', 'GET', read, authorization=credentials['fixture_reader'])
        check('reader_write_denied', 'POST', write, {'_key': 'reader_attempt'}, credentials['fixture_reader'], 403)
        check('writer_read_allowed', 'GET', read, authorization=credentials['fixture_writer'])
        check('writer_write_allowed', 'POST', write, {'_key': 'writer_success'}, credentials['fixture_writer'], 202)
        check('writer_collection_override_denied', 'POST', '/_db/acl_allowed/_api/document/protected', {'_key': 'writer_attempt'}, credentials['fixture_writer'], 403)
        for user in credentials:
            check(user + '_other_database_denied', 'GET', '/_db/acl_denied/_api/document/records/seed', authorization=credentials[user], expected=401)
            check(user + '_administrative_listing_denied', 'GET', '/_api/database', authorization=credentials[user], expected=401)
        for path in (write + '/reader_attempt', '/_db/acl_allowed/_api/document/protected/writer_attempt'):
            request('GET', path, expected=404)
        request('GET', write + '/writer_success')
        isolation.stop_group(child)
        child = None
        result = {'server_version': version, 'status': 'passed', 'cases': cases,
                  'denied_documents_absent': True, 'allowed_document_present': True,
                  'owned_process_stopped': True,
                  'limitations': ['Synthetic Unix-socket ACL matrix, not deployed credentials/grants.',
                                  'No production configuration changes; no TCP or HADES transport tested.']}
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if child is not None:
            isolation.stop_group(child)


if __name__ == '__main__':
    main()
