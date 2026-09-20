#!/usr/bin/env python3
"""Check the configured daemon database through its bounded Unix-socket protocol."""
import argparse
import json
import math
import socket
import struct
import time

MAX_RESPONSE = 64 * 1024


def check_once(path, database, deadline):
    """Require a complete healthy response before the shared monotonic deadline."""
    def budget(peer):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('readiness deadline exceeded')
        peer.settimeout(remaining)

    def receive(peer, count):
        result = bytearray()
        while len(result) < count:
            budget(peer)
            chunk = peer.recv(count - len(result))
            if not chunk:
                raise ValueError('truncated readiness response')
            result.extend(chunk)
        return result

    # The local protocol uses the daemon's configured database, not a request
    # override. Compare the returned identity with the caller's expectation.
    payload = json.dumps({'command': 'db.health', 'params': {'verbose': False},
                          'request_id': 'install-readiness', 'session': 'admin'}).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        budget(peer)
        peer.connect(path)
        budget(peer)
        peer.sendall(struct.pack('!I', len(payload)) + payload)
        size = struct.unpack('!I', receive(peer, 4))[0]
        if not 0 < size <= MAX_RESPONSE:
            raise ValueError('readiness response size outside limit')
        result = json.loads(receive(peer, size))
    if not isinstance(result, dict) or result.get('success') is not True:
        raise ValueError('daemon command failed')
    if result.get('request_id') != 'install-readiness' or result.get('error') is not None or result.get('error_code') is not None:
        raise ValueError('invalid readiness envelope')
    data = result.get('data')
    if not isinstance(data, dict) or data.get('database') != database:
        raise ValueError('daemon database differs from expected database')
    health = data.get('arangodb')
    if not isinstance(health, dict) or health.get('reader_ok') is not True or health.get('writer_ok') is not True or health.get('status') != 'healthy':
        raise ValueError('daemon database connections are degraded')


def wait_ready(path, database, timeout):
    """Retry startup failures within one deadline, never extend it per read."""
    if not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError('timeout must be finite and in (0, 120] seconds')
    deadline = time.monotonic() + timeout
    while True:
        try:
            check_once(path, database, min(deadline, time.monotonic() + 2))
            return
        except (OSError, ValueError, struct.error) as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'daemon readiness failed: {error}') from error
            time.sleep(min(.1, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--database', required=True, help='expected daemon database')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    try:
        wait_ready(args.socket, args.database, args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    print('Daemon socket and configured database connections are ready; ML readiness is not checked.')


if __name__ == '__main__':
    main()
