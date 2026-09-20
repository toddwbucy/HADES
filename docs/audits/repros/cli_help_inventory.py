"""Walk a specified isolated HADES binary using only --help invocations.

Run from the repository root. The caller builds the binary and binds its source
revision separately. This script never executes an operational command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
parser.add_argument('--source-revision', required=True)
args = parser.parse_args()
binary = Path(args.binary).resolve(strict=True)
rows = []
seen = set()
pending = [()]
with tempfile.TemporaryDirectory(prefix='hades-help-only-') as home:
    env = {'PATH':'/usr/bin:/bin', 'HOME':home, 'LANG':'C.UTF-8', 'NO_COLOR':'1'}
    while pending:
        path = pending.pop(0)
        assert path not in seen
        seen.add(path)
        assert len(seen) <= 256 and len(path) <= 8
        # Each path comes only from a command declaration in a prior help response.
        assert all(re.fullmatch(r'[a-z][a-z0-9-]*', part) for part in path)
        result = subprocess.run([str(binary), *path, '--help'], cwd=home,
                                env=env, capture_output=True, timeout=5)
        assert result.returncode == 0, (path, result.stderr[:200])
        assert len(result.stdout) <= 65536 and not result.stderr, path
        text = result.stdout.decode('utf-8')
        commands = []
        if '\nCommands:\n' in text:
            section = text.split('\nCommands:\n', 1)[1].split('\n\n',1)[0]
            for line in section.splitlines():
                match = re.match(r'  ([a-z][a-z0-9-]*)\s', line)
                if match and match[1] != 'help':
                    commands.append(match[1])
        pending.extend((*path, command) for command in commands)
        rows.append({'path':list(path),'children':commands,'help':text})
print(json.dumps({'source_revision':args.source_revision,
    'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
    'scope':'Declared visible CLI help tree only; no operational commands, no config/database/service requests. Hidden commands/aliases and execution behavior are outside enumeration.',
    'nodes':len(rows),'leaf_commands':sum(not row['children'] for row in rows),
    'commands':rows},indent=2))
