#!/usr/bin/env python3
"""Freeze the code-search evaluation corpus from immutable Git objects."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
import subprocess

from prepare_retrieval_passages import MAX_CORPUS_BYTES, MAX_DOCUMENT_BYTES, prepare, write_private

ROOTS = {'crates', 'services', 'proto', 'config', 'deploy', 'scripts', 'web', '.github'}
SUFFIXES = {'.rs', '.py', '.proto', '.toml', '.yaml', '.yml', '.sh', '.service',
            '.socket', '.target', '.html', '.css', '.js'}
ROOT_FILES = {'Cargo.toml', 'rust-toolchain.toml'}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout


def selected(name):
    path = PurePosixPath(name)
    return name in ROOT_FILES or (path.parts[0] in ROOTS and path.suffix in SUFFIXES)


def freeze(repo, revision, output, queries, max_bytes=4096):
    """Create a new private snapshot; include all matching files, never working edits."""
    # Resolve first, so arbitrary revision strings never become options downstream.
    commit = git(repo, 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()
    entries, total = [], 0
    for entry in git(repo, 'ls-tree', '-r', '-z', '-l', '--full-tree', commit).split(b'\0'):
        if not entry:
            continue
        header, raw_name = entry.split(b'\t', 1)
        mode, kind, oid, raw_size = header.split()
        name = raw_name.decode('utf-8')
        if not selected(name):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('invalid selected Git path')
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise ValueError('selected input is not a regular Git blob')
        size = int(raw_size)
        total += size
        if size > MAX_DOCUMENT_BYTES or total > MAX_CORPUS_BYTES:
            raise ValueError('code corpus exceeds preparation bounds')
        entries.append((name, oid.decode('ascii'), size))
    if not entries:
        raise ValueError('no matching source files')
    if not isinstance(queries, list) or not queries:
        raise ValueError('queries must be a nonempty list')
    if any(not isinstance(q, dict) or set(q) != {'id', 'text', 'relevance'}
           or not isinstance(q['id'], str) or not q['id']
           or not isinstance(q['text'], str) or not q['text'].strip()
           or q['relevance'] != {} for q in queries):
        raise ValueError('preparation requires unlabeled id/text/relevance queries')
    if len({q['id'] for q in queries}) != len(queries):
        raise ValueError('duplicate query ID')
    output = Path(output)
    output.mkdir(mode=0o700)  # Refuse to replace an earlier snapshot.
    try:
        documents = output / 'documents'
        documents.mkdir(mode=0o700)
        manifest = {'version': 1, 'commit': commit, 'policy': {
            'roots': sorted(ROOTS), 'suffixes': sorted(SUFFIXES), 'root_files': sorted(ROOT_FILES),
            'source': 'regular tracked Git blobs; no working-tree files or submodules'}, 'documents': []}
        for name, oid, size in entries:
            raw = git(repo, 'cat-file', 'blob', oid)
            if len(raw) != size:
                raise ValueError('Git blob size mismatch')
            raw.decode('utf-8')
            path = documents / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open('xb') as stream:
                os.chmod(path, 0o600)
                stream.write(raw)
            manifest['documents'].append({'path': name, 'git_blob': oid, 'bytes': size,
                                          'sha256': hashlib.sha256(raw).hexdigest()})
        write_private(output / 'manifest.json', manifest)
        artifact = prepare(output, max_bytes)
        artifact.update(workload='code_search', embedding_profile='code_search',
                        source_commit=commit, queries=queries,
                        snapshot_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_private(output / 'code-candidates-v1.json', artifact)
        return {'commit': commit, 'files': len(entries), 'bytes': total,
                'passages': len(artifact['documents']), 'queries': len(queries),
                'dataset_sha256': hashlib.sha256((output / 'code-candidates-v1.json').read_bytes()).hexdigest()}
    except BaseException:
        # mkdir above succeeded: this invocation owns the incomplete directory.
        # Existing outputs fail before this block and must never be removed.
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.revision, args.output,
                            json.loads(args.queries.read_text()), args.max_bytes), indent=2))


if __name__ == '__main__':
    main()
