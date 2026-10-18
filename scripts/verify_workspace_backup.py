#!/usr/bin/env python3
"""Restore an explicit workspace backup into a new private Unix-socket server.

Consumes manifest.json from the documented ThinkPad backup procedure. Never
accepts a target endpoint or inherits HADES credentials. Retains private artifacts
under /tmp; prints counts only. Does not launch Codex or contact ML services.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time

from test_isolated_database import bounded_process, stop_group


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_files(backup, manifest):
    """Reject missing, changed, escaping or symlinked backup inputs."""
    result = {}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe backup path")
        source = backup / relative
        if any(p.is_symlink() for p in [source, *source.parents] if p != backup.parent):
            raise ValueError("symlinked backup input")
        if not source.is_file() or digest(source) != expected:
            raise ValueError("backup file missing or checksum mismatch")
        result[name] = source
    actual = {str(p.relative_to(backup)) for p in backup.rglob("*") if p.is_file()}
    if actual != set(result) | {"manifest.json"}:
        raise ValueError("unmanifested backup files")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--arangod", type=Path, required=True)
    parser.add_argument("--arangorestore", type=Path, required=True)
    parser.add_argument("--hades", type=Path, required=True)
    args = parser.parse_args()
    for executable in (args.arangod, args.arangorestore, args.hades):
        if not executable.is_file():
            parser.error("each executable must exist")
    os.umask(0o077)
    backup = args.backup.resolve()
    manifest = json.loads((backup / "manifest.json").read_text())
    files = verified_files(backup, manifest)
    root = Path(tempfile.mkdtemp(prefix="hades-recovery-", dir="/tmp"))
    print(f"Private recovery artifacts: {root}", flush=True)
    # Stage checked inputs so the restore never reads unverified extra dump files.
    for name, source in files.items():
        target = root / "restored-files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != manifest["files"][name]:
            raise RuntimeError("backup changed while staging")
    sock = root / "arango.sock"
    database = "hades_recovery_fixture"
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "LANG": "C.UTF-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}

    def api(method, path, body=None):
        conn = http.client.HTTPConnection("localhost", timeout=30)
        conn.sock = socket.socket(socket.AF_UNIX)
        conn.sock.settimeout(30)
        conn.sock.connect(str(sock))
        try:
            conn.request(method, path, json.dumps(body) if body is not None else None,
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status >= 400:
                raise RuntimeError(f"private database HTTP {response.status}")
            return json.load(response)
        finally:
            conn.close()

    def db(method, path, body=None):
        return api(method, "/_db/" + database + path, body)

    def rows(collection):
        page = db("POST", "/_api/cursor", {"query": 'FOR d IN @@c SORT d._key RETURN UNSET(d, "_rev")',
                  "bindVars": {"@c": collection}, "batchSize": 100, "options": {"maxRuntime": 30}})
        result = page["result"]
        while page.get("hasMore"):
            page = db("POST", "/_api/cursor/" + page["id"])
            result.extend(page["result"])
        return result

    def interrupted(signum, _frame):
        raise RuntimeError(f"recovery interrupted: {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    flags = ["--configuration", "none", "--database.directory", str(root / "data"),
             "--server.endpoint", "unix://" + str(sock), "--server.authentication", "false",
             "--javascript.enabled", "false", "--foxx.queues", "false", "--vector-index", "true",
             "--server.statistics", "false", "--server.minimal-threads", "4",
             "--server.maximal-threads", "8", "--server.io-threads", "1",
             "--rocksdb.block-cache-size", "67108864", "--rocksdb.total-write-buffer-size", "67108864",
             "--rocksdb.write-buffer-size", "16777216", "--rocksdb.max-background-jobs", "2",
             "--arangosearch.commit-threads", "1", "--arangosearch.consolidation-threads", "1",
             "--arangosearch.execution-threads-limit", "1", "--log.output", "-"]
    with (root / "server.log").open("w") as log:
        child = subprocess.Popen([str(args.arangod.resolve()), *flags], env=env, cwd=root,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, preexec_fn=bounded_process)
        try:
            deadline = time.monotonic() + 60
            while True:
                if child.poll() is not None:
                    raise RuntimeError("private server exited; inspect server.log")
                try:
                    version = api("GET", "/_api/version")["version"]
                    break
                except (OSError, http.client.HTTPException):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("private server startup timed out")
                    time.sleep(0.1)
            api("POST", "/_api/database", {"name": database})
            started = time.monotonic()
            with (root / "restore.log").open("w") as output:
                subprocess.run([str(args.arangorestore.resolve()), "--configuration", "none",
                                "--server.endpoint", "unix://" + str(sock), "--server.authentication", "false",
                                "--server.database", database, "--threads", "1",
                                "--include-system-collections", "true", "--input-directory",
                                str(root / "restored-files/dump")], env=env, cwd=root, stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=180)
            restore_seconds = time.monotonic() - started
            restored = {}
            index_keys = ("type", "fields", "unique", "sparse", "name", "storedValues", "params", "deduplicate", "estimates")
            for item in db("GET", "/_api/collection")["result"]:
                name = item["name"]
                if name.startswith("_") and name != "_graphs":
                    continue
                records = rows(name)
                properties = db("GET", "/_api/collection/" + name + "/properties")
                indexes = db("GET", "/_api/index?collection=" + name)["indexes"]
                restored[name] = {"count": len(records), "sha256": hashlib.sha256(canonical(records)).hexdigest(),
                                  "type": item["type"], "schema": properties.get("schema"),
                                  "computedValues": properties.get("computedValues"),
                                  "indexes": sorted([{k: x[k] for k in index_keys if k in x} for x in indexes], key=canonical)}
            if restored != manifest["collections"]:
                (root / "actual-collections.json").write_text(json.dumps(restored, indent=2))
                raise RuntimeError("restored documents, schema, graphs or indexes differ")
            sources = rows("hades_reading_sources")
            for source in sources:
                content = base64.b64decode(source["content_base64"], validate=True)
                if hashlib.sha256(content).hexdigest() != source["_key"]:
                    raise RuntimeError("restored source bytes do not match their identity")
            config = root / "hades.json"
            config.write_text(json.dumps({"database": {"name": database, "username": "root",
                "sockets": {"readonly": str(sock), "readwrite": str(sock)}}, "reading_chat": {"enabled": False},
                "embedding": {"service": {"socket": str(root / "absent-embedder.sock")}},
                "extraction": {"service": {"socket": str(root / "absent-extractor.sock")}}}))
            cli_env = dict(env, HADES_CONFIG=str(config), ARANGO_USERNAME="root", ARANGO_PASSWORD="fixture-only",
                           ARANGO_RO_SOCKET=str(sock), ARANGO_RW_SOCKET=str(sock))
            sessions = rows("hades_reading_sessions")
            for session in sessions:
                for view in ("state", "export"):
                    run = subprocess.run([str(args.hades.resolve()), "--database", database, "--format", "json",
                                          "reading", "read", "--session", session["_key"], "--view", view],
                                         env=cli_env, cwd=root, capture_output=True, check=True, timeout=30)
                    result = json.loads(run.stdout)
                    if not result["success"]:
                        raise RuntimeError("application reading recovery failed")
            report = {"passed": True, "server_version": version, "collections_compared": len(restored),
                      "documents_compared": sum(c["count"] for c in restored.values()),
                      "reading_sources_verified": len(sources), "reading_sessions_opened_and_exported": len(sessions),
                      "conversation_bindings": restored["hades_reading_conversations"]["count"],
                      "saved_exchanges": restored["hades_reading_exchanges"]["count"],
                      "provider_transcripts_copied": len(manifest["provider_transcripts"]),
                      "empty_bindings_without_provider_transcript": len(manifest.get("empty_bindings_without_provider_transcript", [])),
                      "backup_files_verified": len(files), "restore_seconds": restore_seconds,
                      "limits": ["one local backup, no scheduled or off-device protection",
                                 "provider transcript bytes checked; no provider resume or authentication tested",
                                 "model weights and external ingestion originals excluded"]}
            (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            stop_group(child)


if __name__ == "__main__":
    main()
