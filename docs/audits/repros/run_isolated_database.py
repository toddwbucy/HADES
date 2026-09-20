"""Audit a disposable ArangoDB instance; never contacts the live server.

Uses a fresh 0700 directory, a Unix-only listener, no inherited HADES settings,
one CPU at reduced priority, and small RocksDB caches. Terminates its own child
in finally. Artifacts stay in /tmp for review. Requires an existing arangod.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import sys
import time

from historical_source import source_root

# Reuse the maintained runner's tested address-space and process-group controls.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from test_isolated_database import bounded_process, stop_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arangod", required=True)
    parser.add_argument("--codebase-tests", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    repo = source_root(args.source_root)
    root = Path(tempfile.mkdtemp(prefix="hades-audit-db-"))
    print(f"Audit directory: {root}", flush=True)
    sock = root / "arango.sock"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ARANGO", "HADES", "GH_", "GITHUB_"))}
    env.update({"CUDA_VISIBLE_DEVICES": "", "CARGO_TARGET_DIR": "/tmp/hades-audit-target",
                "CARGO_BUILD_JOBS": "1", "CARGO_INCREMENTAL": "0",
                "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_PROFILE_TEST_DEBUG": "0"})
    config = root / "hades.yaml"
    config.write_text(json.dumps({"database": {"name": "audit", "username": "root",
        "sockets": {"readonly": str(sock), "readwrite": str(sock)}},
        "embedding": {"service": {"socket": "unix://" + str(root / "absent-embedder.sock")}},
        "extraction": {"service": {"socket": str(root / "absent-extractor.sock")}}}))
    env.update({"ARANGO_SOCKET": str(sock), "ARANGO_RO_SOCKET": str(sock),
                "ARANGO_RW_SOCKET": str(sock), "ARANGO_PASSWORD": "isolated-audit-only",
                "ARANGO_TESTS": "1", "HADES_TEST_USER": "root", "HADES_CONFIG": str(config),
                "HADES_EMBEDDER_SOCKET": "unix://" + str(root / "absent-embedder.sock"),
                "HADES_EXTRACTOR_SOCKET": str(root / "absent-extractor.sock")})

    def api(method, path, body=None):
        connection = http.client.HTTPConnection("localhost", timeout=5)
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.settimeout(5)
        connection.sock.connect(str(sock))
        try:
            connection.request(method, path, body=json.dumps(body) if body is not None else None,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read())
            if response.status >= 400:
                raise RuntimeError(f"Audit database HTTP {response.status}: {data}")
            return data
        finally:
            connection.close()

    def command(name, argv, timeout=600):
        with (root / f"{name}.log").open("w") as log:
            child_command = subprocess.Popen(argv, cwd=repo, env=env, stdout=log,
                stderr=subprocess.STDOUT, preexec_fn=bounded_process, start_new_session=True)
            try:
                code = child_command.wait(timeout=timeout)
            finally:
                stop_group(child_command)
        print(f"{name}: exit={code}; log={root / (name + '.log')}", flush=True)
        return code

    with (root / "arangod.log").open("w") as log:
        child = subprocess.Popen([args.arangod, "--configuration", "none",
            "--log.output", "-", "--log.force-direct", "true",
            "--database.directory", str(root / "data"),
            "--server.endpoint", "unix://" + str(sock),
            "--server.authentication", "false", "--javascript.enabled", "false",
            "--foxx.queues", "false", "--server.statistics", "false",
            "--server.minimal-threads", "4", "--server.maximal-threads", "8",
            "--server.io-threads", "1", "--rocksdb.block-cache-size", "67108864",
            "--rocksdb.total-write-buffer-size", "67108864",
            "--rocksdb.write-buffer-size", "16777216", "--rocksdb.max-background-jobs", "2",
            "--arangosearch.threads", "1", "--arangosearch.threads-limit", "1"],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=bounded_process, start_new_session=True)
        try:
            for _ in range(100):
                if child.poll() is not None:
                    raise RuntimeError(f"Isolated server exited {child.returncode}; see {root / 'arangod.log'}")
                try:
                    api("GET", "/_api/version")
                    break
                except (OSError, http.client.HTTPException):
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated server did not become ready")
            print(f"Isolated server ready, PID {child.pid}, Unix socket only", flush=True)
            if command("cache-tests", ["cargo", "test", "--locked", "--offline", "-p", "hades-core",
                    "--test", "arango_cache", "--", "--test-threads=1", "--nocapture"]):
                raise RuntimeError("Isolated cache tests failed; inspect their log")
            if command("cli-build", ["cargo", "build", "--locked", "--offline", "-p", "hades-cli"]):
                raise RuntimeError("Audit CLI build failed; inspect its log")
            api("POST", "/_api/database", {"name": "audit"})
            tree = root / "fixture"
            (tree / "a").mkdir(parents=True)
            (tree / "a_b.py").write_text("def first():\n    return 1\n")
            (tree / "a" / "b.py").write_text("def second():\n    return 2\n")
            if command("collision-ingest", ["/tmp/hades-audit-target/debug/hades", "--db", "audit",
                    "codebase", "ingest", str(tree)], timeout=60):
                raise RuntimeError("Collision fixture ingest failed; inspect its log")
            rows = api("POST", "/_db/audit/_api/cursor", {"query":
                "FOR f IN codebase_files RETURN {key:f._key,path:f.path,symbol_count:f.symbol_count}"})
            result = {"input_files": ["a_b.py", "a/b.py"], "stored_files": rows["result"]}
            (root / "collision-result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            if args.codebase_tests:
                env["HADES_TEST_DB"] = "audit"
                if command("codebase-tests", ["cargo", "test", "--locked", "--offline", "-p", "hades-cli",
                    "--bin", "hades", "commands::codebase_", "--", "--test-threads=1", "--nocapture"]):
                    raise RuntimeError("Isolated codebase tests failed; inspect their log")
        finally:
            stop_group(child)
            print(f"Isolated server stopped; exit={child.returncode}", flush=True)


if __name__ == "__main__":
    main()
