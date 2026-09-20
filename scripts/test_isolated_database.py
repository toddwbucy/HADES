#!/usr/bin/env python3
"""Run strict contracts against a private server; never discover a live endpoint."""
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import resource
import socket
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
# Official library/arangodb:3.12 manifest, resolved 2026-09-20.
IMAGE = "arangodb@sha256:3ce7aa54ac9b0942a2b201cd47a397cb89c3d5ff5a087d701f3c785109d0dfb7"


def lower_priority():
    os.nice(10)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def bounded_process():
    lower_priority()
    limit = 8 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def group_running(pgid):
    # Linux-only runner. Zombies have exited and cannot consume resources;
    # an orphan may remain a zombie until the host's init reaps it.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def stop_group(child):
    # The session leader may already have exited while descendants are alive.
    # Only signal the private group created with start_new_session=True.
    for signum, budget in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            child.poll()
            return
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            child.poll()  # reap the direct child even while descendants exit
            if not group_running(child.pid):
                return
            time.sleep(0.05)
    raise RuntimeError(f"private process group {child.pid} did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    server = parser.add_mutually_exclusive_group(required=True)
    server.add_argument("--arangod", type=Path, help="existing binary; never installs a server")
    server.add_argument("--docker", action="store_true", help="run the pinned CI image with no network")
    parser.add_argument("--command-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.command_timeout <= 0:
        parser.error("command timeout must be positive")
    if args.arangod and not args.arangod.is_file():
        parser.error("arangod must name an existing binary")

    root = Path(tempfile.mkdtemp(prefix="hades-tests-", dir="/tmp"))
    print(f"Isolated test artifacts: {root}", flush=True)
    sock = root / "arango.sock"
    # Pass only build/runtime necessities, never inherited production HADES or
    # ArangoDB configuration, socket paths, application tokens, or credentials.
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ",
               "RUSTUP_HOME", "CARGO_HOME", "CARGO_TARGET_DIR", "LIBCLANG_PATH"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "CARGO_BUILD_JOBS": "1", "CARGO_INCREMENTAL": "0",
                "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_PROFILE_TEST_DEBUG": "0"})
    config = root / "hades.yaml"
    config.write_text(json.dumps({"database": {"name": "unused", "username": "root",
        "sockets": {"readonly": str(sock), "readwrite": str(sock)}},
        "embedding": {"service": {"socket": str(root / "absent-embedder.sock")}},
        "extraction": {"service": {"socket": str(root / "absent-extractor.sock")}}}))
    env.update({"ARANGO_SOCKET": str(sock), "ARANGO_RO_SOCKET": str(sock),
                "ARANGO_RW_SOCKET": str(sock), "ARANGO_PASSWORD": "isolated-fixture-only",
                "ARANGO_TESTS": "1", "HADES_TEST_USER": "root", "HADES_CONFIG": str(config),
                "HADES_EMBEDDER_SOCKET": str(root / "absent-embedder.sock"),
                "HADES_EXTRACTOR_SOCKET": str(root / "absent-extractor.sock")})

    def api(path):
        connection = http.client.HTTPConnection("localhost", timeout=3)
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.settimeout(3)
        connection.sock.connect(str(sock))
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            data = json.loads(response.read())
            if response.status >= 400:
                raise RuntimeError(f"private server HTTP {response.status}")
            return data
        finally:
            connection.close()

    results = {}

    def command(name, cargo_args, command_env=None, expect_missing_socket=False):
        argv = ["cargo", "test", "--locked", "--offline", *cargo_args,
                "--", "--test-threads=1", "--nocapture"]
        log_path = root / f"{name}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(argv, cwd=REPO, env=command_env or env,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, preexec_fn=bounded_process)
            try:
                code = child.wait(timeout=args.command_timeout)
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                results[name] = {"passed": False, "command": argv, "error": str(error)}
                raise
            finally:
                stop_group(child)
        if expect_missing_socket:
            passed = code != 0 and "ARANGO_TESTS requires ARANGO_SOCKET" in log_path.read_text()
        else:
            passed = code == 0
        results[name] = {"passed": passed, "exit_code": code, "command": argv}
        print(f"{name}: {'PASS' if passed else 'FAIL'}; {log_path}", flush=True)
        if not passed:
            raise RuntimeError(f"{name} failed; inspect {log_path}")

    def interrupted(signum, _frame):
        raise RuntimeError(f"test runner interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    directory = "/fixture" if args.docker else str(root)
    flags = ["--configuration", "none", "--log.output", "-", "--log.force-direct", "true",
             "--database.directory", directory + "/data",
             "--server.endpoint", "unix://" + directory + "/arango.sock",
             "--server.authentication", "false", "--javascript.enabled", "false",
             "--experimental-vector-index", "true", "--foxx.queues", "false",
             "--server.statistics", "false", "--server.minimal-threads", "4",
             "--server.maximal-threads", "8", "--server.io-threads", "1",
             "--rocksdb.block-cache-size", "67108864",
             "--rocksdb.total-write-buffer-size", "67108864",
             "--rocksdb.write-buffer-size", "16777216", "--rocksdb.max-background-jobs", "2",
             "--arangosearch.threads", "1", "--arangosearch.threads-limit", "1"]
    container = f"hades-tests-{os.getpid()}-{root.name}"
    if args.docker:
        argv = ["docker", "run", "--rm", "--name", container, "--network", "none",
                "--cpus", "1", "--memory", "1g", "--memory-swap", "1g", "--pids-limit", "128",
                "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
                "--mount", f"type=bind,src={root},dst=/fixture", "--env", "OMP_NUM_THREADS=1",
                "--entrypoint", "arangod", IMAGE, *flags]
    else:
        argv = [str(args.arangod.resolve()), *flags]
    with (root / "arangod.log").open("w") as log:
        child = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 preexec_fn=lower_priority if args.docker else bounded_process)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"private server exited {child.returncode}; see {root / 'arangod.log'}")
                try:
                    version = api("/_api/version")["version"]
                    break
                except (OSError, http.client.HTTPException):
                    time.sleep(0.1)
            else:
                raise RuntimeError("private server startup timed out")
            print(f"Private ArangoDB {version} ready; no TCP listener", flush=True)
            missing = dict(env)
            missing.pop("ARANGO_SOCKET")
            command("strict-prerequisite", ["-p", "hades-core", "--test", "arango_crud", "test_count_collection"],
                    missing, expect_missing_socket=True)
            targets = ["arango_crud", "arango_index", "arango_query", "arango_transport", "arango_cache",
                       "graph_loader", "graph_contract", "cursor_lifecycle"]
            command("database-contracts", ["-p", "hades-core", *[arg for target in targets for arg in ("--test", target)]])
            command("codebase", ["-p", "hades-cli", "--bin", "hades", "commands::codebase_"])
            command("cli-lifecycle", ["-p", "hades-cli", "--test", "file_identity", "--test", "codebase_lifecycle"])
        finally:
            stop_group(child)
            if args.docker:
                removed = subprocess.run(["docker", "rm", "--force", container], env=env,
                                         capture_output=True, text=True, timeout=20, check=False)
                if removed.returncode and "No such container" not in removed.stderr:
                    raise RuntimeError("could not remove private test container: " + removed.stderr.strip())
            (root / "results.json").write_text(json.dumps(results, indent=2))
            print(f"Private server stopped; exit={child.returncode}", flush=True)


if __name__ == "__main__":
    main()
