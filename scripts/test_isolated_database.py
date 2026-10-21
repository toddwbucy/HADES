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
DATABASE_CONTRACTS = ["arango_crud", "arango_index", "arango_query", "arango_transport", "arango_cache",
                      "graph_loader", "graph_contract", "kg_authored", "graphable_import", "reading", "reading_chat", "cursor_lifecycle", "transaction", "structural_export_db", "structural_vectors_db", "document_replacement_db", "embedding_profile_store", "profiled_documents", "workspace_setup"]
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
            child.wait(timeout=1)
            return
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            child.poll()  # reap the direct child even while descendants exit
            if not group_running(child.pid):
                child.wait(timeout=1)
                return
            time.sleep(0.05)
    raise RuntimeError(f"private process group {child.pid} did not stop")


def run_reading_e2e(argv, env, timeout, results):
    """Record the browser flow's outcome and reap its private process group."""
    # CPU Docling/Chromium need more virtual address space than bounded_process
    # permits. Keep low scheduling priority and group ownership without that cap.
    child = subprocess.Popen(argv, cwd=REPO, env=env, start_new_session=True,
                             preexec_fn=lower_priority)
    try:
        code = child.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        results["reading-e2e"] = {"passed": False, "command": argv, "error": str(error)}
        raise
    finally:
        stop_group(child)
    results["reading-e2e"] = {"passed": code == 0, "exit_code": code, "command": argv}
    if code:
        raise RuntimeError("guided reading end-to-end test failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    server = parser.add_mutually_exclusive_group(required=True)
    server.add_argument("--arangod", type=Path, help="existing binary; never installs a server")
    server.add_argument("--docker", action="store_true", help="run the pinned CI image with no network")
    parser.add_argument("--command-timeout", type=int, default=900)
    benchmark = parser.add_mutually_exclusive_group()
    benchmark.add_argument("--benchmark", action="store_true", help="run the opt-in synthetic retrieval benchmark")
    benchmark.add_argument("--handler-benchmark", action="store_true", help="measure full search handlers on disposable fixtures")
    benchmark.add_argument("--training-alignment-python", type=Path,
                           help="run the cross-language training/export contract with this CPU interpreter")
    benchmark.add_argument("--cli-lifecycle", action="store_true",
                           help="run the CLI lifecycle contracts on the private server")
    benchmark.add_argument("--mcp-authored", action="store_true",
                           help="run the authored-graph MCP roundtrip on the private server")
    benchmark.add_argument("--contract", choices=DATABASE_CONTRACTS,
                           help="run one database contract on the same strict private server")
    benchmark.add_argument("--verified-jina", type=Path,
                           help="run the opt-in real provider contract using this qualification context")
    parser.add_argument("--verified-jina-report", type=Path,
                        help="new report file for the real provider contract")
    benchmark.add_argument("--graphable-import", action="store_true",
                           help="run the mechanical importer CLI on the private server")
    parser.add_argument("--graphable-source", type=Path,
                        help="reviewed graphable Markdown for the opt-in real-document test")
    parser.add_argument("--graphable-archive", type=Path,
                        help="original source archive for the real-document test")
    benchmark.add_argument("--reading-e2e-python", type=Path,
                           help="run the real guided-reading browser flow with this Python interpreter")
    benchmark.add_argument("--reading-chat-browser", action="store_true",
                           help="run browser/Codex reading contracts on a disposable database")
    parser.add_argument("--reading-chat-auth", type=Path,
                        help="opt into real Codex inference using a temporary private copy of this auth.json")
    parser.add_argument("--reading-chat-codex", type=Path, default=Path("/usr/bin/codex"))
    parser.add_argument("--reading-node", type=Path)
    parser.add_argument("--reading-playwright", type=Path)
    parser.add_argument("--reading-chromium", type=Path)
    parser.add_argument("--reading-model-cache", type=Path)
    parser.add_argument("--benchmark-rows", type=int, default=1024)
    parser.add_argument("--benchmark-dimension", type=int, default=64)
    parser.add_argument("--benchmark-trials", type=int, default=16)
    args = parser.parse_args()
    if args.command_timeout <= 0:
        parser.error("command timeout must be positive")
    if args.arangod and not args.arangod.is_file():
        parser.error("arangod must name an existing binary")
    if args.training_alignment_python and not args.training_alignment_python.is_file():
        parser.error("training alignment requires an existing Python interpreter")
    if bool(args.verified_jina) != bool(args.verified_jina_report):
        parser.error("--verified-jina and --verified-jina-report are required together")
    if args.verified_jina and not args.verified_jina.is_file():
        parser.error("verified Jina context must name an existing file")
    if args.verified_jina_report and (
        not args.verified_jina_report.parent.is_dir()
        or args.verified_jina_report.exists() or args.verified_jina_report.is_symlink()
    ):
        parser.error("verified Jina report must be a new file in an existing directory")

    if (args.graphable_source or args.graphable_archive) and not args.graphable_import:
        parser.error("graphable source/archive require --graphable-import")
    if args.graphable_archive and not args.graphable_source:
        parser.error("graphable archive requires --graphable-source")
    for path in (args.graphable_source, args.graphable_archive):
        if path and not path.is_file():
            parser.error("graphable input must name an existing file")

    if args.reading_chat_browser and not (args.reading_playwright and args.reading_chromium):
        parser.error("reading chat browser requires --reading-playwright and --reading-chromium")
    if args.reading_chat_auth and not args.reading_chat_browser:
        parser.error("reading chat authentication requires --reading-chat-browser")
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

    def command(name, cargo_args, command_env=None, expect_missing_socket=False, ignored=False):
        argv = ["cargo", "test", "--locked", "--offline", *cargo_args,
                "--", "--test-threads=1", "--nocapture"]
        if ignored:
            argv.append("--ignored")
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
             "--arangosearch.commit-threads", "1", "--arangosearch.consolidation-threads", "1",
             "--arangosearch.execution-threads-limit", "1"]
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
            if args.verified_jina:
                env["HADES_VERIFIED_JINA_CONTEXT"] = str(args.verified_jina.absolute())
                env["HADES_VERIFIED_JINA_REPORT"] = str(args.verified_jina_report.absolute())
                command("verified-jina", ["-p", "hades-core", "--test", "verified_jina_documents"],
                        ignored=True)
                return
            if args.graphable_import:
                if args.graphable_source:
                    env["HADES_GRAPHABLE_SOURCE"] = str(args.graphable_source.absolute())
                if args.graphable_archive:
                    env["HADES_GRAPHABLE_ARCHIVE"] = str(args.graphable_archive.absolute())
                env["HADES_GRAPHABLE_REPORT"] = str(root / "graphable-report.json")
                command("graphable-cli", ["-p", "hades-cli", "--test", "graphable_import"])
                return
            if args.cli_lifecycle:
                command("cli-lifecycle", ["-p", "hades-cli", "--test", "codebase_lifecycle"])
                return
            if args.reading_chat_browser:
                env.update(HADES_CHAT_TEST_ROOT=str(root),
                           PLAYWRIGHT_MODULE=str(args.reading_playwright.absolute()),
                           CHROMIUM_PATH=str(args.reading_chromium.absolute()),
                           HADES_CHAT_CODEX=str(args.reading_chat_codex.absolute()))
                if args.reading_chat_auth:
                    env["HADES_CHAT_TEST_AUTH"] = str(args.reading_chat_auth.absolute())
                run_reading_e2e(["cargo", "test", "--locked", "--offline", "-p", "hades-core",
                                 "--test", "reading_chat", "reading_chat_browser", "--", "--ignored", "--nocapture"],
                                env, args.command_timeout, results)
                return
            if args.reading_e2e_python:
                argv = [str(args.reading_e2e_python), str(REPO / "scripts/test_guided_reading.py"),
                        "--root", str(root), "--node", str(args.reading_node),
                        "--playwright", str(args.reading_playwright), "--chromium", str(args.reading_chromium),
                        "--model-cache", str(args.reading_model_cache)]
                run_reading_e2e(argv, env, args.command_timeout, results)
                return
            if args.mcp_authored:
                command("mcp-authored", ["-p", "hades-cli", "--test", "codebase_lifecycle",
                                         "mcp_authored_graph_tool_roundtrip"])
                return
            if args.training_alignment_python:
                # Preserve a virtualenv interpreter symlink: resolving it can
                # silently select the base interpreter without CPU dependencies.
                env["HADES_ALIGNMENT_PYTHON"] = str(args.training_alignment_python.absolute())
                command("training-alignment", ["-p", "hades-cli", "--test", "training_alignment"], ignored=True)
                return
            if args.benchmark or args.handler_benchmark:
                if not args.docker:
                    env["HADES_BENCH_SERVER_PID"] = str(child.pid)
                env.update({"HADES_BENCH_ROWS": str(args.benchmark_rows),
                            "HADES_BENCH_DIMENSION": str(args.benchmark_dimension),
                            "HADES_BENCH_TRIALS": str(args.benchmark_trials)})
                target = "search_handler_benchmark" if args.handler_benchmark else "retrieval_benchmark"
                command(target, ["-p", "hades-core", "--test", target], ignored=True)
                return
            if args.contract:
                if args.contract == "graphable_import":
                    command("graphable-unit", ["-p", "hades-core", "--lib", "graphable"])
                if args.contract == "reading_chat":
                    command("reading-chat-monitor", ["-p", "hades-core", "--lib", "reading::chat::tests::monitor_"])
                command(args.contract, ["-p", "hades-core", "--test", args.contract])
                return
            missing = dict(env)
            missing.pop("ARANGO_SOCKET")
            command("strict-prerequisite", ["-p", "hades-core", "--test", "arango_crud", "test_count_collection"],
                    missing, expect_missing_socket=True)
            command("graphable-unit", ["-p", "hades-core", "--lib", "graphable"])
            command("reading-chat-monitor", ["-p", "hades-core", "--lib", "reading::chat::tests::monitor_"])
            targets = DATABASE_CONTRACTS
            command("database-contracts", ["-p", "hades-core", *[arg for target in targets for arg in ("--test", target)]])
            command("codebase", ["-p", "hades-cli", "--bin", "hades", "commands::codebase_"])
            command("cli-lifecycle", ["-p", "hades-cli", "--test", "file_identity", "--test", "codebase_lifecycle", "--test", "graphable_import"])
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
