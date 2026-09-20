#!/usr/bin/env python3
"""Private synthetic slow-reader measurement; never invoke an installed HADES backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import tempfile
import threading
import time


def confined_child():
    os.nice(10)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def status(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, value, _ = line.split()
                fields[name[:-1]] = int(value)
        return fields
    except FileNotFoundError:
        return {}


def wait_until(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError("private fixture deadline exceeded")
        time.sleep(0.01)


def connect(address):
    client = socket.socket()
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    client.settimeout(3)
    try:
        client.connect(address)
    except BaseException:
        client.close()
        raise
    return client


def request(client, address, path):
    client.sendall(f"GET {path} HTTP/1.1\r\nHost: {address[0]}:{address[1]}\r\nConnection: close\r\n\r\n".encode())


def main():
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("private benchmark interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True, help="explicit isolated build of hades-viewer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-bytes", type=int, default=8 * 1024 * 1024 - 1024)
    args = parser.parse_args()
    if not 1 <= args.payload_bytes <= 8 * 1024 * 1024 - 1024:
        parser.error("payload must fit the bounded synthetic CLI response")
    viewer_bin = args.viewer.resolve(strict=True)
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    peak = {"viewer_rss_kib": 0, "viewer_hwm_kib": 0, "backend_sum_rss_kib": 0}
    clients = []
    child = None
    stop = threading.Event()
    sampler = None
    result = {"passed": False, "scenario": "16 stalled HTTP/1 readers with synthetic graph names",
              "payload_bytes_per_response": args.payload_bytes,
              "cargo_lock_sha256": hashlib.sha256((repo / "Cargo.lock").read_bytes()).hexdigest(),
              "sample_interval_ms": 10, "viewer_sha256": hashlib.sha256(viewer_bin.read_bytes()).hexdigest(),
              "viewer_cpu_affinity_count": 1, "viewer_address_space_limit_bytes": 2 * 1024**3,
              "excludes": ["kernel socket buffers", "production corpora", "GPU", "real HADES CLI/DB"]}
    result["source_sha256"] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo / "crates/hades-frontend/src").glob("*.rs"))
    }
    with tempfile.TemporaryDirectory(prefix="hades-viewer-memory-", dir="/tmp") as private:
        root = Path(private)
        backend = root / "backend"
        backend.write_text("""#!/usr/bin/python3
import json, os, pathlib
root = pathlib.Path(__file__).parent
(root / ('pid-' + str(os.getpid()))).touch()
print(json.dumps({'data': {'graphs': [{'name': 'X' * PAYLOAD_BYTES, 'edge_definitions': []}]}}))
""".replace("PAYLOAD_BYTES", str(args.payload_bytes)))
        result["synthetic_backend_sha256"] = hashlib.sha256(backend.read_bytes()).hexdigest()
        backend.chmod(0o700)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            address = reservation.getsockname()
        log = (root / "viewer.log").open("wb")
        try:
            child = subprocess.Popen([str(viewer_bin), "serve", "--db", "fixture", "--hades-bin", str(backend),
                                      "--bind", f"{address[0]}:{address[1]}"],
                                     env={"PATH": "/usr/bin:/bin", "TOKIO_WORKER_THREADS": "2"},
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log,
                                     start_new_session=True, preexec_fn=confined_child)

            def ready():
                if child.poll() is not None:
                    raise RuntimeError("private viewer exited before readiness")
                try:
                    with connect(address):
                        return True
                except ConnectionRefusedError:
                    return False
            wait_until(ready)
            # Allow the readiness connection's permit to return.
            time.sleep(0.05)
            result["baseline_viewer_rss_kib"] = status(child.pid).get("VmRSS", 0)

            def sample():
                while not stop.is_set():
                    current = status(child.pid)
                    peak["viewer_rss_kib"] = max(peak["viewer_rss_kib"], current.get("VmRSS", 0))
                    peak["viewer_hwm_kib"] = max(peak["viewer_hwm_kib"], current.get("VmHWM", 0))
                    backend_rss = sum(status(int(p.name[4:])).get("VmRSS", 0) for p in root.glob("pid-*"))
                    peak["backend_sum_rss_kib"] = max(peak["backend_sum_rss_kib"], backend_rss)
                    stop.wait(0.01)
            sampler = threading.Thread(target=sample)
            sampler.start()
            for count in range(1, 17):
                client = connect(address)
                clients.append(client)
                request(client, address, "/api/graphs")
                wait_until(lambda: len(list(root.glob("pid-*"))) == count)
                wait_until(lambda: not any(status(int(p.name[4:])) for p in root.glob("pid-*")))
            time.sleep(0.2)
            result["held_clients"] = len(clients)
            result["steady_viewer_rss_kib"] = status(child.pid).get("VmRSS", 0)
            with connect(address) as rejected:
                try:
                    request(rejected, address, "/api/databases")
                    refused = rejected.recv(1) == b""
                except (ConnectionResetError, BrokenPipeError):
                    refused = True
            if not refused:
                raise RuntimeError("seventeenth connection was not refused")
            result["seventeenth_connection_refused"] = True
            clients.pop().close()
            time.sleep(0.1)
            with connect(address) as admitted:
                request(admitted, address, "/api/databases")
                response = bytearray()
                while chunk := admitted.recv(4096):
                    response.extend(chunk)
                    if len(response) > 8192:
                        raise RuntimeError("unexpected oversized readiness response")
                if not response.startswith(b"HTTP/1.1 200"):
                    raise RuntimeError("released connection was not readmitted")
            result["released_connection_readmitted"] = True
            result["passed"] = True
        finally:
            for client in clients:
                client.close()
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                    result["passed"] = False
            result["viewer_exit_code"] = None if child is None else child.returncode
            stop.set()
            if sampler is not None:
                sampler.join(timeout=3)
            log.close()
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["peak"] = peak
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"] or result["viewer_exit_code"] != 0:
        raise SystemExit("private viewer benchmark did not pass")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
