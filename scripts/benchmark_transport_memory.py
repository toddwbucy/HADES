#!/usr/bin/env python3
"""Measure synthetic MCP slow readers on ephemeral loopback sockets, never HADES."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from test_isolated_database import REPO, bounded_process, stop_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-bytes", type=int, default=2*1024*1024-1024)
    args = parser.parse_args()
    if not 1024*1024 <= args.text_bytes <= 2*1024*1024-1024:
        parser.error("text bytes must be between 1 MiB and 2 MiB minus envelope headroom")
    def interrupted(signum, _frame):
        raise RuntimeError(f"transport benchmark interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    root = Path(tempfile.mkdtemp(prefix="hades-transport-bench-", dir="/tmp"))
    print(f"Transport benchmark artifacts: {root}", flush=True)
    env = dict(os.environ, HADES_TRANSPORT_BENCH="1", HADES_TRANSPORT_TEXT_BYTES=str(args.text_bytes))
    command = ["cargo", "test", "--locked", "--offline", "-p", "hades-cli", "--bin", "hades",
               "mcp_sessions::memory_benchmark::measure_mcp_slow_readers", "--",
               "--ignored", "--test-threads=1", "--nocapture"]
    with (root / "run.log").open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True,
                                 preexec_fn=bounded_process)
        try:
            code = child.wait(timeout=180)
        finally:
            stop_group(child)
    results = [json.loads(line.split("TRANSPORT_BENCH ", 1)[1])
               for line in (root / "run.log").read_text().splitlines()
               if "TRANSPORT_BENCH " in line]
    passed = code == 0 and len(results) == 1
    (root / "results.json").write_text(json.dumps({"passed": passed, "exit_code": code,
                                                 "command": command, "results": results}, indent=2) + "\n")
    print(f"Transport benchmark {'PASS' if passed else 'FAIL'}; {root / 'results.json'}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
