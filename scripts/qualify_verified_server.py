#!/usr/bin/env python3
"""Qualify python -m embedding.verified_server over its private Unix socket (#225).

Starts the real service as a child process with already registered local assets,
then checks, over the socket only: eager load in GET /v1/status, the selected space
in POST /v1/verify, the matching planning endpoint, and a clean SIGTERM shutdown that
removes the socket. Run it with the operator's embedder.env sourced; the legacy
embedder must be stopped first, because two loads of the model do not share a card.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

import httpx

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "jinaai/jina-embeddings-v4"
PLANNING_TEXT = REPO / "services/tests/fixtures/jina_passage_planning.txt"
STARTUP_SECONDS = 180
SHUTDOWN_SECONDS = 45
# The first managed database's late policy (db-setup-door.md, Part C defaults).
PASSAGE_TOKENS, OVERLAP_TOKENS = 500, 200
CONTEXT_CAP_TOKENS, MINIMUM_WINDOW_OVERLAP_TOKENS = 16384, 1000


def check_description(description: dict, space: str) -> None:
    """Refuse a verify body whose model or capabilities do not match the selected space.

    The late and conventional providers are distinct spaces; the late-chunk and
    tokenizer-offset capabilities are what setup uses to tell them apart, so both
    must agree with the flag, and with each other.
    """
    capabilities = description["capabilities"]
    late = space == "late"
    if not (capabilities["late_chunking"] is late and capabilities["tokenizer_offsets"] is late):
        raise ValueError(f"verify reports capabilities that do not match --space {space}")
    if description["model"]["id"] != MODEL_ID:
        raise ValueError("verify reports an unexpected model")


def planning_request(space: str, text: str, receipt: dict,
                     max_input_tokens: int) -> tuple[str, dict]:
    """Build the planning call for the selected space, with field names from verified_http."""
    body = {"model": MODEL_ID, "input": text, "task": "retrieval.passage",
            "expected_receipt": receipt, "passage_tokens": PASSAGE_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS, "max_input_tokens": max_input_tokens}
    if space == "late":
        body.update(context_cap_tokens=min(CONTEXT_CAP_TOKENS, max_input_tokens),
                    minimum_window_overlap_tokens=MINIMUM_WINDOW_OVERLAP_TOKENS)
        return "/v1/late-passages", body
    return "/v1/passages", body


def require(condition: bool, message: str) -> None:
    """Fail the qualification with a message that names the broken expectation."""
    if not condition:
        raise RuntimeError(message)


def _raise_interrupt(signum, frame):
    """Treat SIGTERM like Ctrl-C so the child is always stopped and the report written."""
    raise KeyboardInterrupt


def _stop(proc: subprocess.Popen) -> None:
    """Kill a child that is still running after a failure or interrupt."""
    if proc.poll() is None:
        proc.kill()
        proc.wait(SHUTDOWN_SECONDS)


def qualify(args, report: dict, output: Path) -> None:
    """Run the service and every check; raise on the first failed expectation."""
    report["implementation_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
        check=True).stdout.strip()
    text = PLANNING_TEXT.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="hades-verified-server-") as directory:
        sock = Path(directory) / "provider.sock"
        env = dict(os.environ, HADES_EMBEDDER_DEVICE=args.device,
                   HADES_EMBEDDER_IDLE_TIMEOUT=str(args.idle_timeout),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        command = [sys.executable, "-m", "embedding.verified_server", "--registry",
                   str(args.registry), "--provider", args.provider, "--space", args.space,
                   "--uds", str(sock)]
        with (output / "server.log").open("w") as log:
            started = time.monotonic()
            proc = subprocess.Popen(command, cwd=REPO / "services", env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                while not (sock.exists() and stat.S_ISSOCK(sock.lstat().st_mode)):
                    require(proc.poll() is None,
                            f"service exited during startup ({proc.returncode}); see server.log")
                    require(time.monotonic() - started < STARTUP_SECONDS,
                            "service startup timed out")
                    time.sleep(0.2)
                transport = httpx.HTTPTransport(uds=str(sock))
                with httpx.Client(transport=transport, base_url="http://provider",
                                  timeout=600) as client:
                    before = client.get("/v1/status")
                    require(before.status_code == 200, f"status returned {before.status_code}")
                    report["status_before"] = before.json()
                    report["startup_seconds"] = round(time.monotonic() - started, 3)
                    require(report["status_before"]["space"] == args.space,
                            "status reports a different space")
                    require(report["status_before"]["loaded"] is True,
                            "status reports the model unloaded after an eager start")
                    verify = client.post("/v1/verify")
                    require(verify.status_code == 200, f"verify returned {verify.status_code}")
                    report["description"] = description = verify.json()
                    check_description(description, args.space)
                    path, body = planning_request(args.space, text, description["receipt"],
                                                  description["capabilities"]["max_input_tokens"])
                    report["planning_endpoint"] = path
                    planned = client.post(path, json=body)
                    report["planning_status"] = planned.status_code
                    require(planned.status_code == 200, f"{path} returned {planned.status_code}")
                    after = client.get("/v1/status")
                    require(after.status_code == 200, f"status returned {after.status_code}")
                    report["status_after"] = after.json()
                stopping = time.monotonic()
                proc.send_signal(signal.SIGTERM)
                code = proc.wait(SHUTDOWN_SECONDS)
                report["shutdown_seconds"] = round(time.monotonic() - stopping, 3)
                require(code == 0, f"service exited {code} after SIGTERM")
                require(not sock.exists(), "service left its socket behind")
            finally:
                _stop(proc)


def main(argv=None) -> int:
    """Write report.json into a new private directory; exit 0 pass, 1 fail, 130 interrupt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--space", required=True, choices=("conventional", "late"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idle-timeout", type=float, default=0)
    args = parser.parse_args(argv)
    # The child runs with cwd services/, so a relative --registry would name a different
    # directory there. Resolve it against the operator's cwd; resolve() also shows the
    # real path the server's symlink-ancestor refusal will judge (#225).
    args.registry = args.registry.resolve(strict=False)
    os.umask(0o077)
    # Evidence never overwrites an earlier run.
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"protocol": "hades-verified-server-qualification-v1", "space": args.space,
              "device": args.device, "idle_timeout": args.idle_timeout, "passed": False}
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    code = 1
    try:
        qualify(args, report, args.output_dir)
        report["passed"] = True
        code = 0
    except KeyboardInterrupt:
        report["error"] = "interrupted"
        code = 130
    except Exception as exc:  # recorded in the evidence, not swallowed: passed stays false
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.signal(signal.SIGTERM, previous)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
