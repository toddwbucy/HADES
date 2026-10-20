#!/usr/bin/env python3
"""Qualify an approved local Jina registration and a disposable Rust database flow.

Use the installed ML Python environment with HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1. No model download, live database discovery, or production
service activation is performed. The output directory must not already exist.
"""

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services"))


async def database_roundtrip(provider, context, output, server_args):
    """Own a private verified HTTP server while the strict runner owns its database."""
    import uvicorn
    from embedding.verified_http import create_verified_app
    from embedding.qualification import check_stored_plan

    class OwnedServer(uvicorn.Server):
        """Leave process signals with the qualification owner, including runner cleanup."""
        @contextmanager
        def capture_signals(self):
            """Do not install server-wide signal handlers inside the owned task."""
            yield

    with tempfile.TemporaryDirectory(prefix="hades-jina-") as directory:
        socket = Path(directory) / "provider.sock"
        context["socket"] = str(socket)
        context_path = Path(directory) / "context.json"
        context_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        app = create_verified_app(provider)
        # A cold Rust build can exceed normal idle expiry. This owner retains
        # the observation for the bounded qualification and unloads at shutdown.
        app.state.app_state.config.idle_timeout_seconds = 0
        server = OwnedServer(uvicorn.Config(
            app, uds=str(socket), log_level="warning",
            access_log=False, timeout_graceful_shutdown=30,
        ))
        serving = asyncio.create_task(server.serve())
        child = None
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError("private provider stopped before startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("private provider startup timed out")
                await asyncio.sleep(0.05)
            command = [sys.executable, str(REPO / "scripts/test_isolated_database.py"), *server_args,
                       "--verified-jina", str(context_path),
                       "--verified-jina-report", str(output / "database.json")]
            with (output / "database.log").open("w") as log:
                child = await asyncio.create_subprocess_exec(
                    *command, cwd=REPO, stdout=log, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                # The runner bounds its cargo command and owns cleanup of every
                # database process. Allow its shutdown budget before escalation.
                code = await asyncio.wait_for(child.wait(), timeout=1100)
            if code:
                raise RuntimeError(f"isolated Rust round trip failed ({code}); inspect database.log")
            result = json.loads((output / "database.json").read_text())
            if result.get("passed") is not True:
                raise RuntimeError("database report did not acknowledge success")
            result["planning"]["tokenizer_recheck"] = check_stored_plan(
                provider, context["planning_text"], result["planning"]["plan"])
            return result
        finally:
            if child is not None and child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(), 100)
                except TimeoutError:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await child.wait()
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(serving), 45)
            except TimeoutError:
                server.force_exit = True
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)


def main(argv=None):
    """Run fixed qualification gates and publish an explicit success or failure report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--device", required=True, help="explicit cpu or CUDA device; no fallback")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True, help="new private evidence directory")
    server = parser.add_mutually_exclusive_group(required=True)
    server.add_argument("--arangod", type=Path, help="existing binary for a private Unix-only server")
    server.add_argument("--docker", action="store_true", help="use the runner's pinned isolated image")
    args = parser.parse_args(argv)
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting")
    if args.device != "cpu" and not (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        parser.error("device must be cpu or cuda:<ordinal>")
    if not 32 <= args.max_input_tokens <= 8192:
        parser.error("qualification ceiling must be 32..8192 tokens")
    if args.arangod and not args.arangod.is_file():
        parser.error("arangod must name an existing binary")
    os.umask(0o077)
    output = args.output_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"passed": False, "started_at": datetime.now(timezone.utc).isoformat(),
              "stage": "preflight"}
    provider = None
    interrupted_run = False
    try:
        import torch
        from embedding.qualification import load_cases, qualify
        from embedding.registry import ModelRegistry
        from embedding.verified_jina import VerifiedJinaProvider
        if args.device.startswith("cuda:") and (
            not torch.cuda.is_available() or int(args.device[5:]) >= torch.cuda.device_count()
        ):
            raise RuntimeError("requested CUDA device is unavailable; CPU fallback is refused")
        cases, digest = load_cases(REPO / "services/tests/fixtures/jina_qualification.json")
        report["suite_sha256"] = digest
        report["implementation_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["implementation_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True).strip())
        registration = ModelRegistry(args.registry).load(args.provider)
        report["manifest"] = json.loads(registration.manifest.canonical_bytes())
        provider = VerifiedJinaProvider(registration, device=args.device,
                                       use_fp16=args.device.startswith("cuda:"),
                                       max_input_tokens=args.max_input_tokens)
        if provider._backend.device != args.device:
            raise RuntimeError("provider selected a different device")
        warnings.filterwarnings("error", message="Found missing adapter keys.*")
        report["stage"] = "provider"
        print("Running fixed provider qualification gates...", flush=True)
        report["provider"] = {}
        qualify(provider, cases, evidence=report["provider"])
        report["stage"] = "database"
        print("Running verified HTTP -> Rust -> disposable database round trip...", flush=True)
        context = {"provider_id": args.provider, "description": report["provider"]["description"],
                   "manifest": report["manifest"], "cases": cases, "suite_sha256": digest,
                   "planning_text": (REPO / "services/tests/fixtures/jina_passage_planning.txt").read_text(encoding="utf-8")}
        server_args = ["--docker"] if args.docker else ["--arangod", str(args.arangod.absolute())]
        report["database"] = asyncio.run(database_roundtrip(provider, context, output, server_args))
        report.update(passed=True, stage="complete")
    except (Exception, KeyboardInterrupt) as error:
        interrupted_run = isinstance(error, KeyboardInterrupt)
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        print(f"Qualification failed during {report['stage']}: {error}", file=sys.stderr)
    finally:
        if provider is not None:
            try:
                provider.unload()
            except Exception as error:
                if report["passed"]:
                    report["stage"] = "cleanup"
                report["passed"] = False
                report["cleanup_error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Qualification evidence: {output}", flush=True)
    return 130 if interrupted_run else (0 if report["passed"] else 1)


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        """Unwind through owned provider, runner, and evidence cleanup on termination."""
        raise KeyboardInterrupt("qualification interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
