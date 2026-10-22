"""Owned private HTTP/Rust lifecycle shared by opt-in model qualification tools."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import tempfile
import time


async def run_rust_contract(provider, context, output, *, repo, target, test, environment, recheck):
    """Keep the provider and child owned through checks, timeout, cancellation and cleanup."""
    import uvicorn
    from embedding.verified_http import create_verified_app

    class OwnedServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            yield

    with tempfile.TemporaryDirectory(prefix="hades-qualified-") as directory:
        socket = Path(directory) / "provider.sock"
        context["socket"] = str(socket)
        context_path = Path(directory) / "context.json"
        context_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        app = create_verified_app(provider)
        app.state.app_state.config.idle_timeout_seconds = 0
        server = OwnedServer(uvicorn.Config(app, uds=str(socket), log_level="warning",
                                            access_log=False, timeout_graceful_shutdown=30))
        serving = asyncio.create_task(server.serve())
        child = None
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError("provider stopped before startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("provider startup timed out")
                await asyncio.sleep(0.05)
            env = dict(os.environ, **{environment + "_CONTEXT": str(context_path),
                                      environment + "_REPORT": str(output / "rust.json")})
            with (output / "rust.log").open("w") as log:
                child = await asyncio.create_subprocess_exec(
                    "cargo", "test", "--locked", "--offline", "-p", "hades-core", "--test", target,
                    "--", "--ignored", "--exact", test, "--nocapture",
                    cwd=repo, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
                code = await asyncio.wait_for(child.wait(), 900)
            if code:
                raise RuntimeError(f"Rust qualification failed ({code}); inspect rust.log")
            result = json.loads((output / "rust.json").read_text())
            if result.get("passed") is not True:
                raise RuntimeError("Rust did not acknowledge success")
            result["tokenizer_recheck"] = recheck(provider, context, result)
            return result
        finally:
            if child is not None and child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(), 30)
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
