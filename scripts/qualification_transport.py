"""Owned private HTTP/child lifecycle shared by opt-in model qualification tools."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import tempfile
import time

from embedding.qualification_run import attach_cleanup_errors, error_record


async def _cleanup(child, server, serving, primary, child_timeout):
    """Attempt both owners' cleanup and retain secondary failures on the original error."""
    errors = []
    first = None
    try:
        if child is not None and child.returncode is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(child.wait(), child_timeout)
            except TimeoutError:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await child.wait()
    except BaseException as error:
        first = error
        errors.append(dict(error_record(error), component="child"))
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(serving), 45)
    except BaseException as error:
        if first is None:
            first = error
        errors.append(dict(error_record(error), component="server"))
        if not serving.done():
            server.force_exit = True
            serving.cancel()
        # Retrieve task failures even when cancellation interrupts shutdown.
        await asyncio.gather(serving, return_exceptions=True)
    if errors:
        if primary is not None:
            attach_cleanup_errors(primary, errors)
        else:
            attach_cleanup_errors(first, errors)
            raise first


async def run_http_contract(provider, context, output, *, repo, name, label, command,
                            recheck, timeout, child_timeout):
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
        child, primary = None, None
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError("provider stopped before startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("provider startup timed out")
                await asyncio.sleep(0.05)
            report_path = output / (name + ".json")
            args, environment = command(context_path, report_path)
            with (output / (name + ".log")).open("w") as log:
                child = await asyncio.create_subprocess_exec(
                    *args, cwd=repo, env=dict(os.environ, **environment), stdout=log,
                    stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
                code = await asyncio.wait_for(child.wait(), timeout)
            if code:
                raise RuntimeError(f"{label} failed ({code}); inspect {name}.log")
            result = json.loads(report_path.read_text())
            if result.get("passed") is not True:
                raise RuntimeError(f"{label} did not acknowledge success")
            recheck(provider, context, result)
            return result
        except BaseException as error:
            primary = error
            raise
        finally:
            await _cleanup(child, server, serving, primary, child_timeout)


async def run_rust_contract(provider, context, output, *, repo, target, test, environment, recheck):
    """Run a bounded opt-in Rust transport target without creating a database."""
    def command(context_path, report_path):
        return (["cargo", "test", "--locked", "--offline", "-p", "hades-core", "--test", target,
                 "--", "--ignored", "--exact", test, "--nocapture"],
                {environment + "_CONTEXT": str(context_path), environment + "_REPORT": str(report_path)})

    def checked(current, ctx, result):
        result["tokenizer_recheck"] = recheck(current, ctx, result)

    return await run_http_contract(
        provider, context, output, repo=repo, name="rust", label="Rust qualification",
        command=command, recheck=checked, timeout=900, child_timeout=30,
    )
