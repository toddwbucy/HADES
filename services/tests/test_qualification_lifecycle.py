"""Preserve primary and cleanup failures through the shared qualification owner."""

import asyncio
import json
import os
from pathlib import Path
import types

import pytest

from test_model_registration import registry, source  # noqa: F401
from test_verified_jina_provider import provider  # noqa: F401
from test_code_qualification import script
from embedding.qualification_run import QualificationRun

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("child_fails", [False, True])
def test_server_cleanup_failure_preserves_primary_error(provider, tmp_path, monkeypatch, child_fails):
    """Run actual app/provider shutdown with both successful and failed child outcomes."""
    import uvicorn
    p, _ = provider
    p.describe()
    transport = script("qualification_transport")

    async def serve(server, **_kwargs):
        app = server.config.app
        async with app.router.lifespan_context(app):
            server.started = True
            while not server.should_exit:
                await asyncio.sleep(0.001)
        raise ValueError("server shutdown fixture")

    async def child(*_args, **kwargs):
        Path(kwargs["env"]["FIXTURE_REPORT"]).write_text('{"passed": true}')
        class Child:
            returncode = 1 if child_fails else 0
            async def wait(self):
                return self.returncode
        return Child()

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", child)
    with QualificationRun(tmp_path / "evidence", ROOT) as run:
        run.own(p)
        run.report["stage"] = "rust_transport"
        asyncio.run(transport.run_rust_contract(p, {}, run.output, repo=ROOT, target="fixture",
            test="fixture", environment="FIXTURE", recheck=lambda *_: {"passed": True}))
        run.report.update(passed=True, stage="complete")
    report = json.loads((run.output / "report.json").read_text())
    assert run.exit_code == 1 and report["passed"] is False
    assert report["error"]["message"].startswith("Rust qualification failed" if child_fails else "server shutdown fixture")
    assert report["cleanup_errors"] == [
        {"component": "server", "type": "ValueError", "message": "server shutdown fixture"},
    ]
    assert not p.is_loaded and not list(p._registration.sessions.iterdir())


def test_child_cleanup_failure_does_not_skip_server_cleanup(monkeypatch):
    """Both owners get a cleanup attempt; neither secondary error replaces the first failure."""
    transport = script("qualification_transport")
    primary = RuntimeError("original transport error")
    class Child:
        returncode, pid = None, 12345
        async def wait(self):
            raise ValueError("child cleanup fixture")
    async def failed_server():
        raise ValueError("server cleanup fixture")
    monkeypatch.setattr(os, "killpg", lambda *_: None)
    async def run():
        server = types.SimpleNamespace(should_exit=False)
        serving = asyncio.create_task(failed_server())
        await transport._cleanup(Child(), server, serving, primary, 1)
        assert server.should_exit and serving.done()
    asyncio.run(run())
    assert str(primary) == "original transport error"
    assert [row["component"] for row in primary.qualification_cleanup_errors] == ["child", "server"]


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_interruption_retains_evidence_and_releases_owner(tmp_path, during_cleanup):
    calls = []
    def unload():
        calls.append("unload")
        if during_cleanup:
            raise KeyboardInterrupt("cleanup interruption")
    with QualificationRun(tmp_path / "evidence", ROOT) as run:
        run.own(types.SimpleNamespace(unload=unload))
        run.report["stage"] = "provider"
        if not during_cleanup:
            raise KeyboardInterrupt("provider interruption")
        run.report.update(passed=True, stage="complete")
    report = json.loads((run.output / "report.json").read_text())
    assert run.exit_code == 130 and report["passed"] is False and calls == ["unload"]
    field = "cleanup_error" if during_cleanup else "error"
    assert report[field]["type"] == "KeyboardInterrupt"


def test_existing_evidence_restores_umask_without_overwriting(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "report.json").write_text("keep prior evidence")
    previous = os.umask(0o022)
    try:
        with pytest.raises(FileExistsError):
            with QualificationRun(output, ROOT):
                pytest.fail("existing evidence was admitted")
        assert os.umask(0o022) == 0o022
        assert (output / "report.json").read_text() == "keep prior evidence"
    finally:
        os.umask(previous)
