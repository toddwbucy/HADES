"""The verified provider service: space labels, status route, entry point (#225)."""
import asyncio
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

import httpx
import pytest

from test_verified_jina_provider import provider  # noqa: F401
from test_model_registration import source, registry  # noqa: F401
from embedding import jina_v4, verified_server
from embedding.config import EmbeddingConfig
from embedding.verified_http import create_verified_app
from embedding.verified_jina import VerifiedJinaProvider
from embedding.verified_late import VerifiedJinaLateProvider


def test_space_labels_and_device(provider):
    p, _ = provider
    assert p.space == "conventional"
    assert VerifiedJinaLateProvider(p._registration, max_input_tokens=16).space == "late"
    assert p.device == "cpu"


def test_status_route_answers_without_loading(provider):
    p, _ = provider
    app = create_verified_app(p)

    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as c:
                before = await c.get("/v1/status")
                assert before.status_code == 200
                assert before.json() == {"model": p.model_name, "space": "conventional",
                                         "loaded": False}
                assert (await c.post("/v1/verify")).status_code == 200
                after = await c.get("/v1/status")
                assert after.json()["loaded"] is True
    asyncio.run(run())


def test_parse_args_requires_all_four():
    with pytest.raises(SystemExit):
        verified_server.parse_args(["--registry", "/r", "--provider", "p", "--uds", "/s"])
    with pytest.raises(SystemExit):
        verified_server.parse_args(["--registry", "/r", "--provider", "p", "--space", "both",
                                    "--uds", "/s"])
    args = verified_server.parse_args(["--registry", "/r", "--provider", "p", "--space", "late",
                                       "--uds", "/s/x.sock"])
    assert (args.space, str(args.uds)) == ("late", "/s/x.sock")


def test_socket_parent_must_be_private(tmp_path):
    parent = tmp_path / "run"
    parent.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="0700"):
        verified_server.check_socket_path(parent / "p.sock")
    parent.chmod(0o700)
    assert verified_server.check_socket_path(parent / "p.sock") == parent / "p.sock"


def test_socket_ancestor_symlink_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (real / "run").mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        verified_server.check_socket_path(link / "run" / "p.sock")


def test_stale_socket_unlinked_but_file_refused(tmp_path):
    parent = tmp_path / "run"
    parent.mkdir(mode=0o700)
    path = parent / "p.sock"
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(path))
    s.close()
    assert stat.S_ISSOCK(path.lstat().st_mode)
    verified_server.check_socket_path(path)
    assert not path.exists()
    path.write_text("")
    with pytest.raises(ValueError, match="non-socket"):
        verified_server.check_socket_path(path)


def test_live_socket_refused_and_kept(tmp_path):
    """A second start must not unlink a running instance's socket (#225)."""
    parent = tmp_path / "run"
    parent.mkdir(mode=0o700)
    path = parent / "p.sock"
    live = socket.socket(socket.AF_UNIX)
    live.bind(str(path))
    live.listen()
    try:
        with pytest.raises(ValueError, match="in use"):
            verified_server.check_socket_path(path)
        assert stat.S_ISSOCK(path.lstat().st_mode)
    finally:
        live.close()


def test_build_provider_selects_space(provider):
    p, _ = provider
    cfg = EmbeddingConfig(device="cpu", model_name=p.model_name, idle_timeout_seconds=0)
    late = verified_server.build_provider(p._registration, "late", cfg, max_input_tokens=16)
    conv = verified_server.build_provider(p._registration, "conventional", cfg,
                                          max_input_tokens=16)
    assert isinstance(late, VerifiedJinaLateProvider) and type(conv) is VerifiedJinaProvider


def test_build_provider_refuses_device_fallback(provider, monkeypatch):
    # Force the backend's documented CUDA-unavailable fallback to CPU, so the refusal is
    # exercised on a GPU host too (where an out-of-range ordinal raises a different error).
    monkeypatch.setattr(jina_v4.torch.cuda, "is_available", lambda: False)
    p, _ = provider
    cfg = EmbeddingConfig(device="cuda:7", model_name=p.model_name)
    with pytest.raises(ValueError, match="CPU fallback is refused"):
        verified_server.build_provider(p._registration, "conventional", cfg,
                                       max_input_tokens=16)


def test_build_app_applies_idle_timeout(provider):
    p, _ = provider
    cfg = EmbeddingConfig(device="cpu", model_name=p.model_name, idle_timeout_seconds=0)
    app = verified_server.build_app(p, cfg)
    assert app.state.app_state.config.idle_timeout_seconds == 0


SERVICES = Path(__file__).resolve().parents[1]
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def run_cli(args, env):
    """Run the module as the unit does, importing this checkout's services tree."""
    return subprocess.run([sys.executable, "-m", "embedding.verified_server", *args],
                          capture_output=True, text=True, timeout=60,
                          env={**env, "PYTHONPATH": str(SERVICES)})


def test_main_requires_offline_mode(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE}
    r = run_cli(["--registry", str(tmp_path), "--provider", "p", "--space", "late",
                 "--uds", str(tmp_path / "s.sock")], env)
    assert r.returncode == 1 and "HF_HUB_OFFLINE=1" in r.stderr


def test_main_refuses_missing_registry(tmp_path):
    r = run_cli(["--registry", str(tmp_path / "none"), "--provider", "p", "--space", "late",
                 "--uds", str(tmp_path / "s.sock")], {**os.environ, **OFFLINE})
    assert r.returncode == 1 and "registry root does not exist" in r.stderr
    assert not (tmp_path / "none").exists()


def test_main_serves_and_stops(provider, monkeypatch, tmp_path):
    import uvicorn
    from embedding.registry import ModelRegistry
    p, _ = provider
    root = tmp_path / "serving-registry"
    root.mkdir(mode=0o700)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    uds = run_dir / "provider.sock"
    seen = {}

    async def serve(server, sockets=None):
        """Stand in for uvicorn: own the app lifespan and the socket until told to exit."""
        seen["server"], seen["uds"] = server, server.config.uds
        app = server.config.app
        bound = socket.socket(socket.AF_UNIX)
        bound.bind(server.config.uds)
        try:
            async with app.router.lifespan_context(app):
                seen["loaded"] = p_status(app)
                server.started = True
                while not server.should_exit:
                    await asyncio.sleep(0.001)
        finally:
            bound.close()

    def p_status(app):
        return app.state.app_state.embedder.is_loaded

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    monkeypatch.setattr(ModelRegistry, "load", lambda self, provider_id: p._registration)
    monkeypatch.setattr(EmbeddingConfig, "from_env", classmethod(
        lambda cls: EmbeddingConfig(device="cpu", model_name=p.model_name,
                                    idle_timeout_seconds=0)))
    for key, value in OFFLINE.items():
        monkeypatch.setenv(key, value)
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("code", verified_server.main(
        ["--registry", str(root), "--provider", "verified", "--space", "late",
         "--uds", str(uds)])))
    worker.start()
    deadline = time.monotonic() + 30
    while not getattr(seen.get("server"), "started", False):
        assert worker.is_alive() and time.monotonic() < deadline, result
        time.sleep(0.01)
    assert seen["uds"] == str(uds) and seen["loaded"] is True
    assert uds.exists()
    seen["server"].should_exit = True
    worker.join(30)
    assert not worker.is_alive() and result["code"] == 0
    assert not uds.exists()


SERVE_STUB = """
import sys
from pathlib import Path
from fastapi import FastAPI
from embedding import verified_server
uds = Path(sys.argv[1])
try:
    verified_server.serve(FastAPI(), uds)
finally:
    uds.unlink(missing_ok=True)
"""


def test_serve_exits_cleanly_on_sigterm(tmp_path):
    """A stop signal must end in exit 0 with the socket removed, as main() promises.

    uvicorn re-raises a captured SIGTERM after its graceful shutdown; under the default
    handler that kills the process before main() unlinks the socket (#225). The stub
    mirrors main()'s tail around serve() with a real uvicorn server, not a stand-in.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    uds = run_dir / "provider.sock"
    proc = subprocess.Popen([sys.executable, "-c", SERVE_STUB, str(uds)],
                            env={**os.environ, "PYTHONPATH": str(SERVICES)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not (uds.exists() and stat.S_ISSOCK(uds.lstat().st_mode)):
            assert proc.poll() is None, proc.stderr.read()
            assert time.monotonic() < deadline, "stub server did not bind"
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(45)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert (code, uds.exists()) == (0, False)


def patch_main_inputs(monkeypatch, p):
    """Point main() at the CPU fixture provider without a registry or the real config."""
    from embedding.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "load", lambda self, provider_id: p._registration)
    monkeypatch.setattr(EmbeddingConfig, "from_env", classmethod(
        lambda cls: EmbeddingConfig(device="cpu", model_name=p.model_name,
                                    idle_timeout_seconds=0)))
    for key, value in OFFLINE.items():
        monkeypatch.setenv(key, value)


def private_paths(tmp_path):
    root = tmp_path / "serving-registry"
    root.mkdir(mode=0o700)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    return root, run_dir / "provider.sock"


def test_main_exit_keeps_a_socket_another_instance_serves(provider, monkeypatch, tmp_path):
    """main()'s cleanup removes only a dead socket, never one another process listens on."""
    p, _ = provider
    patch_main_inputs(monkeypatch, p)
    root, uds = private_paths(tmp_path)
    other = socket.socket(socket.AF_UNIX)
    monkeypatch.setattr(verified_server, "serve",
                        lambda app, path: (other.bind(str(path)), other.listen()))
    try:
        assert verified_server.main(["--registry", str(root), "--provider", "verified",
                                     "--space", "late", "--uds", str(uds)]) == 0
        assert stat.S_ISSOCK(uds.lstat().st_mode)
    finally:
        other.close()


def test_main_reports_a_load_refusal_in_one_line(provider, monkeypatch, tmp_path, capsys):
    """A busy card (preflight RuntimeError) must exit 1 with one line, not a traceback."""
    p, _ = provider
    patch_main_inputs(monkeypatch, p)
    root, uds = private_paths(tmp_path)

    def refuse(self):
        raise RuntimeError("refusing to load on cuda:0: 131 MiB free of 24463 MiB")

    monkeypatch.setattr(VerifiedJinaLateProvider, "describe", refuse)
    code = verified_server.main(["--registry", str(root), "--provider", "verified",
                                 "--space", "late", "--uds", str(uds)])
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("verified server: refusing to load on cuda:0")
    assert not uds.exists()


RUN_STUB = """
import sys, time
from pathlib import Path
from fastapi import FastAPI
from embedding import verified_server
uds, hold = Path(sys.argv[1]), float(sys.argv[2])

def load():
    # Stand in for the eager model load: announce the claim, then hold the window
    # in which a second start used to pass the socket check too.
    print("loading", flush=True)
    time.sleep(hold)
    return FastAPI()

raise SystemExit(verified_server.run(uds, load))
"""


def start_run_stub(uds, hold):
    return subprocess.Popen([sys.executable, "-c", RUN_STUB, str(uds), str(hold)],
                            env={**os.environ, "PYTHONPATH": str(SERVICES)},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_second_start_during_load_cannot_take_over_the_socket(tmp_path):
    """Two starts on one path: the second refuses while the first is still loading.

    Without a lock both pass the socket check before either binds; uvicorn then unlinks
    the first instance's live socket and binds the second in its place, leaving the
    first running but unreachable (#225, Codex on #252).
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    uds = run_dir / "provider.sock"
    first = start_run_stub(uds, 3.0)
    second = None
    try:
        assert first.stdout.readline().strip() == "loading", first.stderr.read()
        second = start_run_stub(uds, 0.0)
        try:
            code = second.wait(20)
        except subprocess.TimeoutExpired:
            code = None
        assert code == 1, "second start was not refused"
        assert second.stderr.read().strip() == (
            "verified server: socket path is being started or served by another instance")
        deadline = time.monotonic() + 30
        while not (uds.exists() and stat.S_ISSOCK(uds.lstat().st_mode)):
            assert first.poll() is None, first.stderr.read()
            assert time.monotonic() < deadline, "first instance did not bind"
            time.sleep(0.05)
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.connect(str(uds))
        finally:
            probe.close()
        first.send_signal(signal.SIGTERM)
        assert first.wait(45) == 0
        assert not uds.exists()
    finally:
        for proc in (first, second):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
