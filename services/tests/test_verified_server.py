"""The verified provider service: space labels, status route, entry point (#225)."""
import asyncio
import socket
import stat

import httpx
import pytest

from test_verified_jina_provider import provider  # noqa: F401
from test_model_registration import source, registry  # noqa: F401
from embedding import verified_server
from embedding.verified_http import create_verified_app
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
