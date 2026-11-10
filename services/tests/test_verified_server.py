"""The verified provider service: space labels, status route, entry point (#225)."""
import asyncio

import httpx

from test_verified_jina_provider import provider  # noqa: F401
from test_model_registration import source, registry  # noqa: F401
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
