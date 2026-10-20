"""Exact candidate accounting, byte coverage, and fail-closed provider planning."""

import asyncio
from dataclasses import replace
import hashlib
import json
import threading
import unicodedata

import httpx
import pytest
import torch

from test_model_registration import registry, source  # noqa: F401
from test_verified_jina_provider import provider  # noqa: F401
from embedding.identity import ProviderReceipt
from embedding.passage_plan import PlanningError, plan_passages
from embedding.verified_http import create_verified_app
from embedding.verified_jina import BehaviorChangedError, PROMPTS, ReceiptMismatchError


def inspect_plan(plan, text, count):
    """Independently verify exact slices, complete coverage, policy, and observed counts."""
    raw = text.encode("utf-8")
    assert plan["source_sha256"] == hashlib.sha256(raw).hexdigest()
    previous = None
    passages = []
    for span in plan["spans"]:
        start, end = span["start_byte"], span["end_byte"]
        value = raw[start:end].decode("utf-8")
        assert value.strip() and count(value, False) == span["body_tokens"] <= plan["passage_tokens"]
        assert count(value, True) == span["input_tokens"] <= plan["max_input_tokens"]
        if previous is None:
            assert start == 0 and span["overlap_tokens"] == 0
        else:
            assert previous["start_byte"] < start <= previous["end_byte"] < end
            overlap = raw[start:previous["end_byte"]].decode("utf-8")
            assert span["overlap_tokens"] == count(overlap, False) <= plan["overlap_tokens"]
            if plan["overlap_tokens"] == 0:
                assert start == previous["end_byte"]
        passages.append(value)
        previous = span
    assert previous["end_byte"] == len(raw)
    return passages


@pytest.mark.parametrize("overlap", [0, 1, 5, 10])
def test_unicode_counts_prefixes_and_overlap_keep_original_bytes(overlap):
    """NFC changes token accounting while original combining marks/UTF-8 spans survive."""
    text = "café é 東京 🚀 alpha beta\n" * 5

    def count(value, rendered):
        """Use an independent byte-token oracle with NFC and explicit rendered overhead."""
        return len(unicodedata.normalize("NFC", value).encode("utf-8")) + (3 if rendered else 0)

    plan = plan_passages(text, passage_tokens=11, overlap_tokens=overlap, max_input_tokens=14, count=count)
    inspect_plan(plan, text, count)
    assert len(plan["spans"]) > 1
    assert text.encode("utf-8")[plan["spans"][-1]["end_byte"]:] == b""


def test_nonmonotone_counts_are_measured_not_inferred():
    """BPE-like merges can reduce token count; every emitted candidate still fits exactly."""
    def count(value, rendered):
        """Model a sharp count decrease when a token's final character arrives."""
        return len(value.replace("abcdef", "x")) + (2 if rendered else 0)

    text = "abcdef" * 19
    plan = plan_passages(text, passage_tokens=5, overlap_tokens=2, max_input_tokens=7, count=count)
    inspect_plan(plan, text, count)


@pytest.mark.parametrize("kind", ["source", "fanout", "overlap", "calls", "bytes", "no_fit", "deadline"])
def test_resource_or_policy_failure_returns_no_partial_plan(monkeypatch, kind):
    """Refuse oversized sources, work amplification, and unrepresentable minimum passages."""
    import embedding.passage_plan as planning
    text, tokens, overlap, ceiling = "X" * 20, 8, 7, 9
    if kind == "source":
        monkeypatch.setattr(planning, "MAX_TEXT_BYTES", 5)
    elif kind == "fanout":
        monkeypatch.setattr(planning, "MAX_PASSAGES", 1)
    elif kind == "overlap":
        monkeypatch.setattr(planning, "MAX_PASSAGE_BYTES", 10)
    elif kind == "calls":
        monkeypatch.setattr(planning, "MAX_COUNT_CALLS", 1)
    elif kind == "bytes":
        monkeypatch.setattr(planning, "MAX_COUNT_BYTES", 1)
    elif kind == "deadline":
        ticks = iter([0, 0, 31])
        monkeypatch.setattr(planning.time, "monotonic", lambda: next(ticks))
    else:
        ceiling = 1
    with pytest.raises(PlanningError):
        plan_passages(text, passage_tokens=tokens, overlap_tokens=overlap,
                      max_input_tokens=ceiling, count=lambda value, rendered: len(value) + int(rendered))


def test_actual_provider_plans_without_forward_and_embed_accepts_every_slice(provider):
    """Use the real registered tokenizer, receipt admission, and actual raw forward path."""
    p, state = provider
    expected = ProviderReceipt.from_dict(p.describe()["receipt"])
    text = "one two three é " * 8
    plan = p.plan(model=p.model_name, text=text, task="retrieval.passage", expected=expected,
                  passage_tokens=4, overlap_tokens=1, max_input_tokens=32)
    assert plan["receipt"] == expected.to_dict() and plan["max_input_tokens"] == 16
    assert not state.calls

    def count(value, rendered):
        """Independently count the exact serving input, including prompt and special tokens."""
        return len(p._backend._tokenizer(
            (PROMPTS["retrieval.passage"] if rendered else "") + value,
            add_special_tokens=rendered, padding=False, truncation=False,
            split_special_tokens=False)["input_ids"])

    passages = inspect_plan(plan, text, count)
    result = p.embed(model=p.model_name, texts=passages, task="retrieval.passage", expected=expected)
    assert len(result["data"]) == len(passages) == len(state.calls)
    for wrong in [replace(expected, serving_generation="other"), replace(expected, space_id="0"*64)]:
        with pytest.raises(ReceiptMismatchError):
            p.plan(model=p.model_name, text=text, task="retrieval.passage", expected=wrong,
                   passage_tokens=4, overlap_tokens=1, max_input_tokens=16)
    p.unload()
    with pytest.raises(ReceiptMismatchError):
        p.plan(model=p.model_name, text=text, task="retrieval.passage", expected=expected,
               passage_tokens=4, overlap_tokens=1, max_input_tokens=16)
    assert not p.is_loaded


def test_drift_during_planning_discards_plan_and_invalidates_load(provider, monkeypatch):
    """Receipt ownership extends through the final tokenizer/parameter checks."""
    import embedding.verified_jina as verified
    p, state = provider
    expected = ProviderReceipt.from_dict(p.describe()["receipt"])
    original = verified.plan_passages

    def drift(*args, **kwargs):
        """Change a tracked parameter after planning but before the response is admitted."""
        result = original(*args, **kwargs)
        with torch.no_grad():
            p._backend._model.weight.add_(1)
        return result

    monkeypatch.setattr(verified, "plan_passages", drift)
    with pytest.raises(BehaviorChangedError):
        p.plan(model=p.model_name, text="one two", task="retrieval.passage", expected=expected,
               passage_tokens=4, overlap_tokens=1, max_input_tokens=16)
    assert not state.calls and not p.is_loaded


def test_qualification_recounts_persisted_plan_without_planner_helpers(provider):
    """The independent real-run gate rejects stored count corruption and stale receipts."""
    import copy
    from embedding.qualification import check_stored_plan
    p, _ = provider
    expected = ProviderReceipt.from_dict(p.describe()["receipt"])
    text = "one two three é " * 8
    plan = p.plan(model=p.model_name, text=text, task="retrieval.passage", expected=expected,
                  passage_tokens=4, overlap_tokens=1, max_input_tokens=16)
    result = check_stored_plan(p, text, plan)
    assert result["passed"] and result["checked_passages"] == len(plan["spans"])
    for field in ["body_tokens", "input_tokens", "overlap_tokens"]:
        wrong = copy.deepcopy(plan)
        wrong["spans"][1][field] += 1
        with pytest.raises(ValueError):
            check_stored_plan(p, text, wrong)
    wrong = copy.deepcopy(plan)
    wrong["receipt"]["serving_generation"] = "other"
    with pytest.raises(ReceiptMismatchError):
        check_stored_plan(p, text, wrong)


def test_http_planning_refuses_ambiguous_or_unsupported_requests(provider):
    """Exercise strict JSON, role, receipt, policy and complete successful planning over ASGI."""
    p, state = provider
    expected = p.describe()["receipt"]

    async def run():
        """Own the same app lifecycle used for real serving without opening a listener."""
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                request = {"model":p.model_name, "task":"retrieval.passage", "input":"one two "*8,
                           "expected_receipt":expected, "passage_tokens":4,
                           "overlap_tokens":1, "max_input_tokens":16}
                result = await client.post("/v1/passages", json=request)
                assert result.status_code == 200, result.text
                assert len(result.json()["spans"]) > 1 and result.json()["receipt"] == expected
                for key, value in [("input",["one"]),("task","retrieval.query"),("spans",[]),
                                   ("passage_tokens",True),("overlap_tokens",-1),("max_input_tokens",0)]:
                    assert (await client.post("/v1/passages", json=dict(request, **{key:value}))).status_code == 422
                duplicate = json.dumps(request).replace('"passage_tokens":', '"passage_tokens":4,"passage_tokens":')
                assert (await client.post("/v1/passages", content=duplicate)).status_code == 422
                assert (await client.post("/v1/passages", json=dict(request, overlap_tokens=4))).status_code == 400
                wrong = dict(request, expected_receipt=dict(expected, serving_generation="other"))
                assert (await client.post("/v1/passages", json=wrong)).status_code == 409
                assert not state.calls
    asyncio.run(run())


def test_cancelled_planning_drains_before_unload(provider, monkeypatch):
    """A cancelled request retains its tokenizer/load until owned planning work finishes."""
    import embedding.verified_jina as verified
    p, state = provider
    expected = p.describe()["receipt"]
    entered, release = threading.Event(), threading.Event()
    original = verified.plan_passages

    def held(*args, **kwargs):
        """Hold admitted planning so shutdown must wait for its actual worker."""
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(verified, "plan_passages", held)

    async def run():
        """Cancel through the real HTTP ownership path and observe shutdown exclusion."""
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                work = asyncio.create_task(client.post("/v1/passages", json={
                    "model":p.model_name,"task":"retrieval.passage","input":"one two",
                    "expected_receipt":expected,"passage_tokens":4,"overlap_tokens":1,"max_input_tokens":16}))
                try:
                    async def wait_entered():
                        """Observe worker entry without blocking the event loop."""
                        while not entered.is_set():
                            await asyncio.sleep(0.005)
                    await asyncio.wait_for(wait_entered(), 3)
                    work.cancel()
                    closing = asyncio.create_task(app.state.app_state.close())
                    await asyncio.sleep(0.02)
                    assert not work.done() and not closing.done() and p.is_loaded
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await work
                await closing
                assert not p.is_loaded and not state.calls
    asyncio.run(run())
