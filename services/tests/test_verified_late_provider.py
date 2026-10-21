"""Actual tokenizer/registered lifecycle with tiny CPU weights for contextual pooling."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from test_verified_jina_provider import provider  # noqa: F401
from test_model_registration import source, registry  # noqa: F401
from embedding.identity import ProviderReceipt
from embedding.late_qualification import recheck
from embedding.jina_v4 import InputTooLargeError
from embedding.passage_plan import PlanningError
from embedding.verified_http import create_verified_app
from embedding.verified_jina import BehaviorChangedError, ReceiptMismatchError
from embedding.verified_late import VerifiedJinaLateProvider


@pytest.fixture
def late(provider):
    """Select the contextual contract explicitly, leaving the conventional fixture intact."""
    conventional, state = provider
    result = VerifiedJinaLateProvider(conventional._registration, max_input_tokens=16)
    yield result, state
    result.unload()


def observed(p):
    return ProviderReceipt.from_dict(p.describe()["receipt"])


def plan(p, text, **changes):
    args = dict(model=p.model_name, task="retrieval.passage", expected=observed(p), text=text,
                passage_tokens=3, overlap_tokens=1, context_cap_tokens=16,
                minimum_window_overlap_tokens=3, max_input_tokens=16)
    args.update(changes)
    return p.plan_late(**args)


def embed(p, text="one two three", boundaries=None, expected=None):
    return p.embed_late(model=p.model_name, task="retrieval.passage", text=text,
                        boundaries=[[0, 3], [4, 7], [0, 7]] if boundaries is None else boundaries,
                        expected=observed(p) if expected is None else expected)


def test_contextual_space_is_explicit_and_cannot_fall_back_to_conventional(provider, late):
    conventional, state = provider
    p, _ = late
    ordinary = conventional.describe()
    contextual = p.describe()
    assert ordinary["receipt"]["artifact_md5"] == contextual["receipt"]["artifact_md5"]
    assert ordinary["receipt"]["space_id"] != contextual["receipt"]["space_id"]
    assert ordinary["capabilities"]["late_chunking"] is False
    assert contextual["capabilities"]["late_chunking"] is True
    assert contextual["capabilities"]["tokenizer_offsets"] is True
    assert contextual["capabilities"]["specialized_code"] is False
    with pytest.raises(ValueError, match="contextual passage"):
        p.embed(model=p.model_name, task="retrieval.passage", texts=["one"], expected=observed(p))
    assert not state.calls
    assert p.embed(model=p.model_name, task="retrieval.query", texts=["one"], expected=observed(p))["receipt"] == contextual["receipt"]


def test_one_window_forward_pools_only_intersecting_passage_tokens(late):
    p, state = late
    result = embed(p)
    assert state.calls == [[1, 3, 4, 5, 6]]
    assert result["input_tokens"] == 5
    for row, mean, token_range in zip(result["data"], [4, 5, 4.5], [(2, 3), (3, 4), (2, 4)]):
        expected = np.ones(2048, dtype=np.float32)
        expected[0] = mean
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(row["embedding"], expected, rtol=1e-6)
        assert (row["token_start"], row["token_end"]) == token_range
    unicode = embed(p, "one é two", [[4, 5]])
    assert unicode["data"][0]["char_start"] == 4
    assert unicode["data"][0]["char_end"] == 5


def test_complete_unicode_plan_obeys_independently_recounted_passage_and_window_budgets(late):
    p, state = late
    text = ("one two é three 東京\n" * 12).strip()
    result = plan(p, text)
    assert not state.calls
    assert len(result["windows"]) > 1
    assert len(result["passages"]) == len(result["assignments"])
    source = text.encode("utf-8")
    for name in ("passages", "windows"):
        assert result[name][0]["start_byte"] == 0
        assert result[name][-1]["end_byte"] == len(source)
        previous_end = 0
        for span in result[name]:
            value = source[span["start_byte"]:span["end_byte"]].decode()
            count = lambda s, rendered: len(p._backend._tokenizer(("Passage: " if rendered else "") + s,
                                                                add_special_tokens=rendered,
                                                                truncation=False)["input_ids"])
            assert span["body_tokens"] == count(value, False)
            assert span["input_tokens"] == count(value, True) <= 16
            overlap = source[span["start_byte"]:previous_end].decode() if span["start_byte"] < previous_end else ""
            assert span["overlap_tokens"] == count(overlap, False)
            if previous_end and name == "windows":
                assert span["overlap_tokens"] >= 3
            previous_end = span["end_byte"]
    for passage, index in zip(result["passages"], result["assignments"]):
        window = result["windows"][index]
        assert window["start_byte"] <= passage["start_byte"] < passage["end_byte"] <= window["end_byte"]


def test_invalid_bounds_receipts_and_live_budgets_fail_before_inference(late):
    p, state = late
    for spans in [[], [[True, 3]], [[3, 3]], [[0, 999]], [[0]], [[3, 4]]]:
        with pytest.raises(ValueError):
            embed(p, boundaries=spans)
    with pytest.raises(InputTooLargeError):
        embed(p, "one " * 20, [[0, 3]])
    old = observed(p)
    p.unload()
    observed(p)
    with pytest.raises(ReceiptMismatchError):
        embed(p, expected=old)
    with pytest.raises(PlanningError, match="minimum window overlap"):
        plan(p, "one two", context_cap_tokens=32, max_input_tokens=32, minimum_window_overlap_tokens=17)
    assert not state.calls


def test_changed_loaded_behavior_discards_contextual_results(late):
    p, state = late
    expected = observed(p)
    state.hook = lambda: setattr(p._backend._model.config, "fixture_changed", True)
    with pytest.raises(BehaviorChangedError):
        embed(p, expected=expected)
    assert not p.is_loaded


def test_http_late_routes_require_capabilities_and_strict_receipts(late, provider):
    p, _ = late
    conventional, _ = provider

    async def exercise():
        for selected in [p, conventional]:
            app = create_verified_app(selected)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                    descriptor = (await client.post("/v1/verify")).json()
                    base = dict(model=selected.model_name, task="retrieval.passage", input="one two three",
                                expected_receipt=descriptor["receipt"])
                    response = await client.post("/v1/late-passages", json=dict(base, passage_tokens=3, overlap_tokens=1,
                        context_cap_tokens=16, minimum_window_overlap_tokens=3, max_input_tokens=16))
                    assert response.status_code == (200 if selected is p else 400), response.text
                    response = await client.post("/v1/late-embeddings", json=dict(base, boundaries=[[0, 3]]))
                    assert response.status_code == (200 if selected is p else 400), response.text
                    response = await client.post("/v1/late-embeddings", json=dict(base, boundaries=[[0, 3]], untrusted=True))
                    assert response.status_code == 422
    asyncio.run(exercise())


def test_qualification_recounts_every_window_and_returned_token_coordinate(late):
    p, _ = late
    text = ("one two é three 東京\n" * 8).strip()
    planned = plan(p, text)
    raw = text.encode("utf-8")
    ranges = [None] * len(planned["passages"])
    for index, window in enumerate(planned["windows"]):
        selected = [i for i, selected in enumerate(planned["assignments"]) if selected == index]
        if not selected:
            continue
        boundaries = [[len(raw[window["start_byte"]:planned["passages"][i][key]].decode())
                       for key in ["start_byte", "end_byte"]] for i in selected]
        result = embed(p, raw[window["start_byte"]:window["end_byte"]].decode(), boundaries)
        for i, row in zip(selected, result["data"]):
            ranges[i] = {key: row[key] for key in ["token_start", "token_end"]}
    evidence = {"plan": planned, "token_ranges": ranges}
    assert recheck(p, text, evidence)["passed"] is True
    for failure in ["body", "range", "tail", "receipt"]:
        bad = copy.deepcopy(evidence)
        if failure == "body":
            bad["plan"]["windows"][0]["body_tokens"] += 1
        elif failure == "range":
            bad["token_ranges"][0]["token_end"] -= 1
        elif failure == "tail":
            bad["plan"]["windows"].pop()
        else:
            bad["plan"]["receipt"]["serving_generation"] = "other"
        with pytest.raises(ValueError):
            recheck(p, text, bad)


def test_qualification_cli_refuses_device_fallback_and_overwriting_evidence(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/qualify_verified_late.py"
    spec = importlib.util.spec_from_file_location("late_qualification_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "evidence"
    args = ["--registry", str(tmp_path / "absent"), "--provider", "fixture",
            "--device", "cuda:0", "--output-dir", str(output)]
    previous_umask = os.umask(0o077)
    try:
        assert module.main(args) == 1
        report = json.loads((output / "report.json").read_text())
        assert report["passed"] is False and "fallback is refused" in report["error"]["message"]
        before = (output / "report.json").read_bytes()
        with pytest.raises(FileExistsError):
            module.main(args)
        assert (output / "report.json").read_bytes() == before
    finally:
        os.umask(previous_umask)
