"""Exercise real registration/tokenization with independently observable code routing."""

import asyncio
import json

import httpx
import numpy as np
import pytest

from test_model_registration import REVISION, registry, source  # noqa: F401
from test_verified_jina_provider import provider  # noqa: F401
from embedding.identity import ProviderReceipt, embedding_space_id
from embedding.jina_v4 import InputTooLargeError
from embedding.registry import MODEL_ID
from embedding.verified_code import VerifiedJinaCodeProvider
from embedding.verified_http import create_verified_app
from embedding.verified_jina import BehaviorChangedError, ReceiptMismatchError


@pytest.fixture
def code_provider(provider):
    """Give documents and code intentionally different serving budgets."""
    original, state = provider
    result = VerifiedJinaCodeProvider(original._registration, max_input_tokens=16,
                                      code_max_input_tokens=8)
    yield result, state
    result.unload()


def receipts(provider):
    """Obtain both receipts from one observation, without making either a default."""
    return {kind: ProviderReceipt.from_dict(value["receipt"])
            for kind, value in provider.describe()["workloads"].items()}


def embed(provider, expected, texts=None, task="code"):
    """Send an explicitly selected task and receipt to the real raw-forward adapter."""
    return provider.embed(model=MODEL_ID, texts=texts or ["one two"], task=task, expected=expected)


def test_description_preserves_documents_and_separates_code(code_provider, provider):
    """V2 adds a workload without changing conventional document embedding behavior."""
    p, _ = code_provider
    original, _ = provider
    old, description = original.describe(), p.describe()
    documents, code = (description["workloads"][kind] for kind in ("documents", "code"))
    assert description["format"] == "hades-provider-description-v2"
    assert description["model"] == old["model"]
    assert documents["space"] == old["space"]
    assert documents["capabilities"] == old["capabilities"]
    assert code["receipt"]["space_id"] == embedding_space_id(code["space"])
    assert code["receipt"]["space_id"] != documents["receipt"]["space_id"]
    for key in ("artifact_md5", "serving_generation"):
        assert code["receipt"][key] == documents["receipt"][key]
    assert code["space"]["passage"] == code["space"]["query"]
    assert code["capabilities"] == dict(documents["capabilities"], specialized_code=True,
                                        tasks=["code"], max_input_tokens=8)
    code["space"]["query"]["prompt"] = "caller mutation"
    assert p.describe()["workloads"]["code"]["space"]["query"]["prompt"] == "Passage: "


@pytest.mark.parametrize("task,prefix,adapter", [
    ("retrieval.passage", 1, "retrieval"), ("retrieval.query", 2, "retrieval"), ("code", 1, "code"),
])
def test_selected_prompt_adapter_and_receipt(code_provider, task, prefix, adapter):
    """Independent vector arithmetic detects accidentally using the retrieval LoRA for code."""
    p, state = code_provider
    expected = receipts(p)["code" if task == "code" else "documents"]
    result = embed(p, expected, task=task)
    assert state.calls == [[prefix, 3, 4, 5]] and state.adapters == [adapter]
    wanted = np.ones(2048, dtype=np.float32)
    wanted[0] = np.mean([prefix, 3, 4, 5])
    wanted[1] = 5 if task == "code" else 1
    wanted /= np.linalg.norm(wanted)
    np.testing.assert_allclose(result["data"][0]["embedding"], wanted, rtol=1e-6)
    assert result["receipt"] == expected.to_dict()


def test_equal_width_receipts_are_not_interchangeable(code_provider):
    """Same artifact, generation and dimension do not admit the other embedding space."""
    p, state = code_provider
    observed = receipts(p)
    for task, wrong in [("code", "documents"), ("retrieval.query", "code"), ("retrieval.passage", "code")]:
        with pytest.raises(ReceiptMismatchError):
            embed(p, observed[wrong], task=task)
    with pytest.raises(ReceiptMismatchError):
        p.plan(model=MODEL_ID, text="one two", task="code", expected=observed["documents"],
               passage_tokens=6, overlap_tokens=1, max_input_tokens=16)
    assert not state.calls


@pytest.mark.parametrize("code_limit", [8, 24])
def test_per_workload_ceiling_checks_entire_batch_before_inference(provider, code_limit):
    """Code may have a smaller or larger budget than documents without inheriting it."""
    original, state = provider
    p = VerifiedJinaCodeProvider(original._registration, max_input_tokens=16,
                                 code_max_input_tokens=code_limit)
    try:
        observed = receipts(p)
        for task, kind, limit in [("code", "code", code_limit), ("retrieval.passage", "documents", 16)]:
            embed(p, observed[kind], [" ".join(["one"] * (limit - 2))], task)
            assert len(state.calls[-1]) == limit
            state.calls.clear()
            with pytest.raises(InputTooLargeError) as caught:
                embed(p, observed[kind], ["one", " ".join(["one"] * (limit - 1))], task)
            assert (caught.value.index, caught.value.n_tokens, caught.value.max_tokens) == (1, limit + 1, limit)
            assert not state.calls
    finally:
        p.unload()


@pytest.mark.parametrize("saved_limit", [6, 16])
def test_code_planning_uses_selected_live_and_saved_limits(code_provider, saved_limit):
    """Recount UTF-8 spans independently and require complete source coverage."""
    p, state = code_provider
    expected = receipts(p)["code"]
    text = "one é two three " * 6
    plan = p.plan(model=MODEL_ID, text=text, task="code", expected=expected,
                  passage_tokens=6, overlap_tokens=1, max_input_tokens=saved_limit)
    assert plan["max_input_tokens"] == min(saved_limit, 8)
    assert plan["receipt"] == expected.to_dict() and plan["task"] == "code"
    assert not state.calls and len(plan["spans"]) > 1
    raw, end = text.encode(), 0
    for span in plan["spans"]:
        assert span["start_byte"] <= end < span["end_byte"]
        body = raw[span["start_byte"]:span["end_byte"]].decode()
        tokens = p._backend._tokenizer("Passage: " + body, add_special_tokens=True)["input_ids"]
        assert len(tokens) == span["input_tokens"] <= min(saved_limit, 8)
        end = span["end_byte"]
    assert end == len(raw)


@pytest.mark.parametrize("change", ["missing_adapter", "context", "code_budget"])
@pytest.mark.parametrize("during", [False, True])
def test_code_drift_invalidates_both_observations(code_provider, change, during):
    """A code change cannot leave document receipts valid on the same observed load."""
    p, state = code_provider
    observed = receipts(p)
    def mutate():
        if change == "missing_adapter":
            p._backend._model.config.task_names.remove("code")
        elif change == "context":
            p._backend._tokenizer.model_max_length = 7
        else:
            p._code_max_input_tokens = 7
    if during:
        state.hook = mutate
    else:
        mutate()
    with pytest.raises(BehaviorChangedError):
        embed(p, observed["code"])
    assert not p.is_loaded
    with pytest.raises(ReceiptMismatchError):
        embed(p, observed["documents"], task="retrieval.query")


def test_reload_refreshes_both_receipts_without_changing_spaces(code_provider):
    """No request implicitly reloads or substitutes a generation for either workload."""
    p, state = code_provider
    old = receipts(p)
    p.unload()
    for task, kind in [("code", "code"), ("retrieval.passage", "documents")]:
        with pytest.raises(ReceiptMismatchError):
            embed(p, old[kind], task=task)
    assert len(state.loads) == 1
    new = receipts(p)
    for kind in old:
        assert old[kind].space_id == new[kind].space_id
        assert old[kind].serving_generation != new[kind].serving_generation
    for task, kind in [("code", "code"), ("retrieval.query", "documents")]:
        with pytest.raises(ReceiptMismatchError):
            embed(p, old[kind], task=task)
        embed(p, new[kind], task=task)


@pytest.mark.parametrize("limit", [True, 0, -1, 32769, 1.5, "8"])
def test_invalid_code_budget_never_loads(provider, limit):
    original, state = provider
    with pytest.raises(ValueError, match="code token budget"):
        VerifiedJinaCodeProvider(original._registration, code_max_input_tokens=limit)
    assert not state.loads


def test_http_code_admission_limits_planning_and_late_refusal(code_provider):
    """The factory exposes code only with strict receipts and never claims contextual code."""
    p, state = code_provider
    async def run():
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                description = (await client.post("/v1/verify")).json()
                expected = description["workloads"]["code"]["receipt"]
                request = dict(model=MODEL_ID, input="one two", task="code", expected_receipt=expected)
                result = await client.post("/v1/embeddings", json=request)
                assert result.status_code == 200 and result.json()["receipt"] == expected
                state.calls.clear()
                wrong = dict(request, expected_receipt=description["workloads"]["documents"]["receipt"])
                assert (await client.post("/v1/embeddings", json=wrong)).status_code == 409
                duplicate = json.dumps(request).replace('"task":', '"task":"code","task":')
                assert (await client.post("/v1/embeddings", content=duplicate)).status_code == 422
                result = await client.post("/v1/embeddings", json=dict(request, input="one " * 7))
                assert result.status_code == 400
                assert result.json()["error"]["ceiling"] == 8
                assert result.json()["error"]["reported_tokens"] == 9
                planning = dict(request, input="one two three " * 6, passage_tokens=6,
                                overlap_tokens=1, max_input_tokens=16)
                result = await client.post("/v1/passages", json=planning)
                assert result.status_code == 200 and result.json()["max_input_tokens"] == 8
                assert not state.calls
                late = dict(request, boundaries=[[0, 3]])
                assert (await client.post("/v1/late-embeddings", json=late)).status_code == 422
                late.update(task="retrieval.passage", expected_receipt=wrong["expected_receipt"])
                result = await client.post("/v1/late-embeddings", json=late)
                assert result.status_code == 400 and result.json()["error"]["code"] == "PE_UNSUPPORTED_CAPABILITY"
        assert not p.is_loaded
    asyncio.run(run())
