"""Real registered loading, tokenizer, pooling, receipts and HTTP; fake model weights only."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
from pathlib import Path
import threading
import types

import httpx
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, PretrainedConfig, Qwen2TokenizerFast

from test_model_registration import REVISION, registry, source  # noqa: F401
from embedding.identity import ProviderReceipt, embedding_space_id
from embedding.jina_v4 import InputTooLargeError
from embedding.registry import MODEL_ID
from embedding.snapshot import ArtifactChangedError
from embedding.verified_http import create_verified_app
from embedding.verified_jina import BehaviorChangedError, ReceiptMismatchError, VerifiedJinaProvider


@pytest.fixture
def provider(source, registry, monkeypatch):
    """Use a real local Qwen2 fast tokenizer and a tiny deterministic raw-forward model."""
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "Passage": 1, "Query": 2, ":": 3,
                                     "one": 4, "two": 5, "three": 6, "é": 7}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = Qwen2TokenizerFast(tokenizer_object=tokenizer, model_max_length=32768)
    fast.save_pretrained(source)
    config = {"hidden_size": 2048, "task_names": ["retrieval", "text-matching", "code"],
              "max_position_embeddings": 32768, "auto_map": {"AutoModel": "modeling_fixture.Fixture"}}
    (source / "config.json").write_text(json.dumps(config))
    registration = registry.register("verified", source, REVISION)
    state = types.SimpleNamespace(calls=[], adapters=[], loads=[], hook=None, invalid=None)

    class Model(torch.nn.Module):
        """Expose token-dependent vectors so tests can verify exact prompts and pooling."""
        def __init__(self):
            """Define real parameter precision and portable model configuration."""
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = PretrainedConfig(**config)
            self.config._attn_implementation = "eager"

        def forward(self, input_ids, attention_mask, task_label, output_vlm_last_hidden_states, **kwargs):
            """Record real token IDs and return deterministic hidden states on CPU."""
            assert task_label in {"retrieval", "code"} and output_vlm_last_hidden_states
            assert not self.training and not torch.is_grad_enabled()
            assert not torch.is_autocast_enabled("cpu")
            state.calls.append(input_ids.tolist()[0])
            state.adapters.append(task_label)
            if state.hook:
                state.hook()
            hidden = torch.ones((1, input_ids.shape[1], 2048), dtype=torch.float32)
            hidden[:, :, 0] = input_ids.float()
            if task_label == "code":
                hidden[:, :, 1] = 5
            if state.invalid == "nan":
                hidden[:, :, 0] = float("nan")
            if state.invalid == "shape":
                hidden = hidden[:, :, :2]
            if state.invalid == "zero":
                hidden.zero_()
            return types.SimpleNamespace(vlm_last_hidden_states=hidden)

    def load(path, **kwargs):
        """Stub acquisition only; keep the actual registered model lifecycle."""
        assert kwargs["local_files_only"] and kwargs["use_safetensors"]
        state.loads.append(path)
        return Model()

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    # Production checks the initialized libraries, not merely these env strings.
    import huggingface_hub.constants as constants
    import transformers.utils.hub as hub
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub, "_is_offline_mode", True)
    result = VerifiedJinaProvider(registration, max_input_tokens=16)
    yield result, state
    result.unload()


def receipt(provider):
    """Obtain the receipt through the explicit successful observation operation."""
    return ProviderReceipt.from_dict(provider.describe()["receipt"])


def embed(provider, expected, texts=None, task="retrieval.passage"):
    """Send a conventional request with explicit model, role and expected identity."""
    return provider.embed(model=MODEL_ID, texts=texts or ["one two"], task=task, expected=expected)


def mutate_snapshot(provider):
    """Simulate an accidental serving-file write while leaving registration intact."""
    path = provider._backend._snapshot.root / "model.safetensors"
    path.chmod(0o600)
    path.write_bytes(b"changed")
    path.chmod(0o400)


def test_observation_derives_space_from_actual_load_and_explicit_behavior(provider):
    """Describe has no inference side effect and advertises only implemented capabilities."""
    p, state = provider
    assert not p.is_loaded
    descriptor = p.describe()
    assert p.is_loaded and len(state.loads) == 1 and not state.calls
    assert descriptor["receipt"]["space_id"] == embedding_space_id(descriptor["space"])
    assert descriptor["model"]["manifest_md5"] == p._registration.manifest.digest()
    assert descriptor["space"]["passage"]["prompt"] == "Passage: "
    assert descriptor["space"]["query"]["prompt"] == "Query: "
    for capability in ("late_chunking", "image_embedding", "specialized_code", "tokenizer_offsets"):
        assert descriptor["capabilities"][capability] is False
    descriptor["space"]["query"]["prompt"] = "caller mutation"
    assert p.describe()["space"]["query"]["prompt"] == "Query: "


def test_exact_prompt_tokenization_masked_mean_and_float32_normalization(provider):
    """Actual input IDs and vectors demonstrate the behavior named by the receipt."""
    p, state = provider
    expected = receipt(p)
    for task, prefix_id in [("retrieval.passage", 1), ("retrieval.query", 2)]:
        result = embed(p, expected, task=task)
        assert state.calls[-1] == [prefix_id, 3, 4, 5]
        assert state.adapters[-1] == "retrieval"
        wanted = np.ones(2048, dtype=np.float32)
        wanted[0] = np.mean([prefix_id, 3, 4, 5])
        wanted /= np.linalg.norm(wanted)
        np.testing.assert_allclose(result["data"][0]["embedding"], wanted, rtol=1e-6)
        assert result["receipt"] == expected.to_dict()
        assert result["usage"]["total_tokens"] == 4


def test_receipt_admission_never_loads_or_silently_rebinds(provider):
    """A stale generation after unload or reload is refused before the first forward pass."""
    p, state = provider
    expected = receipt(p)
    for wrong in [replace(expected, artifact_md5="0" * 32), replace(expected, space_id="0" * 64),
                  replace(expected, serving_generation="another-load")]:
        with pytest.raises(ReceiptMismatchError):
            embed(p, wrong)
    assert not state.calls
    p.unload()
    with pytest.raises(ReceiptMismatchError):
        embed(p, expected)
    assert len(state.loads) == 1
    current = receipt(p)
    assert current.space_id == expected.space_id
    assert current.artifact_md5 == expected.artifact_md5
    assert current.serving_generation != expected.serving_generation
    with pytest.raises(ReceiptMismatchError):
        embed(p, expected)
    embed(p, current)
    assert len(state.calls) == 1


@pytest.mark.parametrize("during", [False, True])
@pytest.mark.parametrize("change", ["files", "training", "precision", "parameter", "tokenizer", "config"])
def test_drift_invalidates_entire_operation(provider, during, change):
    """Artifact, weights, numerical mode and tokenizer changes cannot return a receipt."""
    p, state = provider
    expected = receipt(p)
    def mutate():
        """Apply one ordinary accidental mutation to the loaded provider."""
        if change == "files":
            mutate_snapshot(p)
        elif change == "training":
            p._backend._model.train()
        elif change == "precision":
            p._backend._model.double()
        elif change == "parameter":
            with torch.no_grad():
                p._backend._model.weight.add_(1)
        elif change == "tokenizer":
            p._backend._tokenizer.add_tokens(["new-token"])
        else:
            p._backend._model.config.hidden_size = 1024
    if during:
        state.hook = mutate
    else:
        mutate()
    with pytest.raises((BehaviorChangedError, ArtifactChangedError)):
        embed(p, expected, ["one", "two"])
    assert not p.is_loaded and p._descriptor is None


def test_all_inputs_count_exact_prompts_before_any_forward_pass(provider):
    """An oversized later input cannot cause truncation or partial inference."""
    p, state = provider
    expected = receipt(p)
    embed(p, expected, [" ".join(["one"] * 14)])
    assert len(state.calls[-1]) == 16
    state.calls.clear()
    with pytest.raises(InputTooLargeError) as caught:
        embed(p, expected, ["one", " ".join(["one"] * 15)])
    assert caught.value.index == 1 and caught.value.n_tokens == 17 and caught.value.max_tokens == 16
    assert not state.calls


@pytest.mark.parametrize("invalid", ["nan", "shape", "zero"])
def test_invalid_model_outputs_never_produce_receipted_vectors(provider, invalid):
    """Provider validation rejects wrong widths, nonfinite values, and zero norms."""
    p, state = provider
    expected = receipt(p)
    state.invalid = invalid
    with pytest.raises(RuntimeError):
        embed(p, expected)


def test_unload_waits_for_all_inputs_and_the_final_receipt_check(provider):
    """An idle/shutdown unload cannot change generation between forward passes."""
    p, state = provider
    expected = receipt(p)
    entered, release, unloading = threading.Event(), threading.Event(), threading.Event()
    def forward():
        """Hold an admitted operation until an unload attempt is waiting."""
        entered.set()
        assert release.wait(5)
    def unload():
        """Signal intent before taking the provider lifecycle lock."""
        unloading.set()
        p.unload()
    state.hook = forward
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(embed, p, expected, ["one", "two"])
        try:
            assert entered.wait(5)
            cleanup = pool.submit(unload)
            assert unloading.wait(5) and not cleanup.done()
        finally:
            release.set()
        assert work.result(timeout=5)["receipt"] == expected.to_dict()
        cleanup.result(timeout=5)
    assert len(state.calls) == 2 and not p.is_loaded


def test_http_verification_admission_and_strict_request_schema(provider):
    """Exercise the actual HTTP route with successful and refused bound requests."""
    p, state = provider
    async def run():
        """Own the same lifecycle used by an ASGI server, with no listening port."""
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                observed = await client.post("/v1/verify")
                assert observed.status_code == 200, observed.text
                expected = observed.json()["receipt"]
                request = {"model": MODEL_ID, "input": ["one", "two"], "task": "retrieval.query",
                           "expected_receipt": expected}
                response = await client.post("/v1/embeddings", json=request)
                assert response.status_code == 200, response.text
                assert response.json()["receipt"] == expected and len(response.json()["data"]) == 2
                state.calls.clear()
                for key, value in [("late_chunk", {}), ("images", []), ("encoding_format", "base64"),
                                   ("task", "code"), ("batch_size", True), ("expected_receipt", None)]:
                    bad = dict(request, **{key: value})
                    response = await client.post("/v1/embeddings", json=bad)
                    assert response.status_code == 422, response.text
                for key in ("expected_receipt", "task"):
                    bad = dict(request)
                    del bad[key]
                    assert (await client.post("/v1/embeddings", json=bad)).status_code == 422
                duplicate = json.dumps(request).replace('"serving_generation":', '"serving_generation":"duplicate","serving_generation":')
                assert (await client.post("/v1/embeddings", content=duplicate)).status_code == 422
                bad = copy.deepcopy(request)
                bad["expected_receipt"]["serving_generation"] = "stale"
                response = await client.post("/v1/embeddings", json=bad)
                assert response.status_code == 409 and response.json()["error"]["code"] == "PE_RECEIPT_MISMATCH"
                bad = dict(request, input=["one", " ".join(["one"] * 15)])
                response = await client.post("/v1/embeddings", json=bad)
                assert response.status_code == 400 and response.json()["error"]["input_index"] == 1
                assert not state.calls
                state.hook = lambda: mutate_snapshot(p)
                response = await client.post("/v1/embeddings", json=request)
                assert response.status_code == 409 and "data" not in response.json()
                assert not p.is_loaded
        assert not p.is_loaded
    asyncio.run(run())


def test_operational_budget_and_private_copy_path_do_not_change_embedding_space(provider):
    """Separate copies with equal behavior keep one space while generations differ."""
    first, _ = provider
    second = VerifiedJinaProvider(first._registration, max_input_tokens=8)
    try:
        a, b = first.describe(), second.describe()
        assert a["receipt"]["space_id"] == b["receipt"]["space_id"]
        assert a["receipt"]["serving_generation"] != b["receipt"]["serving_generation"]
        assert first._backend._snapshot.root != second._backend._snapshot.root
        assert a["capabilities"]["max_input_tokens"] == 16
        assert b["capabilities"]["max_input_tokens"] == 8
    finally:
        second.unload()


def test_failed_behavior_resolution_publishes_no_receipt(provider, monkeypatch):
    """A model that loads but cannot establish behavior must release its serving copy."""
    p, _ = provider
    def fail():
        """Inject a post-load adapter qualification failure."""
        raise BehaviorChangedError("synthetic unsupported model behavior")
    monkeypatch.setattr(p, "_derive", fail)
    with pytest.raises(BehaviorChangedError):
        p.describe()
    assert not p.is_loaded and p._descriptor is None
    assert not list(p._registration.sessions.iterdir())


def test_http_body_limit_precedes_inference(provider, monkeypatch):
    """An oversized body is rejected while streaming, before provider work is admitted."""
    import embedding.verified_http as server
    p, state = provider
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 10)
    async def run():
        """Send a bounded synthetic request through the actual ASGI transport."""
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                response = await client.post("/v1/embeddings", content=b" " * 11)
                assert response.status_code == 413
                assert not p.is_loaded and not state.calls
    asyncio.run(run())


def test_cancelled_http_request_drains_before_shutdown_unloads(provider):
    """Cancellation keeps the complete inference and receipt check owned by the service."""
    p, state = provider
    expected = receipt(p)
    entered, release = threading.Event(), threading.Event()
    def block():
        """Hold the first inference until cancellation and shutdown are both pending."""
        entered.set()
        assert release.wait(5)
    state.hook = block
    async def run():
        """Cancel the request task while the provider continues its admitted work."""
        app = create_verified_app(p)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                work = asyncio.create_task(client.post("/v1/embeddings", json={
                    "model": MODEL_ID, "task": "retrieval.passage", "input": ["one", "two"],
                    "expected_receipt": expected.to_dict()}))
                try:
                    async def wait_for_entry():
                        """Wait without blocking the asyncio loop or consuming a worker."""
                        while not entered.is_set():
                            await asyncio.sleep(0.005)
                    await asyncio.wait_for(wait_for_entry(), 3)
                    work.cancel()
                    closing = asyncio.create_task(app.state.app_state.close())
                    await asyncio.sleep(0.02)
                    assert not work.done() and not closing.done()
                    assert app.state.app_state.active_requests == 1 and p.is_loaded
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await work
                await closing
                assert len(state.calls) == 2 and not p.is_loaded
                assert app.state.app_state.active_requests == 0
    asyncio.run(run())
