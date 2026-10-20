"""Exercise the qualification gates on CPU; never load real weights in ordinary CI."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_model_registration import registry, source  # noqa: F401
from test_verified_jina_provider import provider  # noqa: F401
from embedding.identity import ProviderReceipt
from embedding.jina_v4 import EMBEDDING_DIM
from embedding.qualification import checked_vectors, load_cases, qualify, rank, tokenizer_observation

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(__file__).parent / "fixtures/jina_qualification.json"


def test_complete_protocol_observes_boundaries_and_real_reload_lifecycle(provider):
    """Run real tokenizer/receipt/lifecycle logic with deterministic synthetic weights."""
    p, state = provider
    cases = {"documents": [{"id": "one", "text": "one"}],
             "queries": [{"id": "one", "text": "one", "expected_document": "one"}],
             "tokenizer_samples": ["one"]}
    result = qualify(p, cases, adapter_check=lambda _: 1)
    assert result["retrieval"][0]["rank"] == 1
    assert result["repeat_cosine"] >= 0.99999
    assert result["reload_cosine"] >= 0.9999
    assert len(state.loads) == 2
    assert len(state.calls) == 6  # No forward for overflow or any stale/wrong receipt.
    assert all(row["accepted_tokens"] == 16 and row["rejected_tokens"] == 17
               and row["overflow_forward_calls"] == 0 for row in result["boundaries"])
    assert result["initial_receipt"]["serving_generation"] != result["description"]["receipt"]["serving_generation"]


@pytest.mark.parametrize("failure", ["nan", "shape", "zero"])
def test_broken_model_outputs_never_pass_the_protocol(provider, failure):
    """A bad forward result must fail qualification rather than produce a success report."""
    p, state = provider
    state.invalid = failure
    cases = {"documents": [{"id": "one", "text": "one"}],
             "queries": [{"id": "one", "text": "one", "expected_document": "one"}],
             "tokenizer_samples": ["one"]}
    evidence = {}
    with pytest.raises(RuntimeError):
        qualify(p, cases, adapter_check=lambda _: 1, evidence=evidence)
    assert evidence["adapter_tensors_verified"] == 1
    assert evidence["description"]["receipt"]
    assert "reload_cosine" not in evidence


def test_retrieval_gate_rejects_wrong_first_result_and_missing_queries():
    """Keep the predeclared ranking criterion strict even when the expected item ranks second."""
    passages = np.zeros((2, EMBEDDING_DIM))
    passages[0, 0] = passages[1, 1] = 1
    vectors = passages[1:2]
    documents = [{"id": "expected"}, {"id": "wrong"}]
    queries = [{"id": "query", "expected_document": "expected"}]
    with pytest.raises(ValueError, match="retrieval smoke"):
        rank(documents, passages, queries, vectors)
    with pytest.raises(ValueError, match="incomplete"):
        rank(documents, passages, queries, vectors[:0])


def test_tokenizer_gate_requires_declared_normalization_and_preserves_whitespace(monkeypatch):
    """Allow explicit NFC, reject other normalizers, and independently detect decoding loss."""
    from tokenizers import Tokenizer, models, normalizers
    from transformers import Qwen2TokenizerFast
    raw = Tokenizer(models.WordLevel({"<unk>": 0, "é": 1, "  é\t": 2}, unk_token="<unk>"))
    raw.normalizer = normalizers.NFC()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw)
    result = tokenizer_observation(tokenizer, ["é", "é", "  é\t"])
    assert result["normalization"] == "NFC"
    assert [row["decoded"] for row in result["samples"]] == ["é", "é", "  é\t"]
    assert result["samples"][0]["ids"] == result["samples"][1]["ids"]
    with monkeypatch.context() as broken:
        broken.setattr(tokenizer, "decode", lambda *_args, **_kwargs: "é")
        with pytest.raises(ValueError, match="whitespace"):
            tokenizer_observation(tokenizer, ["  é\t"])
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(ValueError, match="only identity or declared NFC"):
        tokenizer_observation(tokenizer, ["É"])


@pytest.mark.parametrize("failure", ["model", "receipt", "width", "norm", "finite", "index", "missing"])
def test_response_checks_reject_false_success(provider, failure):
    """Validate response identity, complete ordering, and numerical representation independently."""
    p, _ = provider
    description = p.describe()
    expected = ProviderReceipt.from_dict(description["receipt"])
    vector = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    response = {"model": p.model_name, "receipt": expected.to_dict(),
                "data": [{"index": 0, "embedding": vector}]}
    if failure == "model":
        response["model"] = "other"
    elif failure == "receipt":
        response["receipt"]["serving_generation"] = "other"
    elif failure == "width":
        vector.pop()
    elif failure == "norm":
        vector[0] = 2.0
    elif failure == "finite":
        vector[0] = float("nan")
    elif failure == "index":
        response["data"][0]["index"] = 1
    else:
        response["data"].clear()
    with pytest.raises(ValueError):
        checked_vectors(response, expected, p.model_name, 1)


def test_cases_are_pinned_complete_and_bounded(tmp_path):
    """Reject ambiguous case files before any model or provider work begins."""
    cases, digest = load_cases(CASES)
    assert len(digest) == 64 and len(cases["documents"]) == 8 and len(cases["queries"]) == 6
    path = tmp_path / "cases.json"
    for change in ("duplicate", "unknown", "empty", "extra", "oversized"):
        bad = copy.deepcopy(cases)
        if change == "duplicate":
            bad["documents"].append(bad["documents"][0])
        elif change == "unknown":
            bad["queries"][0]["expected_document"] = "missing"
        elif change == "empty":
            bad["queries"] = []
        elif change == "extra":
            bad["documents"][0]["unapproved"] = True
        else:
            bad["documents"][0]["text"] = "x" * (128 * 1024)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_cases(path)
    path.write_text('{"format":1,"format":2}')
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_cli_refuses_gpu_fallback_and_preserves_existing_evidence(tmp_path, monkeypatch):
    """An unavailable CUDA device records failure without starting a model or database."""
    spec = importlib.util.spec_from_file_location("qualification_cli", ROOT / "scripts/qualify_verified_jina.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "evidence"
    args = ["--registry", str(tmp_path / "absent"), "--provider", "fixture",
            "--device", "cuda:0", "--output-dir", str(output), "--arangod", "/bin/true"]
    previous_umask = os.umask(0o077)
    try:
        assert module.main(args) == 1
        report = json.loads((output / "report.json").read_text())
        assert report["passed"] is False and report["stage"] == "preflight"
        assert "fallback is refused" in report["error"]["message"]
        before = (output / "report.json").read_bytes()
        with pytest.raises(FileExistsError):
            module.main(args)
        assert (output / "report.json").read_bytes() == before
    finally:
        os.umask(previous_umask)


def test_cli_cleanup_failure_cannot_leave_a_success_report(tmp_path, registry, provider, monkeypatch):
    """Retain failed cleanup evidence even when both qualification stages reported success."""
    import embedding.qualification as protocol
    import embedding.verified_jina as verified
    spec = importlib.util.spec_from_file_location("qualification_cli", ROOT / "scripts/qualify_verified_jina.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p, _ = provider
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(verified, "VerifiedJinaProvider", lambda *args, **kwargs: p)

    def succeeded(_provider, _cases, *, evidence):
        """Substitute only the slow gates; this test exercises CLI cleanup and reporting."""
        evidence["description"] = {}

    async def database(*_args):
        """Report a successful DB gate without starting external resources."""
        return {"passed": True}

    def failed_unload():
        """Inject a release failure after successful stages."""
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(protocol, "qualify", succeeded)
    monkeypatch.setattr(module, "database_roundtrip", database)
    output = tmp_path / "evidence"
    previous_umask = os.umask(0o077)
    # Restore before the real provider fixture performs its own teardown.
    try:
        with monkeypatch.context() as cleanup:
            cleanup.setattr(p, "unload", failed_unload)
            code = module.main([
                "--registry", str(registry.root), "--provider", "verified", "--device", "cpu",
                "--output-dir", str(output), "--arangod", "/bin/true",
            ])
        report = json.loads((output / "report.json").read_text())
        assert code == 1 and report["passed"] is False and report["stage"] == "cleanup"
        assert report["database"]["passed"] is True
        assert report["cleanup_error"]["message"] == "injected cleanup failure"
    finally:
        os.umask(previous_umask)
