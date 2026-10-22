"""Fail-closed qualification evidence using CPU fixtures, never downloaded weights."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import types

import pytest

from test_model_registration import registry, source  # noqa: F401
from test_verified_jina_provider import provider  # noqa: F401
from test_verified_code_provider import code_provider  # noqa: F401
from embedding.code_qualification import qualify_workloads, recheck
from embedding.qualification import load_cases

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    """Import optional tools without requiring global installation of their script directory."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_workloads_run_all_gates_and_keep_one_fresh_load(code_provider):
    p, state = code_provider
    suite = {"documents": [{"id": "one", "text": "one"}],
             "queries": [{"id": "one", "text": "one", "expected_document": "one"}],
             "tokenizer_samples": ["one"]}
    evidence = {}
    observed = qualify_workloads(p, {kind: suite for kind in ("documents", "code")},
                                 evidence, adapter_check=lambda _: 1)
    assert observed == p.describe() and len(state.loads) == 3
    assert evidence["cross_workload_receipts_rejected"] == 3
    for kind, ceiling, adapter in [("documents", 16, "retrieval"), ("code", 8, "code")]:
        result = evidence[kind]
        assert result["retrieval"][0]["rank"] == 1 and result["reload_cosine"] >= 0.9999
        assert all(row["accepted_tokens"] == ceiling and row["overflow_forward_calls"] == 0
                   and row["adapter"] == adapter for row in result["boundaries"])


def test_code_suite_is_fixed_and_uses_distinct_languages():
    cases, digest = load_cases(ROOT / "services/tests/fixtures/jina_code_qualification.json")
    assert len(cases["documents"]) == 8 and len(cases["queries"]) == 6 and len(digest) == 64
    assert {"rust-parse", "sql-total", "debounce", "binary-search"} <= {row["id"] for row in cases["documents"]}


def test_rust_recount_rejects_wrong_code_counts_and_borrowed_receipts(code_provider):
    from embedding.identity import ProviderReceipt
    p, _ = code_provider
    context = {"planning_texts": {kind: "one é two three " * 10 for kind in ("documents", "code")}}
    description = p.describe()
    result = {}
    for kind, task in [("documents", "retrieval.passage"), ("code", "code")]:
        expected = ProviderReceipt.from_dict(description["workloads"][kind]["receipt"])
        plan = p.plan(model=p.model_name, text=context["planning_texts"][kind], task=task,
                      expected=expected, passage_tokens=6, overlap_tokens=1, max_input_tokens=16)
        result[kind] = {"planning": plan}
    assert all(row["passed"] for row in recheck(p, context, result).values())
    bad = json.loads(json.dumps(result))
    bad["code"]["planning"]["spans"][0]["input_tokens"] += 1
    with pytest.raises(ValueError, match="rendered count"):
        recheck(p, context, bad)
    bad = json.loads(json.dumps(result))
    bad["code"]["planning"]["receipt"] = result["documents"]["planning"]["receipt"]
    with pytest.raises(ValueError, match="receipt changed"):
        recheck(p, context, bad)


def test_cli_refuses_cuda_fallback_and_evidence_overwrite(tmp_path, monkeypatch):
    import torch
    module = script("qualify_verified_code")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "evidence"
    args = ["--registry", str(tmp_path / "absent"), "--provider", "fixture", "--device", "cuda:0",
            "--output-dir", str(output), "--max-input-tokens", "1024", "--code-max-input-tokens", "512"]
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


@pytest.mark.parametrize("fail_provider", [False, True])
def test_cli_cleanup_failure_never_publishes_success(tmp_path, provider, monkeypatch, fail_provider):
    import embedding.code_qualification as protocol
    import embedding.verified_code as adapter
    import embedding.registry as registry_module
    p, _ = provider
    module = script("qualify_verified_code")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(registry_module.ModelRegistry, "load", lambda *_: p._registration)
    def fail_cleanup():
        raise RuntimeError("cleanup fixture")
    fake = types.SimpleNamespace(_backend=types.SimpleNamespace(device="cpu"), unload=fail_cleanup)
    monkeypatch.setattr(adapter, "VerifiedJinaCodeProvider", lambda *_, **__: fake)
    def qualify(*_):
        if fail_provider:
            raise RuntimeError("provider fixture")
        return {}
    async def transport(*_):
        return {"passed": True}
    monkeypatch.setattr(protocol, "qualify_workloads", qualify)
    monkeypatch.setattr(module, "rust_roundtrip", transport)
    output = tmp_path / "output"
    previous_umask = os.umask(0o077)
    try:
        assert module.main(["--registry", str(tmp_path), "--provider", "fixture", "--device", "cpu",
                            "--output-dir", str(output), "--max-input-tokens", "1024",
                            "--code-max-input-tokens", "512"]) == 1
    finally:
        os.umask(previous_umask)
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False and report["cleanup_error"] == "cleanup fixture"
    if fail_provider:
        assert report["error"]["message"] == "provider fixture"


@pytest.mark.parametrize("failure", ["child", "recount"])
def test_private_transport_unloads_after_child_or_recount_failure(provider, tmp_path, monkeypatch, failure):
    """Exercise actual server ownership while replacing only the expensive Rust child."""
    p, _ = provider
    p.describe()
    module = script("qualification_transport")
    async def child(*_args, **kwargs):
        if failure != "child":
            Path(kwargs["env"]["FIXTURE_REPORT"]).write_text('{"passed": true}')
        class Child:
            returncode = 1 if failure == "child" else 0
            async def wait(self):
                return self.returncode
        return Child()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", child)
    def recount(*_):
        raise ValueError("recount fixture")
    with pytest.raises((RuntimeError, ValueError), match="Rust qualification failed|recount fixture"):
        asyncio.run(module.run_rust_contract(p, {}, tmp_path, repo=ROOT, target="fixture",
                                             test="fixture", environment="FIXTURE", recheck=recount))
    assert not p.is_loaded and not list(p._registration.sessions.iterdir())
