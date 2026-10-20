"""Opt-in qualification protocol for an already approved, registered Jina model.

This is a small deterministic smoke suite, not a retrieval-quality benchmark or
authorization to activate production pairing. Thresholds are fixed before runs.
The caller owns the provider and must unload it even after a failed assertion.
"""

from dataclasses import replace
import hashlib
import json
import time
import unicodedata

import numpy as np
from safetensors import safe_open
import torch

from .identity import ProviderReceipt, _exact_fields, _unique_object
from .jina_v4 import EMBEDDING_DIM, InputTooLargeError
from .verified_jina import PROMPTS, ReceiptMismatchError, _versions

PROTOCOL = "hades-verified-jina-qualification-v2"
MIN_REPEAT_COSINE = 0.99999
MIN_RELOAD_COSINE = 0.9999
MAX_NORM_ERROR = 0.001


def require(condition, message):
    """Fail qualification explicitly, including when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def load_cases(path):
    """Read a bounded, unambiguous suite and return its exact file-content digest."""
    with path.open("rb") as source:
        encoded = source.read(128 * 1024 + 1)
    require(len(encoded) <= 128 * 1024, "qualification cases exceed 128 KiB")
    cases = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    _exact_fields(cases, {"format", "documents", "queries", "tokenizer_samples"})
    require(cases["format"] == "hades-jina-qualification-cases-v1", "unsupported suite")
    ids = set()
    for family, fields in (("documents", {"id", "text"}),
                           ("queries", {"id", "text", "expected_document"})):
        rows = cases[family]
        require(isinstance(rows, list) and 1 <= len(rows) <= 32, "invalid suite size")
        seen = set()
        for row in rows:
            _exact_fields(row, fields)
            require(isinstance(row["id"], str) and 0 < len(row["id"]) <= 64
                    and row["id"].isascii() and row["id"].replace("-", "").isalnum(),
                    "invalid case identifier")
            require(row["id"] not in seen, "duplicate case identifier")
            seen.add(row["id"])
            require(isinstance(row["text"], str) and row["text"].strip()
                    and len(row["text"].encode("utf-8")) <= 16384, "invalid case text")
            if family == "queries":
                require(row["expected_document"] in ids, "query names an unknown document")
        if family == "documents":
            ids = seen
    samples = cases["tokenizer_samples"]
    require(isinstance(samples, list) and 1 <= len(samples) <= 16
            and all(isinstance(s, str) and 0 < len(s.encode("utf-8")) <= 4096 for s in samples),
            "invalid tokenizer samples")
    return cases, hashlib.sha256(encoded).hexdigest()


def checked_vectors(response, expected, model, count):
    """Reject incomplete, misidentified, nonfinite, or unnormalized provider output."""
    ProviderReceipt.from_dict(response["receipt"]).check_matches(expected)
    require(response["model"] == model, "response model changed")
    rows = response["data"]
    require(len(rows) == count and [row["index"] for row in rows] == list(range(count)),
            "incomplete or reordered embedding output")
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float64)
    require(vectors.shape == (count, EMBEDDING_DIM) and np.isfinite(vectors).all(),
            "invalid vector shape or values")
    require(np.all(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) <= MAX_NORM_ERROR),
            "output violates L2 normalization")
    return vectors


def cosine(one, two):
    """Compare repeated observations in float64 without assuming exact float equality."""
    return float(np.clip(np.dot(one, two) / (np.linalg.norm(one) * np.linalg.norm(two)), -1.0, 1.0))


def rank(documents, passages, queries, vectors):
    """Require each independently declared expected document to rank first."""
    require(passages.shape == (len(documents), EMBEDDING_DIM)
            and vectors.shape == (len(queries), EMBEDDING_DIM), "incomplete ranking inputs")
    scores = (vectors @ passages.T) / (
        np.linalg.norm(vectors, axis=1)[:, None] * np.linalg.norm(passages, axis=1)[None, :]
    )
    hits = []
    for query, row in zip(queries, scores):
        order = sorted(range(len(documents)), key=lambda i: (-float(row[i]), documents[i]["id"]))
        expected = next(i for i, doc in enumerate(documents) if doc["id"] == query["expected_document"])
        hits.append({"query": query["id"], "expected": query["expected_document"],
                     "first": documents[order[0]]["id"], "rank": order.index(expected) + 1,
                     "score": float(row[expected])})
    require(all(hit["rank"] == 1 for hit in hits), f"retrieval smoke criterion failed: {hits}")
    return hits


def tokenizer_observation(tokenizer, samples):
    """Check the declared normalization against an independent Unicode oracle."""
    normalizer = json.loads(tokenizer.backend_tokenizer.to_str())["normalizer"]
    require(normalizer is None or normalizer == {"type": "NFC"},
            "qualification supports only identity or declared NFC normalization")
    normalization = "identity" if normalizer is None else "NFC"
    observations = []
    for text in samples:
        ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        wanted = text if normalizer is None else unicodedata.normalize("NFC", text)
        require(decoded == wanted, "tokenizer violated declared Unicode normalization or whitespace")
        observations.append({"text": text, "decoded": decoded, "ids": ids})
    return {"normalization": normalization, "samples": observations}


def boundary_texts(tokenizer, task, ceiling):
    """Construct and verify exact fitting/one-over inputs without a token estimate."""
    def count(text):
        """Count exactly the same prefix and special-token policy as the adapter."""
        return len(tokenizer(PROMPTS[task] + text, add_special_tokens=True,
                             padding=False, truncation=False, split_special_tokens=False)["input_ids"])
    # This fixed unit is checked, not assumed, to add one token per repetition.
    first = count("hello ")
    require(count("hello " * 2) == first + 1 and ceiling >= first, "unsupported boundary fixture")
    fitting = "hello " * (ceiling - first + 1)
    oversized = fitting + "hello "
    require(count(fitting) == ceiling and count(oversized) == ceiling + 1,
            "boundary fixture does not reach exact serving limits")
    return fitting, oversized


def check_stored_plan(provider, text, plan):
    """Independently recount persisted byte slices under the same actual provider load."""
    raw = text.encode("utf-8")
    require(plan["source_sha256"] == hashlib.sha256(raw).hexdigest(), "planning source changed")
    expected = ProviderReceipt.from_dict(plan["receipt"])
    with provider._operation(expected):
        tokenizer = provider._backend._tokenizer

        def count(value, rendered=False):
            """Use direct tokenizer calls, without the planner's search or accounting helpers."""
            return len(tokenizer((PROMPTS["retrieval.passage"] if rendered else "") + value,
                                 add_special_tokens=rendered, padding=False, truncation=False,
                                 split_special_tokens=False)["input_ids"])

        whole = count(text, True)
        require(count(text) > plan["passage_tokens"], "long fixture did not exceed the passage budget")
        require(len(plan["spans"]) > 1, "long fixture was not split")
        previous = None
        saw_overlap = False
        for span in plan["spans"]:
            start, end = span["start_byte"], span["end_byte"]
            value = raw[start:end].decode("utf-8")
            require(value.strip() and count(value) == span["body_tokens"] <= plan["passage_tokens"],
                    "stored body count differs from tokenizer")
            require(0 < count(value, True) == span["input_tokens"] <= plan["max_input_tokens"],
                    "stored rendered count differs from tokenizer")
            if previous is None:
                require(start == 0 and span["overlap_tokens"] == 0, "invalid first passage")
            else:
                require(previous["start_byte"] < start <= previous["end_byte"] < end,
                        "stored passages do not cover source in order")
                overlap = raw[start:previous["end_byte"]].decode("utf-8")
                require(count(overlap) == span["overlap_tokens"] <= plan["overlap_tokens"],
                        "stored overlap differs from tokenizer")
                saw_overlap |= start < previous["end_byte"]
            previous = span
        require(previous["end_byte"] == len(raw) and saw_overlap,
                "long fixture did not preserve complete coverage with overlap")
    return {"passed": True, "whole_input_tokens": whole, "checked_passages": len(plan["spans"])}


def verify_adapters(provider):
    """Compare every registered LoRA checkpoint tensor with the actual loaded tensor."""
    backend = provider._backend
    checkpoint = backend._snapshot.root / "adapters/adapter_model.safetensors"
    state = backend._model.state_dict()
    checked = 0
    with safe_open(checkpoint, framework="pt") as source:
        for key in source.keys():
            candidates = [key, key.replace(".lora_A.", ".lora_A.default.")
                          .replace(".lora_B.", ".lora_B.default.")]
            actual = next((state[name] for name in candidates if name in state), None)
            require(actual is not None, f"missing loaded adapter tensor: {key}")
            wanted = source.get_tensor(key).to(actual.dtype)
            require(torch.equal(actual.detach().cpu(), wanted), f"adapter tensor differs: {key}")
            checked += 1
    require(checked > 0, "adapter checkpoint contains no tensors")
    provider.describe()  # Recheck artifacts/behavior after the independent inspection.
    return checked


def qualify(provider, cases, *, adapter_check=verify_adapters, evidence=None):
    """Measure fixed provider gates, retaining a fresh verified load for the DB round trip."""
    device = provider._backend.device
    gpu = device.startswith("cuda")
    if gpu:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    description = provider.describe()
    load_seconds = time.monotonic() - started
    expected = ProviderReceipt.from_dict(description["receipt"])
    model = description["model"]["id"]
    report = evidence if evidence is not None else {}
    report.update({"protocol": PROTOCOL, "criteria": {
        "all_expected_documents_rank_first": True, "max_norm_error": MAX_NORM_ERROR,
        "minimum_repeat_cosine": MIN_REPEAT_COSINE, "minimum_reload_cosine": MIN_RELOAD_COSINE,
    }, "runtime": {"device": device, "libraries": _versions(), "cuda_runtime": torch.version.cuda},
        "initial_load_seconds": load_seconds, "initial_receipt": expected.to_dict(),
        "description": description})
    if gpu:
        report["runtime"]["gpu"] = torch.cuda.get_device_name(device)
    report["adapter_tensors_verified"] = adapter_check(provider)
    tokenizer = provider._backend._tokenizer
    tokens = tokenizer_observation(tokenizer, cases["tokenizer_samples"])
    report["tokenizer"] = tokens
    timings = []
    report["operations"] = timings
    report["max_observed_norm_error"] = 0.0

    def embed(texts, task, receipt=expected):
        """Time a fully synchronized, receipted operation and validate every vector."""
        tokenizer = provider._backend._tokenizer
        expected_tokens = sum(len(tokenizer(
            PROMPTS[task] + text, add_special_tokens=True, padding=False,
            truncation=False, split_special_tokens=False)["input_ids"]) for text in texts)
        if gpu:
            torch.cuda.synchronize(device)
        started = time.monotonic()
        response = provider.embed(model=model, texts=texts, task=task, expected=receipt)
        if gpu:
            torch.cuda.synchronize(device)
        require(response["usage"]["total_tokens"] == expected_tokens,
                "reported token usage disagrees with exact prompt-inclusive count")
        timings.append({"task": task, "inputs": len(texts), "tokens": response["usage"]["total_tokens"],
                        "seconds": time.monotonic() - started})
        vectors = checked_vectors(response, receipt, model, len(texts))
        report["max_observed_norm_error"] = max(
            report["max_observed_norm_error"],
            float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0))),
        )
        return vectors

    passages = embed([doc["text"] for doc in cases["documents"]], "retrieval.passage")
    queries = embed([q["text"] for q in cases["queries"]], "retrieval.query")
    report["retrieval"] = rank(cases["documents"], passages, cases["queries"], queries)
    repeat = embed([cases["queries"][0]["text"]], "retrieval.query")[0]
    report["repeat_cosine"] = cosine(queries[0], repeat)
    require(report["repeat_cosine"] >= MIN_REPEAT_COSINE, "repeatability criterion failed")
    forwards = []

    def observe_forward(_module, _args, kwargs):
        """Observe real input IDs without replacing weights or numerical code."""
        forwards.append(kwargs["input_ids"].detach().cpu().tolist()[0])

    hook = provider._backend._model.register_forward_pre_hook(observe_forward, with_kwargs=True)
    try:
        boundaries = []
        for task in PROMPTS:
            ceiling = description["capabilities"]["max_input_tokens"]
            fitting, oversized = boundary_texts(tokenizer, task, ceiling)
            before = len(forwards)
            embed([fitting], task)
            wanted = tokenizer(PROMPTS[task] + fitting, add_special_tokens=True,
                               padding=False, truncation=False, split_special_tokens=False)["input_ids"]
            require(len(forwards) == before + 1 and forwards[-1] == wanted,
                    "fitting input was truncated or retokenized before the real forward pass")
            before = len(forwards)
            try:
                embed(["short input", oversized], task)
            except InputTooLargeError as error:
                require(error.index == 1 and error.n_tokens == ceiling + 1
                        and error.max_tokens == ceiling and len(forwards) == before,
                        "overflow did not refuse all inputs before inference")
            else:
                raise ValueError("oversized input was accepted")
            boundaries.append({"task": task, "accepted_tokens": ceiling,
                               "observed_forward_tokens": len(forwards[-1]),
                               "rejected_tokens": ceiling + 1, "overflow_forward_calls": len(forwards) - before})
        report["boundaries"] = boundaries
        before = len(forwards)
        for wrong in (replace(expected, artifact_md5="0" * 32), replace(expected, space_id="0" * 64),
                      replace(expected, serving_generation="other-load")):
            try:
                embed(["short input"], "retrieval.query", wrong)
            except ReceiptMismatchError:
                pass
            else:
                raise ValueError("incompatible receipt was accepted")
        require(len(forwards) == before, "wrong receipt reached model inference")
        report["mismatched_receipts_rejected"] = 3
    finally:
        hook.remove()
    provider.unload()
    started = time.monotonic()
    reloaded = provider.describe()
    report["reload_seconds"] = time.monotonic() - started
    current = ProviderReceipt.from_dict(reloaded["receipt"])
    require(current.artifact_md5 == expected.artifact_md5 and current.space_id == expected.space_id
            and current.serving_generation != expected.serving_generation, "reload changed stable identity")
    require(tokenizer_observation(provider._backend._tokenizer, cases["tokenizer_samples"]) == tokens,
            "tokenizer IDs changed on reload")
    try:
        embed(["short input"], "retrieval.query")
    except ReceiptMismatchError:
        pass
    else:
        raise ValueError("pre-reload receipt was accepted")
    after = embed([cases["queries"][0]["text"]], "retrieval.query", current)[0]
    report["reload_cosine"] = cosine(queries[0], after)
    require(report["reload_cosine"] >= MIN_RELOAD_COSINE, "reload repeatability criterion failed")
    report["operations"] = timings
    report["description"] = reloaded
    report["initial_receipt"] = expected.to_dict()
    if gpu:
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return report
