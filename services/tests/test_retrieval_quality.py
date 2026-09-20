"""Independent examples for the offline evaluator; no model loading."""
import importlib.util
from pathlib import Path
import copy
import math
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("retrieval_eval", Path(__file__).parents[2] / "scripts/evaluate_retrieval_quality.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def fixture():
    dataset = {"documents":[{"id":key,"source":key} for key in ["relevant","first","last"]],
               "queries":[{"id":"q","text":"question","relevance":{"relevant":3}}]}
    return dataset, np.array([[0.,1.],[1.,0.],[-1.,0.]]), np.array([[1.,0.]])


def test_known_rank_two_metrics():
    report = evaluator.score(*fixture())
    for metrics in report["aggregate"].values():
        assert metrics["recall_at_5"] == 1
        assert metrics["mrr_at_10"] == 0.5
        assert metrics["ndcg_at_10"] == pytest.approx(1/math.log2(3))


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_invalid_vectors_are_rejected(bad):
    dataset, documents, queries = fixture()
    queries[:] = bad
    with pytest.raises(ValueError,match="finite and nonzero"):
        evaluator.score(dataset,documents,queries)


def test_unjudged_or_unknown_positive_cannot_report_a_score():
    dataset, documents, queries = fixture()
    for relevance in [{}, {"missing":3}, {"relevant":0}, {"relevant":-1}]:
        case = copy.deepcopy(dataset)
        case["queries"][0]["relevance"] = relevance
        with pytest.raises(ValueError,match="valid judgments"):
            evaluator.score(case,documents,queries)


def test_vector_artifact_digest_covers_exact_little_endian_float32(tmp_path):
    import hashlib
    import struct
    path = tmp_path / "vectors.f32"
    digest = evaluator.write_vector_artifact(path, np.array([[1.25, -2.5]], dtype=">f8"))
    expected = struct.pack("<ff", 1.25, -2.5)
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
