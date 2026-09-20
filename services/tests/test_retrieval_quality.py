"""Independent examples for the offline evaluator; no model loading."""
import importlib.util
from pathlib import Path
import copy
import math
import json
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


def test_strict_policy_withholds_unjudged_rankings_without_compressing_ranks():
    dataset, documents, queries = fixture()
    dataset["scoring_policy"] = "complete_top10"
    report = evaluator.score(dataset, documents, queries)
    assert report["fully_judged_queries"] == 0
    assert report["aggregate_status"] == "withheld"
    for row in report["per_query"]:
        assert row["ranking"] == ["first", "relevant", "last"]
        assert row["score_status"] == "unjudged_top10"
        assert row["judgment_coverage"] == {
            "judged_results": 1, "returned_results": 3, "fraction": 1/3,
            "unjudged": ["first", "last"],
        }
        assert row["mrr_at_10"] is None
    assert all(value is None for metrics in report["aggregate"].values() for value in metrics.values())


def test_one_incomplete_query_withholds_aggregate_but_preserves_complete_query():
    dataset, documents, queries = fixture()
    dataset["scoring_policy"] = "complete_top10"
    dataset["queries"][0]["relevance"].update(first=0, last=0)
    dataset["queries"].append({"id":"unjudged", "text":"another", "relevance":{}})
    report = evaluator.score(dataset, documents, np.concatenate([queries, queries]))
    assert report["fully_judged_queries"] == 1
    assert report["total_queries"] == 2
    assert report["aggregate_status"] == "withheld"
    assert all(value is None for metrics in report["aggregate"].values() for value in metrics.values())
    for row in report["per_query"]:
        if row["query"] == "q":
            assert row["mrr_at_10"] == 0.5
            assert row["ndcg_at_10"] == pytest.approx(1/math.log2(3))
        else:
            assert row["mrr_at_10"] is None


def test_complete_strict_judgments_match_hand_calculated_metrics():
    dataset, documents, queries = fixture()
    dataset["scoring_policy"] = "complete_top10"
    dataset["queries"][0]["relevance"].update(first=0, last=0)
    report = evaluator.score(dataset, documents, queries)
    assert report["aggregate_status"] == "scored"
    assert report["metric_scope"] == "judged_pool"
    for metrics in report["aggregate"].values():
        assert metrics["recall_at_5"] == 1
        assert metrics["mrr_at_10"] == 0.5
        assert metrics["ndcg_at_10"] == pytest.approx(1/math.log2(3))


def test_strict_no_evidence_case_is_not_silently_scored_as_zero():
    dataset, documents, queries = fixture()
    dataset["scoring_policy"] = "complete_top10"
    dataset["queries"][0]["relevance"] = {d["id"]:0 for d in dataset["documents"]}
    report = evaluator.score(dataset, documents, queries)
    assert report["fully_judged_queries"] == 1
    assert report["aggregate_status"] == "withheld"
    assert all(row["score_status"] == "no_positive_judgments" for row in report["per_query"])


def test_unknown_policy_and_duplicate_queries_are_rejected():
    dataset, documents, queries = fixture()
    dataset["scoring_policy"] = "complete_top_10_typo"
    with pytest.raises(ValueError, match="unknown scoring policy"):
        evaluator.score(dataset, documents, queries)
    dataset["scoring_policy"] = "complete_top10"
    dataset["queries"].append(copy.deepcopy(dataset["queries"][0]))
    with pytest.raises(ValueError, match="unique query"):
        evaluator.score(dataset, documents, np.concatenate([queries, queries]))


def test_historical_seed_artifact_reproduces_unchanged():
    root = Path(__file__).parents[2] / "evaluation/retrieval"
    dataset = json.loads((root / "repository-v2.json").read_text())
    vectors = np.fromfile(root / "results-v2/vectors.f32", dtype="<f4").reshape(-1, 2048)
    count = len(dataset["documents"])
    expected = json.loads((root / "results-v2/metrics.json").read_text())
    assert evaluator.score(dataset, vectors[:count], vectors[count:]) == expected


def test_top_ten_coverage_and_judged_pool_denominator_are_distinct():
    ids = [f"d{i:02}" for i in range(12)]
    labels = {key:0 for key in ids[:10]}
    labels.update(d01=3, d11=2)
    dataset = {"scoring_policy":"complete_top10",
        "documents":[{"id":key,"source":key} for key in ids],
        "queries":[{"id":"q","relevance":labels}]}
    angles = np.arange(12) / 10
    documents = np.column_stack([np.cos(angles), np.sin(angles)])
    report = evaluator.score(dataset, documents, np.array([[1., 0.]]))
    assert report["aggregate_status"] == "scored"
    for row in report["per_query"]:
        assert row["ranking"] == ids[:10]
        assert row["judgment_coverage"]["fraction"] == 1
        assert row["recall_at_5"] == 0.5
        assert row["ndcg_at_10"] == pytest.approx((7/math.log2(3)) / (7 + 3/math.log2(3)))
