#!/usr/bin/env python3
"""Offline, CPU-only retrieval evaluation; never contacts HADES services."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

PROFILES = {
    "code_search": {"task": "code", "document_prompt": "passage", "query_prompt": "passage"},
    "document_research": {"task": "retrieval", "document_prompt": "passage", "query_prompt": "query"},
}
MAX_TOKENS = 2048


def encoding_plan(dataset):
    """Resolve one adapter space and distinct document/query prompts per corpus."""
    name = dataset.get("embedding_profile", "code_search")
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError("unknown embedding profile")
    if dataset.get("workload", name) != name:
        raise ValueError("workload and embedding profile differ")
    profile = dict(PROFILES[name], name=name)
    inputs = []
    for key, prompt in [("documents", profile["document_prompt"]), ("queries", profile["query_prompt"])]:
        if not isinstance(dataset.get(key), list) or not dataset[key]:
            raise ValueError("encoding needs nonempty documents and queries")
        for item in dataset[key]:
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError("encoding inputs require nonempty text")
            inputs.append((item["text"], prompt))
    return profile, inputs


def preflight_inputs(processor, inputs):
    """Count actual prefixed processor inputs without truncation before inference."""
    limit = min(MAX_TOKENS, processor.text_max_length)
    counts = []
    for index, (text, prompt) in enumerate(inputs):
        prefix = {"passage": "Passage", "query": "Query"}[prompt]
        encoded = processor(text=[f"{prefix}: {text}"], padding=False, truncation=False)
        count = len(encoded["input_ids"][0])
        if not 0 < count <= limit:
            raise ValueError(f"input {index} has {count} tokens; allowed range is 1..{limit}")
        counts.append(count)
    return {"effective_max_tokens": limit, "input_tokens": counts,
        "prefixes": "Jina v4 Passage: / Query: ", "truncation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--vectors", type=Path, help="rescore frozen vectors without loading a model")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/retrieval/repository-v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    evaluator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dataset = json.loads(dataset_bytes)
    profile, inputs = encoding_plan(dataset)
    validate_judgments(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.vectors:
        import numpy as np
        count = len(dataset["documents"])
        provenance = json.loads(args.vectors.with_name("provenance.json").read_text())
        if provenance["dataset_sha256"] != hashlib.sha256(dataset_bytes).hexdigest():
            raise ValueError("frozen vectors belong to a different dataset version")
        if provenance["vectors_sha256"] != hashlib.sha256(args.vectors.read_bytes()).hexdigest():
            raise ValueError("frozen vector artifact hash does not match provenance")
        vectors = np.fromfile(args.vectors, dtype="<f4").reshape(count + len(dataset["queries"]), 2048)
        report = score(dataset, vectors[:count], vectors[count:])
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["aggregate"], indent=2))
        return
    assert args.model.is_dir(), "model must already exist locally"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    for key, value in {"CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "HF_HOME":str(args.output / "hf-cache"), "HF_MODULES_CACHE":str(args.output / "modules"), "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "XDG_CACHE_HOME":str(args.output / "cache"), "TOKENIZERS_PARALLELISM":"false"}.items():
        os.environ[key] = value
    import numpy as np
    import torch
    from transformers import AutoModel
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    docs, questions = dataset["documents"], dataset["queries"]
    started = time.monotonic()
    model = AutoModel.from_pretrained(str(args.model.resolve()), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    token_preflight = preflight_inputs(model.processor, inputs)
    vectors = []
    for i, (text, prompt) in enumerate(inputs):
        result = model.encode_text([text], task=profile["task"], prompt_name=prompt, max_length=MAX_TOKENS,
            batch_size=1, return_numpy=False)
        vector = torch.stack(result).detach().cpu().float().numpy()[0]
        assert vector.shape == (2048,) and np.isfinite(vector).all()
        vectors.append(vector)
        print(f"Encoded {i+1}/{len(docs)+len(questions)}", flush=True)
    vectors = np.stack(vectors)
    np.savez_compressed(args.output / "vectors.npz", documents=vectors[:len(docs)], queries=vectors[len(docs):])
    vectors_sha256 = write_vector_artifact(args.output / "vectors.f32", vectors)
    manifest = {}
    for path in sorted(args.model.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".safetensors"}:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b""): digest.update(block)
            manifest[str(path.relative_to(args.model))] = digest.hexdigest()
    report = score(dataset, vectors[:len(docs)], vectors[len(docs):])
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(report,indent=2)+"\n")
    provenance = {"vectors_sha256":vectors_sha256,
        "evaluator_sha256":evaluator_sha256,
        "metrics_sha256":hashlib.sha256(metrics_path.read_bytes()).hexdigest(), "dataset_sha256":hashlib.sha256(dataset_bytes).hexdigest(),
        "model_files":manifest, "embedding_profile":profile, "token_preflight":token_preflight,
        "device":"cpu", "dtype":"bfloat16",
        "max_tokens":MAX_TOKENS, "dependencies":{name:importlib.metadata.version(name) for name in ["torch","transformers","peft","numpy","tokenizers"]},
        "seconds":time.monotonic()-started, "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":["Author-created seed with unjudged documents, not a production quality certification.","CPU bfloat16 differs from the active GPU precision and runtime.","Graph baseline uses source-file membership, not learned graph weights."]}
    (args.output / "provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(report["aggregate"],indent=2),flush=True)


def write_vector_artifact(path, vectors):
    """Write canonical little-endian float32 and bind provenance to those bytes."""
    vectors.astype("<f4").tofile(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_judgments(dataset):
    """Reject malformed evaluation identities and labels before model loading."""
    policy = dataset.get("scoring_policy", "legacy_seed")
    if policy not in ("legacy_seed", "complete_top10"):
        raise ValueError("unknown scoring policy")
    docs = dataset["documents"]
    ids = [d["id"] for d in docs]
    if not docs or len(ids) != len(set(ids)) or not dataset["queries"]:
        raise ValueError("evaluation requires unique documents and nonempty queries")
    for question in dataset["queries"]:
        labels = question["relevance"]
        if (not isinstance(labels, dict) or not set(labels).issubset(ids)
            or any(type(v) is not int or not 0 <= v <= 3 for v in labels.values())
            or (policy != "complete_top10" and not any(v > 0 for v in labels.values()))):
            raise ValueError("queries need valid judgments; legacy seed queries also need a relevant document")
    query_ids = [q["id"] for q in dataset["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluation requires unique query IDs")
    return policy, docs, ids, query_ids


def score(dataset, document_vectors, query_vectors):
    import numpy as np
    policy, docs, ids, query_ids = validate_judgments(dataset)
    strict = policy == "complete_top10"
    document_vectors = np.asarray(document_vectors, dtype=np.float64)
    query_vectors = np.asarray(query_vectors, dtype=np.float64)
    if (document_vectors.ndim != 2 or query_vectors.ndim != 2
        or len(document_vectors) != len(docs) or len(query_vectors) != len(dataset["queries"])
        or document_vectors.shape[1] != query_vectors.shape[1] or document_vectors.shape[1] == 0):
        raise ValueError("vector matrices do not match the dataset")
    for matrix in (document_vectors, query_vectors):
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("vectors must be finite and nonzero")
    paths = sorted({d["source"] for d in docs})
    structural = np.eye(len(paths))[[paths.index(d["source"]) for d in docs]]
    document_vectors = document_vectors.astype(np.float64)
    document_vectors /= np.linalg.norm(document_vectors, axis=1, keepdims=True)
    rows = []
    for question, vector in zip(dataset["queries"],query_vectors,strict=True):
        vector = vector.astype(np.float64); vector /= np.linalg.norm(vector)
        scores = document_vectors @ vector
        ranked = sorted(range(len(docs)),key=lambda i:(-scores[i],ids[i]))[:10]
        centroid = structural[ranked[:3]].mean(axis=0)
        graph_scores = structural @ centroid / np.linalg.norm(centroid)
        fused = sorted(ranked,key=lambda i:(-(0.7*scores[i]+0.3*graph_scores[i]),ids[i]))
        for method, order in [("vector",ranked),("file_membership_graph",fused)]:
            relevance = question["relevance"]
            missing = [ids[i] for i in order if ids[i] not in relevance]
            coverage = {
                "judged_results": len(order) - len(missing),
                "returned_results": len(order),
                "fraction": (len(order) - len(missing)) / len(order),
                "unjudged": missing,
            }
            if strict and (missing or not any(v > 0 for v in relevance.values())):
                rows.append({"query":question["id"], "method":method,
                    "ranking":[ids[i] for i in order], "judgment_coverage":coverage,
                    "score_status":"unjudged_top10" if missing else "no_positive_judgments",
                    "recall_at_5":None, "mrr_at_10":None, "ndcg_at_10":None})
                continue
            labels = [relevance.get(ids[i],0) for i in order]
            ideal = sorted(relevance.values(),reverse=True)[:10]
            dcg = lambda values: sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values))
            rows.append({"query":question["id"],"method":method,"ranking":[ids[i] for i in order],
                "recall_at_5":sum(v>0 for v in labels[:5])/sum(v>0 for v in relevance.values()),
                "mrr_at_10":next((1/(i+1) for i,v in enumerate(labels) if v>0),0),
                "ndcg_at_10":dcg(labels)/dcg(ideal)})
            if strict:
                rows[-1].update(judgment_coverage=coverage, score_status="scored")
    aggregate = {}
    complete = not strict or all(row["score_status"] == "scored" for row in rows)
    for method in ["vector","file_membership_graph"]:
        selected = [r for r in rows if r["method"]==method]
        aggregate[method]={metric:sum(r[metric] for r in selected)/len(selected) if complete else None
            for metric in ["recall_at_5","mrr_at_10","ndcg_at_10"]}
    report = {"aggregate":aggregate,"per_query":rows}
    if strict:
        fully_judged = sum(all(not row["judgment_coverage"]["unjudged"]
            for row in rows if row["query"] == query) for query in query_ids)
        report.update(scoring_policy=policy, metric_scope="judged_pool",
            aggregate_status="scored" if complete else "withheld",
            fully_judged_queries=fully_judged, total_queries=len(query_ids))
    return report


if __name__ == "__main__":
    main()
