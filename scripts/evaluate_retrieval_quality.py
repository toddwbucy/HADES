#!/usr/bin/env python3
"""Offline, CPU-only code-retrieval seed evaluation; never contacts HADES services."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--vectors", type=Path, help="rescore frozen vectors without loading a model")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/retrieval/repository-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.vectors:
        import numpy as np
        dataset = json.loads(args.dataset.read_text())
        count = len(dataset["documents"])
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
    dataset = json.loads(args.dataset.read_text())
    docs, questions = dataset["documents"], dataset["queries"]
    started = time.monotonic()
    model = AutoModel.from_pretrained(str(args.model.resolve()), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    vectors = []
    for i, text in enumerate([d["text"] for d in docs] + [q["text"] for q in questions]):
        result = model.encode_text([text], task="code", prompt_name="passage", max_length=2048,
            batch_size=1, return_numpy=False)
        vector = torch.stack(result).detach().cpu().float().numpy()[0]
        assert vector.shape == (2048,) and np.isfinite(vector).all()
        vectors.append(vector)
        print(f"Encoded {i+1}/{len(docs)+len(questions)}", flush=True)
    vectors = np.stack(vectors)
    np.savez_compressed(args.output / "vectors.npz", documents=vectors[:len(docs)], queries=vectors[len(docs):])
    vectors.astype("<f4").tofile(args.output / "vectors.f32")
    manifest = {}
    for path in sorted(args.model.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".safetensors"}:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b""): digest.update(block)
            manifest[str(path.relative_to(args.model))] = digest.hexdigest()
    provenance = {"dataset_sha256":hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model_files":manifest, "task":"code", "prompt":"passage", "device":"cpu", "dtype":"bfloat16",
        "max_tokens":2048, "dependencies":{name:importlib.metadata.version(name) for name in ["torch","transformers","peft","numpy","tokenizers"]},
        "seconds":time.monotonic()-started, "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":["Author-created seed with unjudged documents, not a production quality certification.","CPU bfloat16 differs from the active GPU precision and runtime.","Graph baseline uses source-file membership, not learned graph weights."]}
    (args.output / "provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    report = score(dataset, vectors[:len(docs)], vectors[len(docs):])
    (args.output / "metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["aggregate"],indent=2),flush=True)


def score(dataset, document_vectors, query_vectors):
    import numpy as np
    docs = dataset["documents"]
    ids = [d["id"] for d in docs]
    document_vectors = np.asarray(document_vectors, dtype=np.float64)
    query_vectors = np.asarray(query_vectors, dtype=np.float64)
    if not docs or len(ids) != len(set(ids)) or not dataset["queries"]:
        raise ValueError("evaluation requires unique documents and nonempty queries")
    if (document_vectors.ndim != 2 or query_vectors.ndim != 2
        or len(document_vectors) != len(docs) or len(query_vectors) != len(dataset["queries"])
        or document_vectors.shape[1] != query_vectors.shape[1] or document_vectors.shape[1] == 0):
        raise ValueError("vector matrices do not match the dataset")
    for matrix in (document_vectors, query_vectors):
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("vectors must be finite and nonzero")
    for question in dataset["queries"]:
        labels = question["relevance"]
        if (not labels or not set(labels).issubset(ids)
            or any(type(v) is not int or not 0 <= v <= 3 for v in labels.values())
            or not any(v > 0 for v in labels.values())):
            raise ValueError("each query needs valid judgments with a relevant document")
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
            labels = [relevance.get(ids[i],0) for i in order]
            ideal = sorted(relevance.values(),reverse=True)[:10]
            dcg = lambda values: sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values))
            rows.append({"query":question["id"],"method":method,"ranking":[ids[i] for i in order],
                "recall_at_5":sum(v>0 for v in labels[:5])/sum(v>0 for v in relevance.values()),
                "mrr_at_10":next((1/(i+1) for i,v in enumerate(labels) if v>0),0),
                "ndcg_at_10":dcg(labels)/dcg(ideal)})
    aggregate = {}
    for method in ["vector","file_membership_graph"]:
        selected = [r for r in rows if r["method"]==method]
        aggregate[method]={metric:sum(r[metric] for r in selected)/len(selected) for metric in ["recall_at_5","mrr_at_10","ndcg_at_10"]}
    return {"aggregate":aggregate,"per_query":rows}


if __name__ == "__main__":
    main()
