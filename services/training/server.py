"""HADES RGCN training service — gRPC server (Python GPU process).

The Rust orchestrator drives the training loop and issues per-step RPCs; this
server owns the GPU, model parameters, and optimizer state. It implements the
`hades.training.TrainingService` lifecycle:

    InitModel -> LoadGraph -> [TrainStep* / Evaluate]* -> GetEmbeddings
                                                       -> Checkpoint/LoadCheckpoint

HADES-owned compute service (not the Persephone PM system); decoupled from the
`persephone.*` provider brand (issue #106). Listens on a Unix socket
(`/run/hades/training.sock` by default), mirroring the embedder/extractor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
import signal
import stat
import struct
import sys
from pathlib import Path

import grpc
import torch
from grpc import aio as grpc_aio

# Ensure generated stubs are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generated"))

from hades.training import training_pb2, training_pb2_grpc  # noqa: E402

from .session import SessionTrainingServicer
from .artifacts import publish_artifact
from .worker import blocking_rpc
from .contract import validate_contract
from .config import TrainingConfig  # noqa: E402
from .rgcn_model import HadesRGCN  # noqa: E402
from .sage_model import HadesHeteroSAGE  # noqa: E402

# Encoder architectures selectable via ModelConfig.architecture. `_build_model`
# normalizes an empty/blank value to "rgcn" before lookup (back-compat with
# pre-#137 clients/checkpoints), so only the real names appear here.
_ARCHITECTURES = {
    "rgcn": HadesRGCN,
    "hetero_sage": HadesHeteroSAGE,
}

logger = logging.getLogger("hades.training")

# Node feature width when the client doesn't tell us otherwise — Jina V4 width,
# matching the schema_meta `feature_dim` default. LoadGraph rebuilds if the
# actual graph differs.
DEFAULT_FEATURE_DIM = 2048


def _bce_link_loss(pos_score: torch.Tensor, neg_score: torch.Tensor):
    """Binary cross-entropy link loss + accuracy over positives/negatives."""
    _validate_scores(pos_score, neg_score)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score)
    )
    loss = pos_loss + neg_loss
    if not torch.isfinite(loss):
        raise ValueError("link loss is not finite")
    with torch.no_grad():
        pos_acc = (pos_score > 0).float().mean() if pos_score.numel() else torch.tensor(0.0)
        neg_acc = (neg_score <= 0).float().mean() if neg_score.numel() else torch.tensor(0.0)
        acc = 0.5 * (pos_acc + neg_acc)
    return loss, float(acc)


def _validate_scores(pos_score, neg_score):
    if any(scores.ndim != 1 or scores.numel() == 0 or not torch.isfinite(scores).all()
           for scores in (pos_score, neg_score)):
        raise ValueError("metrics require nonempty finite one-dimensional scores for both classes")


def _auc(pos_score: torch.Tensor, neg_score: torch.Tensor) -> float:
    """P(positive > negative) + 0.5 P(tie), without a quadratic pair matrix."""
    _validate_scores(pos_score, neg_score)
    negatives = neg_score.sort().values
    lower = torch.searchsorted(negatives, pos_score, right=False).double()
    upper = torch.searchsorted(negatives, pos_score, right=True).double()
    return float(((lower + upper) * 0.5).mean() / neg_score.numel())


class TrainingServicer(training_pb2_grpc.TrainingServiceServicer):
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device: torch.device | None = None
        self.model: HadesRGCN | HadesHeteroSAGE | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.model_config = None
        self.opt_config = None
        self.in_dim = DEFAULT_FEATURE_DIM
        # Graph tensors (set by LoadGraph)
        self.x = None
        self.node_collections = None
        self.edge_src = None
        self.edge_dst = None
        self.edge_type = None
        self.train_idx = None  # None denotes an inference-only graph.
        self.model_contract = None  # Bound after first graph load or checkpoint restore.
        self.graph_contract = None

    # -- helpers ----------------------------------------------------------
    def _build_model(self, in_dim: int) -> int:
        mc = self.model_config
        arch = mc.architecture or "rgcn"
        try:
            model_cls = _ARCHITECTURES[arch]
        except KeyError:
            raise ValueError(
                f"unknown architecture {arch!r}; expected one of "
                f"{sorted(_ARCHITECTURES)}"
            ) from None
        # HadesRGCN and HadesHeteroSAGE share an identical constructor surface
        # and encode()/score() contract, so the only thing that varies is the
        # class selected here.
        if mc.num_relations == 0 or mc.num_collection_types == 0:
            raise ValueError("num_relations and num_collection_types must be positive")
        if not math.isfinite(mc.dropout) or not 0 <= mc.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        oc = self.opt_config
        if not math.isfinite(oc.learning_rate) or oc.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        if not math.isfinite(oc.weight_decay) or oc.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        model = model_cls(
            num_relations=mc.num_relations,
            num_collection_types=mc.num_collection_types,
            in_dim=in_dim,
            hidden_dim=mc.hidden_dim or 256,
            embed_dim=mc.embed_dim or 128,
            num_bases=mc.num_bases or mc.num_relations,
            dropout=mc.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=oc.learning_rate or 0.01,
            weight_decay=oc.weight_decay or 5e-4,
        )
        self.in_dim, self.model, self.optimizer = in_dim, model, optimizer
        return sum(p.numel() for p in model.parameters())

    async def _require_model(self, context):
        if self.model is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "InitModel not called")

    async def _require_loaded(self, context):
        await self._require_model(context)
        if self.x is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "LoadGraph not called")

    def _clear_graph(self):
        self.x = self.node_collections = None
        self.edge_src = self.edge_dst = self.edge_type = None
        self.train_idx = None
        self.graph_contract = None

    def _training_split(self, tensors):
        """Validate the partition on CPU, including reverse/duplicate leakage."""
        names = ("train_idx", "val_idx", "test_idx")
        if not any(name in tensors for name in names):
            return None  # Full adjacency is permitted only for inference.
        if not all(name in tensors for name in names):
            raise ValueError("training graphs require train_idx, val_idx, and test_idx")
        count = tensors["edge_src"].numel()
        owner = torch.full((count,), -1, dtype=torch.long)
        for group, name in enumerate(names):
            indices = tensors[name]
            if indices.ndim != 1 or not str(indices.dtype).startswith(("torch.int", "torch.uint")):
                raise ValueError(f"{name} must be a one-dimensional integer tensor")
            indices = indices.long()
            if ((indices < 0) | (indices >= count)).any() or indices.unique().numel() != indices.numel():
                raise ValueError(f"{name} contains duplicate or out-of-range indices")
            if (owner[indices] != -1).any():
                raise ValueError("edge splits overlap")
            owner[indices] = group
        if (owner == -1).any() or tensors["train_idx"].numel() == 0:
            raise ValueError("splits must cover all edges and contain training edges")
        pairs = {}
        for src, dst, group in zip(tensors["edge_src"].tolist(), tensors["edge_dst"].tolist(), owner.tolist()):
            pair = (min(src, dst), max(src, dst))
            if pairs.setdefault(pair, group) != group:
                raise ValueError("duplicate/inverse endpoint pairs must share one split")
        return tensors["train_idx"].long()

    def _adjacency(self):
        if self.train_idx is None:
            return self.edge_src, self.edge_dst, self.edge_type
        return tuple(t[self.train_idx] for t in (self.edge_src, self.edge_dst, self.edge_type))

    def _graph_tensors(self, tensors):
        """Validate on CPU before transferring or replacing any active state."""
        names = ("node_features", "node_collections", "edge_src", "edge_dst", "edge_type")
        if any(name not in tensors for name in names):
            raise ValueError("graph must contain " + ", ".join(names))
        x = tensors["node_features"]
        if x.ndim != 2 or min(x.shape) == 0 or not x.is_floating_point():
            raise ValueError("node_features must be a nonempty floating-point matrix")
        x = x.float()
        if not torch.isfinite(x).all():
            raise ValueError("node_features must contain finite float32 values")
        indices = []
        for name in names[1:]:
            values = tensors[name]
            if values.ndim != 1 or not str(values.dtype).startswith(("torch.int", "torch.uint")):
                raise ValueError(f"{name} must be a one-dimensional integer tensor")
            indices.append(values.long())
        nc, src, dst, rel = indices
        if nc.numel() != x.size(0) or not (src.numel() == dst.numel() == rel.numel()):
            raise ValueError("graph tensor lengths do not match")
        for name, values, upper in (
            ("node_collections", nc, self.model_config.num_collection_types),
            ("edge_src", src, x.size(0)), ("edge_dst", dst, x.size(0)),
            ("edge_type", rel, self.model_config.num_relations),
        ):
            if ((values < 0) | (values >= upper)).any():
                raise ValueError(f"{name} contains an out-of-range index")
        return x, nc, src, dst, rel

    def _step_indices(self, edge_indices, neg_src, neg_dst):
        """Reject invalid loss inputs before changing model mode or gradients."""
        if not edge_indices or not neg_src:
            raise ValueError("positive and negative samples must both be nonempty")
        if len(neg_src) != len(neg_dst):
            raise ValueError("neg_src and neg_dst must have equal lengths")
        for name, values, upper in (
            ("edge_indices", edge_indices, self.edge_src.numel()),
            ("neg_src", neg_src, self.x.size(0)),
            ("neg_dst", neg_dst, self.x.size(0)),
        ):
            if any(i < 0 or i >= upper for i in values):
                raise ValueError(f"{name} contains an out-of-range index")
        return self._idx(edge_indices), self._idx(neg_src), self._idx(neg_dst)

    def _idx(self, values) -> torch.Tensor:
        return torch.as_tensor(list(values), dtype=torch.long, device=self.device)

    # -- RPCs -------------------------------------------------------------
    @blocking_rpc
    async def InitModel(self, request, context):
        device = self.config.resolve_device(request.device)
        if device is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "no device declared: training requires an explicit device per "
                "run (hades --gpu N). There is no fallback device by design.",
            )
        candidate = TrainingServicer(self.config)
        try:
            candidate.device = torch.device(device)
            candidate.model_config = request.model
            candidate.opt_config = request.optimizer
            num_params = candidate._build_model(DEFAULT_FEATURE_DIM)
        except (ValueError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self.device, self.model_config, self.opt_config = (
            candidate.device, candidate.model_config, candidate.opt_config
        )
        self.in_dim, self.model, self.optimizer = candidate.in_dim, candidate.model, candidate.optimizer
        self._clear_graph()
        self.model_contract = None
        logger.info("InitModel: %d params on %s", num_params, device)
        return training_pb2.InitModelResponse(num_parameters=num_params, device=device)

    @blocking_rpc
    async def LoadGraph(self, request, context):
        await self._require_model(context)
        from safetensors import safe_open, SafetensorError

        try:
            with safe_open(request.safetensors_path, framework="pt", device="cpu") as file:
                t = {name: file.get_tensor(name) for name in file.keys()}
                metadata = file.metadata() or {}
            if "graph_contract" not in metadata:
                raise ValueError("graph has no semantic contract; rebuild it with the schema-aware loader")
            x, nc, src, dst, rel = self._graph_tensors(t)
            train_idx = self._training_split(t)
            contract = validate_contract(json.loads(metadata["graph_contract"]), self.model_config, x.size(1))
            if self.model_contract is not None and contract != self.model_contract:
                raise ValueError("graph contract differs from the bound model/checkpoint; initialize and train a new model")
            for index, name in enumerate(contract["collection_names"]):
                if not contract["feature_models"][name] and torch.any(x[nc == index] != 0):
                    raise ValueError(f"collection {name} has features without model provenance")
            # Reject incompatible metadata entirely on CPU before allocating
            # a second graph on the model's device.
            x, nc, src, dst, rel = (tensor.to(self.device) for tensor in (x, nc, src, dst, rel))
            if train_idx is not None:
                train_idx = train_idx.to(self.device)
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, SafetensorError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        feature_dim = x.size(1)
        if feature_dim != self.in_dim:
            if self.model_contract is not None:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "checkpoint feature dimension is incompatible")
            logger.info("rebuilding model for feature_dim=%d (was %d)", feature_dim, self.in_dim)
            self._build_model(feature_dim)

        self.x = x
        self.node_collections = nc
        self.edge_src, self.edge_dst, self.edge_type = src, dst, rel
        self.train_idx = train_idx
        self.model_contract = self.graph_contract = contract

        num_nodes = self.x.size(0)
        num_edges = self.edge_src.size(0)
        gpu_bytes = 0
        if self.device.type == "cuda":
            gpu_bytes = torch.cuda.memory_allocated(self.device)
        logger.info("LoadGraph: %d nodes, %d edges, dim=%d", num_nodes, num_edges, feature_dim)
        return training_pb2.LoadGraphResponse(
            num_nodes=num_nodes,
            num_edges=num_edges,
            feature_dim=feature_dim,
            gpu_memory_bytes=gpu_bytes,
        )

    def _encode(self) -> torch.Tensor:
        return self.model.encode(
            self.x, self.node_collections, *self._adjacency()
        )

    def _validate_subset(self, requested: list[int]) -> None:
        if not isinstance(self.model, HadesHeteroSAGE):
            raise ValueError("node subset inference requires architecture 'hetero_sage'")
        num_nodes = self.x.size(0)
        if len(set(requested)) != len(requested):
            raise ValueError("node_indices must not contain duplicates")
        if any(index < 0 or index >= num_nodes for index in requested):
            raise ValueError(f"node index out of range for graph with {num_nodes} nodes")

    def _encode_subset(self, requested: list[int]) -> torch.Tensor:
        """Embed targets using only the incoming neighbourhood each layer needs."""
        self._validate_subset(requested)
        num_nodes = self.x.size(0)
        targets = torch.tensor(requested, device=self.device, dtype=torch.long)
        active = torch.zeros(num_nodes, device=self.device, dtype=torch.bool)
        active[targets] = True
        edge_src, edge_dst, edge_type = self._adjacency()
        for _ in self.model.convs:
            incoming_edges = active[edge_dst]
            active[edge_src[incoming_edges]] = True

        global_nodes = active.nonzero(as_tuple=False).flatten()
        global_to_local = torch.full(
            (num_nodes,), -1, device=self.device, dtype=torch.long
        )
        global_to_local[global_nodes] = torch.arange(
            global_nodes.numel(), device=self.device
        )
        edge_mask = active[edge_src] & active[edge_dst]
        local_src = global_to_local[edge_src[edge_mask]]
        local_dst = global_to_local[edge_dst[edge_mask]]
        local_type = edge_type[edge_mask]

        local_embeddings = self.model.encode(
            self.x[global_nodes],
            self.node_collections[global_nodes],
            local_src,
            local_dst,
            local_type,
        )
        return local_embeddings[global_to_local[targets]]

    @blocking_rpc
    async def TrainStep(self, request, context):
        await self._require_loaded(context)
        if self.train_idx is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "load a graph with explicit training splits")
        try:
            pos_idx, neg_src, neg_dst = self._step_indices(
                request.train_edge_indices, request.neg_src, request.neg_dst
            )
            if not torch.isin(pos_idx, self.train_idx).all():
                raise ValueError("training targets must belong to train_idx")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self.model.train()
        self.optimizer.zero_grad()
        emb = self._encode()
        pos_src, pos_dst = self.edge_src[pos_idx], self.edge_dst[pos_idx]
        pos_score = self.model.score(emb, pos_src, pos_dst)
        neg_score = self.model.score(emb, neg_src, neg_dst)
        try:
            loss, acc = _bce_link_loss(pos_score, neg_score)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        loss.backward()
        self.optimizer.step()
        return training_pb2.TrainStepResponse(loss=float(loss.detach()), accuracy=acc)

    @blocking_rpc
    async def Evaluate(self, request, context):
        await self._require_loaded(context)
        if self.train_idx is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "load a graph with explicit training splits")
        try:
            pos_idx, neg_src, neg_dst = self._step_indices(
                request.edge_indices, request.neg_src, request.neg_dst
            )
            if torch.isin(pos_idx, self.train_idx).any():
                raise ValueError("evaluation targets must belong to a held-out split")
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self.model.eval()
        with torch.no_grad():
            emb = self._encode()
            pos_src, pos_dst = self.edge_src[pos_idx], self.edge_dst[pos_idx]
            pos_score = self.model.score(emb, pos_src, pos_dst)
            neg_score = self.model.score(emb, neg_src, neg_dst)
            try:
                loss, acc = _bce_link_loss(pos_score, neg_score)
                auc = _auc(pos_score, neg_score)
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return training_pb2.EvaluateResponse(loss=float(loss), accuracy=acc, auc=auc)

    @blocking_rpc
    async def GetEmbeddings(self, request, context):
        await self._require_loaded(context)
        try:
            if request.node_indices:
                self._validate_subset(list(request.node_indices))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        self.model.eval()
        try:
            with torch.no_grad():
                if request.node_indices:
                    emb = self._encode_subset(list(request.node_indices))
                else:
                    emb = self._encode()
                emb = emb.cpu().contiguous().float()
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        num_nodes, embed_dim = emb.size(0), emb.size(1)
        if request.output_path:
            # Write a RAW little-endian f32 blob (num_nodes × embed_dim) — the
            # Rust export reader (graph/export.rs) reads this file as raw f32
            # (chunks_exact(4)/from_le_bytes), NOT safetensors. A safetensors
            # header would be miscounted as ~24 extra floats and trip the
            # length check (#134). Matches the inline path's `.tobytes()`.
            import numpy as np

            with publish_artifact(request.output_path) as output:
                np.ascontiguousarray(emb.numpy(), dtype="<f4").tofile(output)
            return training_pb2.GetEmbeddingsResponse(
                num_nodes=num_nodes, embed_dim=embed_dim, output_path=request.output_path
            )
        payload = emb.numpy().tobytes()
        return training_pb2.GetEmbeddingsResponse(
            num_nodes=num_nodes, embed_dim=embed_dim, embeddings=payload
        )

    @blocking_rpc
    async def Checkpoint(self, request, context):
        await self._require_model(context)
        if self.model_contract is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "load a graph with verified provenance before saving a checkpoint")
        if not request.path:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "checkpoint path is required")
        Path(request.path).parent.mkdir(parents=True, exist_ok=True)
        with publish_artifact(request.path) as output:
            torch.save(
                {
                    "graph_contract": self.model_contract,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "in_dim": self.in_dim,
                    "model_config": {
                        "num_relations": self.model_config.num_relations,
                        "num_collection_types": self.model_config.num_collection_types,
                        "hidden_dim": self.model_config.hidden_dim,
                        "embed_dim": self.model_config.embed_dim,
                        "num_bases": self.model_config.num_bases,
                        "dropout": self.model_config.dropout,
                        # Persist the architecture so LoadCheckpoint (e.g. for
                        # `graph-embed update`) rebuilds the matching model rather
                        # than defaulting to RGCN.
                        "architecture": self.model_config.architecture,
                    },
                },
                output,
            )
        size = os.path.getsize(request.path)
        return training_pb2.CheckpointResponse(path=request.path, size_bytes=size)

    @blocking_rpc
    async def LoadCheckpoint(self, request, context):
        device = self.config.resolve_device(request.device)
        if device is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "no device declared: checkpoint loading requires an explicit "
                "device per run (hades --gpu N). There is no fallback device "
                "by design.",
            )
        # weights_only=True: the checkpoint holds only tensors + basic types,
        # and refusing arbitrary unpickling avoids code execution from a
        # tampered checkpoint file.
        candidate = TrainingServicer(self.config)
        try:
            candidate.device = torch.device(device)
            ckpt = torch.load(request.path, map_location=candidate.device, weights_only=True)
            candidate.model_config = training_pb2.ModelConfig(**ckpt["model_config"])
            if "graph_contract" not in ckpt:
                raise ValueError("legacy checkpoint lacks semantic provenance; retrain with a versioned graph contract")
            candidate.model_contract = validate_contract(ckpt["graph_contract"], candidate.model_config, ckpt["in_dim"])
            candidate.opt_config = self.opt_config or training_pb2.OptimizerConfig()
            num_params = candidate._build_model(ckpt.get("in_dim", DEFAULT_FEATURE_DIM))
            candidate.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                candidate.optimizer.load_state_dict(ckpt["optimizer"])
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, EOFError, pickle.UnpicklingError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if candidate.device != self.device or candidate.model_contract != self.graph_contract:
            self._clear_graph()
        self.model_contract = candidate.model_contract
        self.device, self.model_config, self.opt_config = (
            candidate.device, candidate.model_config, candidate.opt_config
        )
        self.in_dim, self.model, self.optimizer = candidate.in_dim, candidate.model, candidate.optimizer
        return training_pb2.LoadCheckpointResponse(
            model_config=self.model_config, num_parameters=num_params, device=device
        )


async def serve() -> None:
    config = TrainingConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    servicer = SessionTrainingServicer(lambda: TrainingServicer(config))
    server = grpc_aio.server(options=[
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ])
    training_pb2_grpc.add_TrainingServiceServicer_to_server(servicer, server)

    socket_path = config.socket_path
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    sock = Path(socket_path)
    if sock.exists():
        if stat.S_ISSOCK(sock.stat().st_mode):
            sock.unlink()
        else:
            raise RuntimeError(f"Path {socket_path} exists but is not a socket")

    server.add_insecure_port(f"unix:{socket_path}")
    logger.info("Starting training service on %s (no fallback device: clients declare per run)", socket_path)
    await server.start()

    # Ensure the socket is group-connectable (the `hades` group). Under systemd
    # the unit's UMask=0007 already births it 0770, making this a no-op
    # verification; outside systemd the ambient umask (typically 022) would
    # leave it 0755 — no group write — blocking same-group clients, so this
    # does the work there.
    try:
        os.chmod(socket_path, 0o770)
    except OSError as exc:
        # The socket is unusable by group clients without this, so don't
        # advertise "ready" — fail loudly and let systemd restart.
        logger.error("failed to chmod %s to 0770; aborting startup: %s", socket_path, exc)
        await server.stop(grace=0)
        raise

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("Training service ready")
    sweeper = asyncio.create_task(servicer.sweep_expired())
    try:
        await stop_event.wait()
    finally:
        sweeper.cancel()
        await asyncio.gather(sweeper, return_exceptions=True)
        logger.info("Shutting down training service...")
        await server.stop(grace=5)
        await servicer.close()
    if sock.exists() and stat.S_ISSOCK(sock.stat().st_mode):
        sock.unlink()
    logger.info("Training service stopped")


if __name__ == "__main__":
    asyncio.run(serve())
