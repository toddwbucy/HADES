"""Jina V4 embedding backend — the ML boundary for vector generation.

Wraps jinaai/jina-embeddings-v4 behind a minimal interface.
The model is Qwen2.5-VL-3B-Instruct with 3 task-specific LoRA adapters
(retrieval, text-matching, code). trust_remote_code is required because
Jina forks the entire Qwen forward pass to route task_label through
every layer for adapter selection.

Everything else (chunking, batching, DB writes) is handled by the Rust
orchestrator.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import threading
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Jina V4 constants
EMBEDDING_DIM = 2048

# **Not the model's advertised 32768.** That is the architectural limit, not
# what fits. Bisected on GPU 2 (RTX 2000 Ada, 16 GiB) on 2026-09-12 against a
# freshly started process with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
#
#     30000  OOM (peak 15,856 MiB of 16,380)
#     24000  OOM (peak 15,796 MiB)
#     16384  OOM        <- Yeomna's figure, see below
#     16000  OOM
#     15000  OK         <- last size that completes
#     12000  OK
#
# The model is 9,216 MiB resident, leaving roughly 6 GiB for activations, and
# a single forward pass over ~16k tokens does not fit in that.
#
# **Yeomna's embedder advertised 16384 on this same card, and that number was
# a refusal threshold rather than a demonstrated forward pass.** It rejected
# longer inputs with "N tokens exceeds max_tokens 16384" and chunked what it
# accepted, so it never actually ran 16k through the model in one pass. Do not
# read it as evidence that 16k works here, because it does not.
#
# 15000 is the default: the largest size that completed. Note it is the
# measured edge with no margin, and a real workload carries batch and
# fragmentation pressure a single clean probe does not, so an OOM here means
# drop to 14000 rather than concluding the card changed.
#
# **16000 is not a safe round number, it is the first size that fails.**
#
# For a document that needs more, move it rather than raising this: the two
# A6000s have 49 GiB apiece, so
#   HADES_EMBEDDER_DEVICE=cuda:0 HADES_EMBEDDER_MAX_TOKENS=32768
# runs a second instance on a big card for the oversized case.
#
# Override with HADES_EMBEDDER_MAX_TOKENS when running elsewhere.
MAX_TOKENS = int(os.environ.get("HADES_EMBEDDER_MAX_TOKENS", "15000"))

# Preflight sizing, measured on olympus 2026-09-12 with the model in fp16.
#
# **The default check detects contention, it does not predict OOM.** Those are
# different jobs and conflating them makes the gate useless: a full activation
# estimate (0.45 MiB/token, derived from the 32,768-token run peaking at
# 23,906 MiB) comes to 15,966 MiB at MAX_TOKENS=15000, which exceeds the
# 15,489 MiB free on an *idle* 16 GiB card — so it refuses a configuration
# that demonstrably works. Predicting the edge to the byte is not achievable
# from a resident-size constant, because peak depends on allocator behaviour
# that varies run to run.
#
# So the default asks only "can this card hold the model with room to work",
# which is what distinguishes a free card from one with another job on it.
# The genuine edge is enforced by actually running out of memory, and by
# MAX_TOKENS being set from measurement rather than hope.
#
# When you want a hard guarantee instead, set the flat floor below.
MODEL_RESIDENT_MIB = 9216
CONTENTION_HEADROOM_MIB = 2048

# A flat floor, in MiB, that overrides the computed estimate when set.
# Prefer this when you want a blunt guarantee rather than a derivation: the
# 32,768-token run peaked at 23,906 MiB, so
#   HADES_EMBEDDER_MIN_FREE_VRAM_MIB=24576
# on the large-document instance means "give me 24 GiB or refuse", and no
# amount of drift in the per-token estimate can quietly erode it.
# Unset (0) uses the computed estimate, which is what the 16 GiB card wants,
# since a 24 GiB floor would refuse that card outright.
MIN_FREE_VRAM_MIB = int(os.environ.get("HADES_EMBEDDER_MIN_FREE_VRAM_MIB", "0"))

# Task label mapping: proto task string → Jina LoRA adapter name
_TASK_TO_ADAPTER = {
    "retrieval.passage": "retrieval",
    "retrieval.query": "retrieval",
    "retrieval": "retrieval",
    "text-matching": "text-matching",
    "code": "code",
}

# Supported task labels exposed via Info RPC
SUPPORTED_TASKS = ["retrieval.passage", "retrieval.query", "text-matching", "code"]


class JinaV4Embedder:
    """Embed text using Jina V4 (Qwen2.5-VL-3B + LoRA adapters).

    Lazy-loads the model on first use. Call unload() to release GPU memory.
    """

    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda:2",
        use_fp16: bool = True,
        batch_size: int = 128,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

        # Validate device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self._device = "cpu"
            else:
                # Validate ordinal if specified (e.g. "cuda:2")
                if ":" in device:
                    try:
                        ordinal = int(device.split(":", 1)[1])
                    except ValueError:
                        raise ValueError(
                            f"Invalid CUDA device ordinal: {device!r}"
                        ) from None
                    count = torch.cuda.device_count()
                    if ordinal < 0 or ordinal >= count:
                        raise ValueError(
                            f"CUDA device {device!r} out of range "
                            f"(available: {count} device(s), indices 0-{count - 1})"
                        )
                self._device = device
        else:
            self._device = device

        self._dtype = (
            torch.float16
            if (use_fp16 and self._device.startswith("cuda"))
            else torch.float32
        )

        self._load_lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _preflight_device(self) -> None:
        """Refuse to load when the target GPU cannot hold the run.

        Without this the service starts, reports ready, accepts a request and
        only then dies inside the forward pass with a CUDA OOM — which reads
        like a bad input rather than a busy card. The failure is cheap to
        predict and expensive to diagnose after the fact, so predict it.

        Sizing comes from measurements on olympus 2026-09-12, model in fp16:

            resident, model only            9,216 MiB
            peak at 15,000 tokens          15,856 MiB   (RTX 2000 Ada, 16 GiB)
            peak at 32,768 tokens          23,906 MiB   (RTX A6000, 49 GiB)

        Activation cost is close to linear in sequence length across that
        range, about 0.45 MiB per token, which is what ACTIVATION_MIB_PER_TOKEN
        encodes. The estimate is deliberately an estimate: it exists to catch
        "another job already owns this card", not to predict OOM to the byte.
        """
        if not self._device.startswith("cuda"):
            return

        ordinal = int(self._device.split(":", 1)[1]) if ":" in self._device else 0
        free_b, total_b = torch.cuda.mem_get_info(ordinal)
        free_mib = free_b // (1024 * 1024)
        total_mib = total_b // (1024 * 1024)

        if MIN_FREE_VRAM_MIB > 0:
            needed_mib = MIN_FREE_VRAM_MIB
            basis = f"HADES_EMBEDDER_MIN_FREE_VRAM_MIB={MIN_FREE_VRAM_MIB}"
        else:
            needed_mib = MODEL_RESIDENT_MIB + CONTENTION_HEADROOM_MIB
            basis = (
                f"model {MODEL_RESIDENT_MIB} MiB plus "
                f"{CONTENTION_HEADROOM_MIB} MiB working headroom"
            )

        if free_mib >= needed_mib:
            logger.info(
                "preflight ok: %s has %d MiB free of %d, need %d (%s)",
                self._device, free_mib, total_mib, needed_mib, basis,
            )
            return

        # Name who is holding the card. Best effort: a missing or unparseable
        # nvidia-smi must not turn a clear refusal into a traceback.
        holders = ""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                 "--format=csv,noheader", "-i", str(ordinal)],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                holders = " Currently on this device: " + "; ".join(out.splitlines()) + "."
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

        raise RuntimeError(
            f"refusing to load on {self._device}: {free_mib} MiB free of "
            f"{total_mib} MiB, need {needed_mib} MiB ({basis}).{holders} "
            f"Free the device, lower HADES_EMBEDDER_MAX_TOKENS, or point "
            f"CUDA_VISIBLE_DEVICES at a card with room."
        )

    def _load_unlocked(self) -> None:
        """Load model and tokenizer. Caller MUST hold _load_lock."""
        if self._model is not None:
            return

        self._preflight_device()

        logger.info(
            "Loading %s on %s (dtype=%s)", self._model_name, self._device, self._dtype
        )
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )

        model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True, torch_dtype=self._dtype
        )

        if self._device != "cpu":
            model = model.to(self._device)

        # Set model to inference mode
        model.requires_grad_(False)

        # Assign atomically — both succeed or neither is visible
        self._tokenizer = tokenizer
        self._model = model

        logger.info(
            "Model loaded in %.2fs (dtype=%s)",
            time.time() - start,
            next(self._model.parameters()).dtype,
        )

    @property
    def model(self):
        """Lazy-load model on first access (double-checked locking)."""
        m = self._model
        if m is not None:
            return m
        with self._load_lock:
            if self._model is None:
                self._load_unlocked()
            return self._model

    @property
    def tokenizer(self):
        """Lazy-load tokenizer on first access (double-checked locking)."""
        t = self._tokenizer
        if t is not None:
            return t
        with self._load_lock:
            if self._tokenizer is None:
                self._load_unlocked()
            return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def max_sequence_length(self) -> int:
        return MAX_TOKENS

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_texts(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed texts using Jina V4.

        Args:
            texts: List of texts to embed.
            task: Task type controlling LoRA adapter selection.
            batch_size: Override default batch size.

        Returns:
            Numpy array of shape (N, 2048) with float32 embeddings.
        """
        all_embeddings: list[np.ndarray] = []
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._embed_batch(batch, task)
                all_embeddings.append(embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32, copy=False)
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _embed_batch(self, batch: list[str], task: str) -> np.ndarray:
        """Embed a single batch, choosing the best available API."""
        model = self.model  # triggers lazy load

        # Prefer Jina's high-level encode_text() when available
        if hasattr(model, "encode_text"):
            jina_task = _TASK_TO_ADAPTER.get(task, "retrieval")
            prompt_name = "query" if task == "retrieval.query" else "passage"
            embeddings = model.encode_text(
                batch, task=jina_task, prompt_name=prompt_name
            )
        else:
            # Fallback: raw forward pass with task_label for LoRA selection
            task_label = _TASK_TO_ADAPTER.get(task, "retrieval")
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(self._device)

            outputs = model(**inputs, task_label=task_label)

            if hasattr(outputs, "single_vec_emb") and outputs.single_vec_emb is not None:
                embeddings = outputs.single_vec_emb
            else:
                available = [a for a in dir(outputs) if not a.startswith("_")]
                raise AttributeError(
                    f"Expected 'single_vec_emb' in output, got: {available}"
                )

        return self._to_numpy(embeddings)

    @staticmethod
    def _to_numpy(embeddings: Any) -> np.ndarray:
        """Convert embeddings (tensor, list, or ndarray) to numpy float32."""
        if torch.is_tensor(embeddings):
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            return embeddings.numpy().astype(np.float32, copy=False)

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach()
            if hasattr(embeddings, "is_cuda") and embeddings.is_cuda:
                embeddings = embeddings.cpu()
            return embeddings.numpy().astype(np.float32, copy=False)

        if isinstance(embeddings, list):
            processed = []
            for e in embeddings:
                if torch.is_tensor(e):
                    if e.is_cuda:
                        e = e.cpu()
                    processed.append(e.numpy())
                else:
                    processed.append(np.array(e))
            return np.vstack(processed).astype(np.float32, copy=False)

        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32, copy=False)

        return np.array(embeddings, dtype=np.float32)

    def unload(self) -> None:
        """Release GPU memory held by the model."""
        with self._load_lock:
            if self._model is not None:
                logger.info("Unloading Jina V4 model...")
                del self._model
                self._model = None
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception as e:
            logger.debug("Failed to clear CUDA cache: %s", e)
        logger.info("Jina V4 model unloaded")
