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

# Export the backend failure type for transport classification without string matching.
BackendOutOfMemoryError = torch.OutOfMemoryError
from transformers import AutoModel, AutoTokenizer

from .tensors import embeddings_to_numpy

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
# Measured directly on olympus 2026-09-13 by loading the model and reading
# torch.cuda.memory_allocated: **7,993 MiB**, of which 7,162 MiB is 3.755B fp16
# parameters and 686 MiB is 1,518 fp32 LoRA tensors that peft keeps unquantized.
#
# An earlier pass set this to 12,288 from the "12.03 GiB allocated" line in an
# OOM traceback. That reading was taken *during* the failing forward pass, so it
# counted the model plus roughly 4 GiB of partial activations, and using it here
# made the preflight demand 14,336 MiB free on a 16 GiB card that has about
# 15,500: any other process touching the card would have made HADES refuse it.
#
# The bisected sequence ceilings are unaffected, since those were measured by
# running documents rather than derived from this number. What this constant
# gates is the contention check, and 8,192 is the measured model rounded up.
MODEL_RESIDENT_MIB = 8192

# Tokens held back from the ceiling for what the encode adds on top of the text.
#
# `encode_text` prepends a task prompt ("Passage: ", "Query: ") and the tokenizer
# adds special tokens, neither of which a bare count of the input can see. Eight
# is a margin rather than a measurement: the prompt is one short word plus a
# colon and space, and the specials are a handful, so this is comfortably above
# both while costing 0.02 percent of a 32,768-token window.
PROMPT_RESERVE_TOKENS = 8
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


class InputTooLargeError(ValueError):
    """An input does not fit the load profile's sequence ceiling.

    Raised instead of letting the input be truncated. A 45,183-token document
    sent to a profile with a 32,768-token ceiling used to return one vector and
    HTTP 200: `encode_text` truncates at its own max_length, silently, so 27
    percent of that document was discarded and the response said success. A
    corpus ingested that way is wrong in a way no count reconciles, because the
    vector count matches the input count exactly.

    Subclasses ValueError so the HTTP layer's existing arm maps it to 400, and
    carries the PE_INPUT_TOO_LARGE code in its message per the PE-API error
    table, which has specified this case from the start.

    The caller's remedy is to pre-chunk, or to run a profile whose card holds
    more: the ceiling is a property of the card the model is loaded on, and it
    is reported on /v1/models so a client never has to guess it.
    """

    def __init__(self, index: int, n_tokens: int, max_tokens: int) -> None:
        self.index = index
        self.n_tokens = n_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"PE_INPUT_TOO_LARGE: input {index} is {n_tokens} tokens, which "
            f"exceeds this profile's {max_tokens}-token ceiling. Refusing "
            f"rather than truncating and reporting success. Pre-chunk the "
            f"input, or load a profile on a card that holds more."
        )


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
        # HuggingFace's fast tokenizer is Rust-backed and NOT thread-safe:
        # concurrent calls raise "RuntimeError: Already borrowed". The HTTP
        # server runs inference in a thread pool, so two overlapping requests
        # collide. Serializing is correct regardless, since the GPU is a single
        # resource and parallel forward passes only contend for it.
        self._infer_lock = threading.Lock()
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

        # Checked for every input before any forward pass runs, so one oversized
        # input fails the request rather than after burning GPU time on the
        # batches ahead of it. The count comes from the same tokenizer that
        # would do the truncating, so it is the authoritative number and not an
        # estimate from character length.
        self._assert_inputs_fit(texts)

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._embed_batch(batch, task)
                all_embeddings.append(embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32, copy=False)
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _assert_inputs_fit(self, texts: list[str]) -> None:
        """Refuse inputs longer than this profile's ceiling.

        `_embed_batch_locked` prefers the model's own `encode_text`, which
        truncates at its configured max_length and says nothing. The dead
        fallback path below it passes `truncation=True` explicitly, so both
        arms silently discarded the tail of an oversized input.

        Holds `_infer_lock` the way every other tokenizer call here does: the
        Rust-backed fast tokenizer is not thread-safe and concurrent calls raise
        "Already borrowed", which is findings #8. Released before the forward
        pass takes the same lock, so this does not deadlock a non-reentrant
        lock.
        """
        budget = MAX_TOKENS - PROMPT_RESERVE_TOKENS
        with self._infer_lock:
            tokenizer = self.tokenizer
            for index, text in enumerate(texts):
                # Counted WITH special tokens, against a budget reduced by the
                # task prompt. `_embed_batch_locked` calls `encode_text` with a
                # `prompt_name`, which prepends a prefix this count cannot see,
                # and the fallback arm tokenizes with specials. Counting bare text
                # against the full ceiling let an input of exactly MAX_TOKENS pass
                # the guard and then be truncated by the overhead, which is the
                # silent loss the guard exists to stop.
                n_tokens = len(tokenizer(text, add_special_tokens=True)["input_ids"])
                if n_tokens > budget:
                    raise InputTooLargeError(index, n_tokens, budget)

    def embed_late_chunked(
        self,
        text: str,
        task: str = "retrieval.passage",
        chunk_size_tokens: int = 500,
        overlap_tokens: int = 200,
        boundaries: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Encode `text` in ONE pass, then pool per chunk (late chunking).

        The whole document is encoded once, so every chunk vector is
        conditioned on the surrounding document. Contrast `embed_texts`, which
        encodes each chunk independently and loses that context entirely.

        **Pools `vlm_last_hidden_states`, not `multi_vec_emb`.** The model's own
        single-vector path is mean-pooling over the sequence followed by L2
        normalisation:

            pooled = sum(hidden * mask) / sum(mask);  normalize(pooled)

        Doing that over a chunk's token range instead of the whole sequence
        puts the result in the *same vector space* as every single-vector
        embedding, so late-chunked and whole-document vectors stay comparable.
        `multi_vec_emb` passes through `multi_vector_projector` into a
        ColBERT-style late-interaction space and is NOT interchangeable.

        `hidden_states` is computed on every forward pass regardless, since
        both pooling paths consume it. It simply has to be requested back with
        output_vlm_last_hidden_states.

        `boundaries` are character ranges over `text`. Pass them to pool at
        meaningful places — AST definitions, document sections — instead of
        arbitrary token windows. This is what lets a code ingest keep chunks
        aligned to symbols while still gaining whole-file context. Ranges are
        mapped to token spans through the fast tokenizer's offset mapping.
        Omit them to fall back to uniform windows of `chunk_size_tokens`.

        Returns the `[num_chunks, 2048]` matrix and, per chunk,
        `(token_start, token_end, char_start, char_end)`. Character ranges are
        returned because callers need them to intersect chunks with symbol
        spans, which token indices cannot express.

        Callers must keep `text` within max_sequence_length. Pre-chunking at
        section boundaries is the escape hatch for documents that exceed it.
        """
        if chunk_size_tokens <= 0:
            raise ValueError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
        if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                f"overlap_tokens must be in [0, chunk_size_tokens), got {overlap_tokens}"
            )

        if boundaries is not None:
            # Checked before the forward pass, and reported as what it is. The
            # only error this path used to raise blamed context-window
            # truncation for everything, so an out-of-range or inverted range
            # from a client bug reached the operator as "send smaller windows",
            # which is the wrong remedy and points at the wrong component.
            n_chars = len(text)
            bad = [(a, b) for a, b in boundaries if a < 0 or b < a or b > n_chars]
            if bad:
                raise ValueError(
                    f"{len(bad)} of {len(boundaries)} boundaries do not describe "
                    f"a range within the {n_chars}-character input. First: "
                    f"{bad[0]}. Boundaries are character offsets over `input`, "
                    f"with 0 <= start <= end <= len(input)."
                )

        model = self.model
        task_label = _TASK_TO_ADAPTER.get(task, "retrieval")

        # **The prefix is not optional.** The normal path calls
        # encode_text(prompt_name=...), and the model's processor prepends
        # "Passage: " or "Query: " before tokenizing. Embedding raw text here
        # instead put late-chunked vectors at cosine 0.978 against the
        # single-vector path for identical input, so they were systematically
        # offset from everything already stored. Same prefix, same space.
        prefix = "Query" if task == "retrieval.query" else "Passage"
        prefixed = f"{prefix}: {text}"
        with self._infer_lock, torch.no_grad():
            encoded = self.tokenizer(
                [prefixed],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_offsets_mapping=True,
            )
            # offset_mapping is not a model input; keep it here and strip it.
            offsets = encoded.pop("offset_mapping")[0].tolist()
            inputs = encoded.to(self._device)

            outputs = model(
                **inputs,
                task_label=task_label,
                output_vlm_last_hidden_states=True,
            )
            hidden = outputs.vlm_last_hidden_states
            if hidden is None:
                raise AttributeError(
                    "model did not return vlm_last_hidden_states; late chunking "
                    "requires it and cannot fall back to single_vec_emb"
                )

            mask = inputs["attention_mask"][0]           # [seq]
            hidden = hidden[0]                            # [seq, dim]
            n_tokens = int(mask.sum().item())

            # Chunk over document tokens only, skipping the prefix, but pool
            # from the full hidden states so each chunk still sees the whole
            # encoded sequence around it.
            prefix_chars = len(f"{prefix}: ")
            # Index of the first document token, taken from this encoding's own
            # offset mapping rather than by tokenizing the prefix separately.
            # A separate tokenization omits the special tokens this one adds and
            # assumes the prefix tokenizes identically standalone, which BPE
            # does not guarantee. Leading special tokens carry (0, 0) offsets,
            # so they fall before the prefix and are counted here too.
            first_doc_token = next(
                (i for i, (ts, te) in enumerate(offsets[:n_tokens]) if te > prefix_chars),
                n_tokens,
            )
            # Highest document character any real token covers. Special and
            # padding tokens carry (0, 0) offsets and are excluded.
            covered_chars = (
                max((te for ts, te in offsets[:n_tokens] if te > ts), default=prefix_chars)
                - prefix_chars
            )
            # Exclusive end of the document's real tokens.
            #
            # `n_tokens` counts every non-padding token, trailing special tokens
            # included, and those carry (0, 0) offsets. A uniform window whose last
            # token is one of them reports `char_end = max(0, 0 - prefix_chars)`,
            # which is 0: an inverted range that a caller intersecting against
            # symbol spans silently matches nothing from. Bounding the walk by the
            # last token that covers a real character keeps the specials out of the
            # spans without excluding them from the pooling, which still runs over
            # the full hidden states.
            doc_token_end = max(
                (i + 1 for i, (ts, te) in enumerate(offsets[:n_tokens]) if te > ts),
                default=first_doc_token,
            )

            spans: list[tuple[int, int]] = []
            if boundaries is not None:
                # Detect truncation directly rather than inferring it from a
                # boundary that mapped to nothing.
                #
                # The "mapped to no tokens" check below only fires when a
                # boundary matches ZERO tokens, and the match is an overlap
                # test, so a boundary straddling the truncation point still
                # matches its surviving head tokens and raises nothing. A single
                # AST chunk larger than the client's window arrives as exactly
                # that: one boundary spanning more text than the model can hold,
                # pooled from the part that fitted, reported as a success. The
                # comment in codebase_ingest.rs claiming truncation now fails
                # loudly was true only for the multi-boundary case.
                if n_tokens >= MAX_TOKENS:
                    over = [b for b in boundaries if b[1] > covered_chars]
                    if over:
                        raise ValueError(
                            f"input filled the {MAX_TOKENS}-token window and was "
                            f"truncated at character {covered_chars} of {len(text)}, "
                            f"so {len(over)} of {len(boundaries)} boundaries extend "
                            f"past what the model saw. First: {over[0]}. Send smaller "
                            f"windows."
                        )
                # Map each character range to the tokens covering it. A token
                # counts as inside when it overlaps the range at all, so a
                # boundary landing mid-token still captures that token.
                unmapped: list[tuple[int, int]] = []
                for c_start, c_end in boundaries:
                    ps, pe = c_start + prefix_chars, c_end + prefix_chars
                    toks = [
                        i
                        for i, (ts, te) in enumerate(offsets)
                        if i < n_tokens and te > ts and ts < pe and te > ps
                    ]
                    if toks:
                        spans.append((toks[0], toks[-1] + 1))
                    else:
                        unmapped.append((c_start, c_end))

                # **Refuse rather than silently drop.** A boundary maps to no
                # tokens when the input was truncated at max_length, so the
                # caller sent more text than the context window holds. Skipping
                # those quietly returns fewer vectors than boundaries and the
                # caller has no way to notice: chunks end up stored with no
                # embedding and are silently unsearchable.
                if unmapped:
                    raise ValueError(
                        f"{len(unmapped)} of {len(boundaries)} boundaries mapped to no "
                        f"tokens, so the input exceeded the {MAX_TOKENS}-token window "
                        f"and was truncated. First unmapped char range: {unmapped[0]}. "
                        f"Send smaller windows."
                    )
            else:
                # Truncation has to fail here as loudly as it does on the
                # boundaries path above. Without this, an oversized input is cut
                # at max_length, the walk below covers only what survived, and the
                # response is HTTP 200 with fewer chunks than the text warrants:
                # the caller cannot tell a short document from a truncated one.
                if n_tokens >= MAX_TOKENS:
                    raise ValueError(
                        f"PE_INPUT_TOO_LARGE: input filled the {MAX_TOKENS}-token "
                        f"window and was truncated at character {covered_chars} of "
                        f"{len(text)}, so uniform windows would cover only the part "
                        f"the model saw. Send a smaller input."
                    )
                step = chunk_size_tokens - overlap_tokens
                start = first_doc_token
                while start < doc_token_end:
                    end = min(start + chunk_size_tokens, doc_token_end)
                    spans.append((start, end))
                    if end >= doc_token_end:
                        break
                    start += step

            # Keep each span next to the vector pooled from it. Reconciling
            # the two lists afterwards with spans[:len(vectors)] truncated from
            # the front, so a span skipped anywhere in the middle relabelled
            # every vector after it with the previous span's range and shifted
            # every chunk_index by one.
            kept: list[tuple[int, int]] = []
            vectors = []
            for s, e in spans:
                span_mask = mask[s:e].unsqueeze(-1)
                denom = span_mask.sum()
                if denom.item() == 0:
                    continue
                pooled = (hidden[s:e] * span_mask).sum(dim=0) / denom
                vectors.append(torch.nn.functional.normalize(pooled, dim=-1))
                kept.append((s, e))

            # With caller-supplied boundaries the count is a contract: one
            # vector per boundary. Returning fewer is the silent-omission
            # failure this path exists to end.
            if boundaries is not None and len(kept) != len(spans):
                raise ValueError(
                    f"{len(spans) - len(kept)} of {len(spans)} boundaries fell "
                    f"entirely on padding and produced no vector"
                )

            if not vectors:
                return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
            stacked = torch.stack(vectors)

        # Report spans relative to the document, not the prefixed string, and
        # include character ranges so callers can intersect with symbol spans.
        out_spans: list[tuple[int, int, int, int]] = []
        for s, e in kept:
            c_start = max(0, offsets[s][0] - prefix_chars) if s < len(offsets) else 0
            c_end = max(0, offsets[e - 1][1] - prefix_chars) if e - 1 < len(offsets) else 0
            # Clamp at zero: a prefix token can overlap the document's first
            # character (the tokenizer may merge ": " with what follows), which
            # would otherwise report a negative document-relative index.
            out_spans.append(
                (max(0, s - first_doc_token), max(0, e - first_doc_token), c_start, c_end)
            )
        return self._to_numpy(stacked), out_spans

    def _embed_batch(self, batch: list[str], task: str) -> np.ndarray:
        """Embed a single batch, choosing the best available API.

        Holds the inference lock for the same reason ``embed_late_chunked``
        does. Both branches below reach the Rust-backed fast tokenizer, one
        directly and one inside ``encode_text``, and it raises "Already
        borrowed" when two threads touch it at once. The HTTP server runs
        inference in a thread pool, so a plain request overlapping a
        late-chunked one is enough to hit it.
        """
        model = self.model  # triggers lazy load

        with self._infer_lock:
            return self._embed_batch_locked(model, batch, task)

    def _embed_batch_locked(self, model, batch: list[str], task: str) -> np.ndarray:
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
        return embeddings_to_numpy(embeddings)

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
