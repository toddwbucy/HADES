"""Opt-in receipt-enforced Jina text adapter; production startup does not enable it.

Own prompts, tokenization and pooling instead of certifying the legacy high-level
encoder's implicit defaults. Approved model code and installed libraries remain
trusted; receipts detect artifact/behavior drift, not malicious runtime code.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json

import torch

from .identity import ArtifactManifest, ProviderReceipt, embedding_space_id
from .jina_v4 import EMBEDDING_DIM, MAX_TOKENS, InputTooLargeError, JinaV4Embedder
from .registry import RegisteredModel
from .snapshot import ArtifactChangedError

PROMPTS = {"retrieval.passage": "Passage: ", "retrieval.query": "Query: "}
TOKENIZER_FILES = frozenset({"config.json", "tokenizer_config.json", "tokenizer.json",
                             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"})
MAX_INPUTS = 128
MAX_REQUEST_TOKENS = 262144


class ReceiptMismatchError(ValueError):
    """The request is not bound to this provider's current loaded generation."""


class BehaviorChangedError(ValueError):
    """The loaded adapter no longer implements its observed numerical contract."""


def _digest(value) -> str:
    """Hash bounded canonical runtime metadata without filesystem locations."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > 32 * 1024 * 1024:
        raise ValueError("runtime behavior metadata exceeds 32 MiB")
    return hashlib.sha256(encoded).hexdigest()


def _portable_config(value):
    """Remove acquisition provenance from configuration, preserving numerical settings."""
    if isinstance(value, dict):
        return {key: _portable_config(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "_commit_hash", "transformers_version"}}
    if isinstance(value, list):
        return [_portable_config(item) for item in value]
    return value


def _versions() -> dict:
    """Record numerical library versions; wheel device suffixes are operational."""
    result = {}
    for package in ("torch", "transformers", "tokenizers", "peft", "accelerate", "flash-attn", "triton"):
        try:
            result[package] = version(package).split("+", 1)[0]
        except PackageNotFoundError:
            result[package] = "absent"
    return result


class VerifiedJinaProvider:
    """Own one registered load with a fixed, conventional retrieval contract.

    Describe explicitly loads and observes the model. Embed requires a prior
    receipt and never silently loads/rebinds after unload. One lifecycle lock
    covers admission, all inputs, response checking, and unload.
    """

    def __init__(self, registration: RegisteredModel, *, device="cpu", use_fp16=False,
                 max_input_tokens: int = MAX_TOKENS):
        """Construct from operator-owned registration with an explicit serving budget."""
        if not isinstance(registration, RegisteredModel):
            raise ValueError("a trusted local model registration is required")
        if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= min(MAX_TOKENS, 32768):
            raise ValueError("token budget must fit the configured Jina serving ceiling")
        self._backend = JinaV4Embedder(model_name=registration.model_id, device=device,
                                     use_fp16=use_fp16, registration=registration)
        self._registration = registration
        self._max_input_tokens = max_input_tokens
        self._descriptor = None
        self._parameters = None

    @property
    def model_name(self):
        """Expose the registered model identifier without loading it."""
        return self._registration.model_id

    @property
    def is_loaded(self):
        """Expose load status for service ownership and idle cleanup."""
        return self._backend.is_loaded

    def unload(self):
        """Invalidate the observed receipt and release the checked serving copy."""
        with self._backend._artifact_lock:
            self._descriptor = None
            self._parameters = None
            self._backend.unload()

    def _parameter_seal(self):
        """Detect ordinary parameter replacement/in-place updates without hashing weights."""
        return [(name, id(p), p.data_ptr(), p._version, str(p.device))
                for name, p in self._backend._model.named_parameters()]

    def _derive(self):
        """Describe the actual loaded tokenizer, parameter precision, and raw-forward policy."""
        provenance = self._backend.loaded_artifact
        if provenance is None:
            raise ReceiptMismatchError("provider is not observed; verify before embedding")
        model = self._backend._model
        tokenizer = self._backend._tokenizer
        # This v1 adapter uses the standard Qwen2 fast tokenizer directly. It
        # does not execute a custom processor or accept an opaque tokenizer hook.
        cls = type(tokenizer)
        if (cls.__module__, cls.__name__) != (
            "transformers.models.qwen2.tokenization_qwen2_fast", "Qwen2TokenizerFast"
        ):
            raise BehaviorChangedError("verified text requires the standard Qwen2 fast tokenizer")
        if model.config.hidden_size != EMBEDDING_DIM or "retrieval" not in model.config.task_names:
            raise BehaviorChangedError("loaded model does not implement the Jina retrieval contract")
        if self._max_input_tokens > min(model.config.max_position_embeddings, tokenizer.model_max_length):
            raise BehaviorChangedError("serving budget exceeds loaded model/tokenizer context")
        if any(module.training for module in model.modules()):
            raise BehaviorChangedError("verified model must remain in evaluation mode")
        precision = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad or parameter.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                raise BehaviorChangedError("unsupported trainable or quantized model parameters")
            precision.append([name, str(parameter.dtype), list(parameter.shape)])
        if not precision:
            raise BehaviorChangedError("loaded model has no parameters")
        # Fast tokenizers mutate these two backend settings per call. This
        # adapter always supplies padding=False/truncation=False itself.
        tokenizer_state = json.loads(tokenizer.backend_tokenizer.to_str())
        tokenizer_state.pop("padding", None)
        tokenizer_state.pop("truncation", None)
        tokenizer_contract = _digest({"backend": tokenizer_state,
                                      "special_ids": tokenizer.all_special_ids})
        runtime = {"libraries": _versions(), "config": _portable_config(model.config.to_dict()),
                   "attention": getattr(model.config, "_attn_implementation", None),
                   "matmul_precision": torch.get_float32_matmul_precision(),
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                   "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                   "deterministic": torch.are_deterministic_algorithms_enabled()}
        tokenizer_files = tuple(file for file in self._registration.manifest.files if file.path in TOKENIZER_FILES)
        tokenizer_md5 = ArtifactManifest(tokenizer_files).digest()
        space = {
            "format": "hades-embedding-space-v1", "artifact_md5": provenance["manifest_md5"],
            "tokenizer_manifest_md5": tokenizer_md5,
            "tokenizer_contract": "qwen2-fast-no-truncation-v1:" + tokenizer_contract,
            "representation": {"dimension": EMBEDDING_DIM, "output": "dense-f32", "normalize": "l2",
                               "pooling": "attention-masked-mean-all-tokens-fp32-v1", "distance": "cosine",
                               "inference_precision": "no-autocast-parameters-sha256:" + _digest(sorted(precision)),
                               "implementation": "hades-jina-raw-text-v1:" + _digest(runtime)},
            "passage": {"task": "retrieval.passage", "prompt": PROMPTS["retrieval.passage"],
                        "adapter": "retrieval", "special_tokens": "qwen2-add-special-tokens-v1"},
            "query": {"task": "retrieval.query", "prompt": PROMPTS["retrieval.query"],
                      "adapter": "retrieval", "special_tokens": "qwen2-add-special-tokens-v1"},
        }
        receipt = ProviderReceipt(provenance["manifest_md5"], embedding_space_id(space),
                                  provenance["serving_generation"])
        if self._backend.loaded_artifact != provenance:
            raise BehaviorChangedError("loaded artifact changed while deriving behavior")
        return {"model": {"id": provenance["model_id"], "revision": provenance["revision"],
                          "manifest_md5": provenance["manifest_md5"],
                          "tokenizer_files": sorted(file.path for file in tokenizer_files)},
                "space": space, "receipt": receipt.to_dict(),
                "capabilities": {"text": True, "image_embedding": False, "specialized_code": False,
                                 "late_chunking": False, "tokenizer_offsets": False,
                                 "tasks": list(PROMPTS), "max_input_tokens": self._max_input_tokens}}

    def _checked(self):
        """Reject accidental artifact/runtime drift instead of minting replacement provenance."""
        if self._descriptor is None or not self._backend.is_loaded:
            raise ReceiptMismatchError("provider is not observed; verify before embedding")
        try:
            if self._parameter_seal() != self._parameters:
                raise BehaviorChangedError("loaded parameters changed; reverify the provider")
            current = self._derive()
            if current != self._descriptor:
                raise BehaviorChangedError("provider behavior changed; reverify the provider")
            return current
        except Exception as exc:
            self.unload()
            if isinstance(exc, (ArtifactChangedError, BehaviorChangedError)):
                raise
            raise BehaviorChangedError("loaded behavior cannot be verified") from exc

    def describe(self):
        """Explicitly observe a verified load and return a detached portable description."""
        with self._backend._artifact_lock:
            if not self._backend.is_loaded:
                self._descriptor = None
                try:
                    self._backend.model.eval()
                    self._descriptor = self._derive()
                    self._parameters = self._parameter_seal()
                except BaseException:
                    self.unload()
                    raise
            return json.loads(json.dumps(self._checked()))

    @contextmanager
    def _operation(self, expected):
        """Pin the actual load across the complete operation, including failure paths."""
        with self._backend._artifact_lock:
            current = self._checked()
            try:
                ProviderReceipt.from_dict(current["receipt"]).check_matches(expected)
            except ValueError as exc:
                raise ReceiptMismatchError("provider receipt changed; reverify binding") from exc
            try:
                yield current
            finally:
                self._checked()

    def embed(self, *, model: str, texts: list[str], task: str, expected: ProviderReceipt):
        """Return vectors only for the observed generation and explicit retrieval task."""
        if model != self._registration.model_id or task not in PROMPTS:
            raise ValueError("model and task must match the verified retrieval provider")
        if not isinstance(expected, ProviderReceipt):
            raise ValueError("an expected provider receipt is required")
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_INPUTS or any(not isinstance(t, str) for t in texts):
            raise ValueError("input must contain 1..=128 text strings")
        if sum(len(text.encode("utf-8")) for text in texts) > 8 * 1024 * 1024:
            raise ValueError("input exceeds text byte budget")
        with self._operation(expected) as observed:
            encoded_inputs = []
            total_tokens = 0
            for index, text in enumerate(texts):
                encoded = self._backend._tokenizer(
                    PROMPTS[task] + text, return_tensors="pt", add_special_tokens=True,
                    padding=False, truncation=False, return_attention_mask=True, split_special_tokens=False,
                )
                tokens = int(encoded["input_ids"].shape[1])
                if tokens > self._max_input_tokens:
                    raise InputTooLargeError(index, tokens, self._max_input_tokens)
                total_tokens += tokens
                if total_tokens > MAX_REQUEST_TOKENS:
                    raise ValueError("request exceeds total token budget")
                encoded_inputs.append(encoded)
            rows = []
            for index, encoded in enumerate(encoded_inputs):
                inputs = encoded.to(self._backend.device)
                with torch.no_grad(), torch.autocast(device_type=torch.device(self._backend.device).type, enabled=False):
                    output = self._backend._model(**inputs, task_label="retrieval", output_vlm_last_hidden_states=True)
                    hidden = output.vlm_last_hidden_states
                    if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, inputs["input_ids"].shape[1], EMBEDDING_DIM):
                        raise RuntimeError("model returned invalid token-level embeddings")
                    mask = inputs["attention_mask"].float().unsqueeze(-1)
                    pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
                    norm = torch.linalg.vector_norm(pooled, dim=-1, keepdim=True)
                    if not torch.isfinite(pooled).all() or not torch.isfinite(norm).all() or not (norm > 0).all():
                        raise RuntimeError("model returned invalid pooled embeddings")
                    vector = (pooled / norm).cpu().numpy().astype("float32", copy=False)
                rows.append({"object": "embedding", "index": index, "embedding": vector[0].tolist()})
            return {"object": "list", "model": self._registration.model_id, "data": rows,
                    "receipt": dict(observed["receipt"]),
                    "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}}
