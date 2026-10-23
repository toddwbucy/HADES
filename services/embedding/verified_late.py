"""Opt-in verified Jina contextual passage pooling; conventional spaces stay unchanged."""
import hashlib
import time

import torch

from .identity import ProviderReceipt, embedding_space_id
from .jina_v4 import EMBEDDING_DIM, InputTooLargeError
from .passage_plan import (MAX_COUNT_BYTES, MAX_COUNT_CALLS, MAX_PASSAGES,
                           MAX_TEXT_BYTES, PlanningError, plan_passages)
from .verified_jina import PROMPTS, VerifiedJinaProvider


class VerifiedJinaLateProvider(VerifiedJinaProvider):
    """A distinct space: full-input query means and contextual passage token means.

    Offsets are original Python character coordinates from the approved fast
    tokenizer. A passage pools the contiguous tokens whose nonempty offsets
    intersect its source range; prompts and special tokens outside that range
    are excluded. Context windows are encoded once for their assigned passages.
    """

    def _derive(self):
        """Give contextual pooling its own identity without relabeling conventional vectors."""
        descriptor = super()._derive()
        representation = descriptor["space"]["representation"]
        representation["pooling"] = "query-all-tokens-passage-intersecting-window-tokens-mean-fp32-v1"
        representation["implementation"] += ":late-passages-v1"
        descriptor["capabilities"].update(late_chunking=True, tokenizer_offsets=True)
        descriptor["receipt"]["space_id"] = embedding_space_id(descriptor["space"])
        return descriptor

    def embed(self, *, model, texts, task, expected):
        """Queries use the declared full-input mean; passage fallback is refused."""
        if task != "retrieval.query":
            raise ValueError("this space requires contextual passage pooling")
        return super().embed(model=model, texts=texts, task=task, expected=expected)

    def plan(self, **kwargs):
        """Conventional passage planning cannot select this contextual workload."""
        raise ValueError("this space requires an explicit late passage policy")

    def _admit(self, model, task, expected):
        """Require the registered passage task and a caller's exact observation."""
        if model != self.model_name or task != "retrieval.passage" or not isinstance(expected, ProviderReceipt):
            raise ValueError("late passage work requires the registered model, passage task and receipt")

    def plan_late(self, *, model, text, task, expected, passage_tokens, overlap_tokens,
                  context_cap_tokens, minimum_window_overlap_tokens, max_input_tokens):
        """Plan complete token-checked passages and covering encoding windows before inference."""
        self._admit(model, task, expected)
        values = (passage_tokens, overlap_tokens, context_cap_tokens,
                  minimum_window_overlap_tokens, max_input_tokens)
        if (any(type(v) is not int for v in values)
                or not 0 <= overlap_tokens < passage_tokens <= context_cap_tokens <= max_input_tokens <= 0xffffffff
                or not 0 <= minimum_window_overlap_tokens < context_cap_tokens):
            raise ValueError("invalid late passage/window token policy")
        with self._operation(expected) as observed:
            ceiling = min(context_cap_tokens, max_input_tokens, self._max_input_tokens)
            if minimum_window_overlap_tokens >= ceiling:
                raise PlanningError("running token ceiling cannot retain the minimum window overlap")
            calls = scanned = 0
            deadline = time.monotonic() + 30

            def count(value, rendered):
                """Share one work budget across both planning passes and count exact prompts."""
                nonlocal calls, scanned
                calls += 1
                scanned += len(value.encode("utf-8"))
                if calls > MAX_COUNT_CALLS or scanned > MAX_COUNT_BYTES or time.monotonic() > deadline:
                    raise PlanningError("late planning work budget exceeded; split the source explicitly")
                encoded = self._backend._tokenizer(
                    (PROMPTS[task] if rendered else "") + value,
                    add_special_tokens=rendered, padding=False, truncation=False,
                    split_special_tokens=False, return_attention_mask=False,
                    return_token_type_ids=False, verbose=False,
                )
                return len(encoded["input_ids"])

            passages = plan_passages(text, passage_tokens=passage_tokens, overlap_tokens=overlap_tokens,
                                     max_input_tokens=ceiling, count=count)
            # Extra overlap headroom accommodates tokenizer boundaries without
            # silently weakening the declared minimum. Counts are checked again.
            upper_overlap = min(ceiling - 1, minimum_window_overlap_tokens + passage_tokens)
            windows = plan_passages(text, passage_tokens=ceiling, overlap_tokens=upper_overlap,
                                    minimum_overlap_tokens=minimum_window_overlap_tokens,
                                    max_input_tokens=ceiling, count=count)
            assignments = []
            for passage in passages["spans"]:
                candidates = [(abs(window["start_byte"] + window["end_byte"]
                                   - passage["start_byte"] - passage["end_byte"]), index)
                              for index, window in enumerate(windows["spans"])
                              if window["start_byte"] <= passage["start_byte"]
                              and passage["end_byte"] <= window["end_byte"]]
                if not candidates:
                    raise PlanningError("no encoding window covers a passage; explicitly increase window overlap or reduce passage size")
                assignments.append(min(candidates)[1])
            return {"format": "hades-late-passages-v1", "source_sha256": passages["source_sha256"],
                    "model": self.model_name, "task": task, "receipt": dict(observed["receipt"]),
                    "passage_tokens": passage_tokens, "overlap_tokens": overlap_tokens,
                    "context_cap_tokens": context_cap_tokens,
                    "minimum_window_overlap_tokens": minimum_window_overlap_tokens,
                    "max_input_tokens": ceiling, "passages": passages["spans"],
                    "windows": windows["spans"], "assignments": assignments}

    def embed_late(self, *, model, text, task, boundaries, expected):
        """Encode one bounded window and pool only its explicitly assigned source ranges."""
        self._admit(model, task, expected)
        if (not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > MAX_TEXT_BYTES
                or not isinstance(boundaries, list) or not 1 <= len(boundaries) <= MAX_PASSAGES):
            raise ValueError("invalid late window or boundary count")
        for span in boundaries:
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or any(type(n) is not int for n in span) or not 0 <= span[0] < span[1] <= len(text)
                    or not text[span[0]:span[1]].strip()):
                raise ValueError("late boundaries must be nonblank original-character ranges")
        with self._operation(expected) as observed:
            prefix = PROMPTS[task]
            rendered = prefix + text
            encoded = self._backend._tokenizer(
                rendered, return_tensors="pt", add_special_tokens=True, padding=False,
                truncation=False, return_attention_mask=True, return_offsets_mapping=True,
                split_special_tokens=False,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            tokens = int(encoded["input_ids"].shape[1])
            if tokens > self._max_input_tokens:
                raise InputTooLargeError(0, tokens, self._max_input_tokens)
            if len(offsets) != tokens:
                raise ValueError("tokenizer offsets do not match the complete input")
            previous = (0, 0)
            for start, end in offsets:
                if start == end == 0:
                    continue
                if not (0 <= start < end <= len(rendered) and start >= previous[0] and end >= previous[1]):
                    raise ValueError("tokenizer returned invalid original-character offsets")
                previous = (start, end)
            ranges = []
            for start, end in boundaries:
                selected = [i for i, (left, right) in enumerate(offsets)
                            if left < right and left < len(prefix) + end and right > len(prefix) + start]
                if not selected or selected != list(range(selected[0], selected[-1] + 1)):
                    raise ValueError("passage has no contiguous tokenizer offset coverage")
                ranges.append((selected[0], selected[-1] + 1))
            inputs = encoded.to(self._backend.device)
            with torch.no_grad(), torch.autocast(device_type=torch.device(self._backend.device).type, enabled=False):
                output = self._backend._model(**inputs, task_label="retrieval", output_vlm_last_hidden_states=True)
                hidden = output.vlm_last_hidden_states
                if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, tokens, EMBEDDING_DIM):
                    raise RuntimeError("model returned invalid token-level embeddings")
                rows = []
                for index, ((start, end), (left, right)) in enumerate(zip(boundaries, ranges)):
                    if not (inputs["attention_mask"][0, left:right] == 1).all():
                        raise ValueError("passage includes masked tokenizer positions")
                    pooled = hidden[0, left:right].float().mean(dim=0)
                    norm = torch.linalg.vector_norm(pooled)
                    if not torch.isfinite(pooled).all() or not torch.isfinite(norm) or norm <= 0:
                        raise RuntimeError("model returned invalid contextual passage embeddings")
                    rows.append({"index": index, "embedding": (pooled / norm).cpu().tolist(),
                                 "char_start": start, "char_end": end, "token_start": left, "token_end": right})
            return {"format": "hades-late-embedding-v1", "model": self.model_name,
                    "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "receipt": dict(observed["receipt"]), "input_tokens": tokens, "data": rows}
