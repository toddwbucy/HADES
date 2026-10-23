"""Bounded conventional passage planning using exact counts of each candidate.

Passage/overlap settings are upper bounds, not promises of maximal packing.
Token counts need not be monotone under substring growth (BPE/NFC can merge).
Search only publishes candidates it actually counted; source bytes are unchanged.
"""

from array import array
import hashlib
import time

FORMAT = "hades-token-passages-v1"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_PASSAGES = 128
MAX_PASSAGE_BYTES = 4 * 1024 * 1024
MAX_COUNT_BYTES = 64 * 1024 * 1024
MAX_COUNT_CALLS = 8192


class PlanningError(ValueError):
    """No complete plan can be returned under the explicit policy/work limits."""


def plan_passages(text, *, passage_tokens, overlap_tokens, max_input_tokens, count,
                  minimum_overlap_tokens=0):
    """Return complete UTF-8 spans and independently counted body/input/overlap sizes.

    count(text, rendered) uses the serving tokenizer; rendered=True includes the
    passage prompt and special tokens. This function performs no model inference.
    """
    if (not isinstance(text, str) or not text.strip()
            or len(text.encode("utf-8")) > MAX_TEXT_BYTES):
        raise PlanningError("passage planning requires nonblank text of at most 2 MiB")
    if (any(type(v) is not int for v in (passage_tokens, overlap_tokens, max_input_tokens, minimum_overlap_tokens))
            or not 0 <= overlap_tokens < passage_tokens <= 0xffffffff
            or not 0 <= minimum_overlap_tokens <= overlap_tokens
            or not 1 <= max_input_tokens <= 0xffffffff):
        raise PlanningError("invalid conventional token policy")
    offsets = array("I", [0])
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    calls = scanned = 0
    deadline = time.monotonic() + 30

    def measured(start, end, rendered=False):
        """Bound tokenizer work before executing another exact substring count."""
        nonlocal calls, scanned
        calls += 1
        scanned += offsets[end] - offsets[start]
        if calls > MAX_COUNT_CALLS or scanned > MAX_COUNT_BYTES or time.monotonic() > deadline:
            raise PlanningError("passage planning work budget exceeded; split the source explicitly")
        value = count(text[start:end], rendered)
        if time.monotonic() > deadline:
            raise PlanningError("passage planning work budget exceeded; split the source explicitly")
        if type(value) is not int or not 0 <= value <= 0xffffffff:
            raise PlanningError("tokenizer returned an invalid count")
        return value

    def candidate(start, end):
        """A candidate is admissible only after both actual token counts fit."""
        body = measured(start, end)
        rendered = measured(start, end, True)
        if body <= passage_tokens and 0 < rendered <= max_input_tokens and text[start:end].strip():
            return body, rendered
        return None

    def fitting(start, minimum_end, proposed_end):
        """Find an observed fitting prefix; never infer that untested lengths fit."""
        end = proposed_end
        high = None
        result = candidate(start, end)
        while result is None and end > minimum_end:
            high = end
            end = minimum_end + (end - minimum_end) // 2
            result = candidate(start, end)
        if result is None:
            return None
        if high is not None:
            # Conservative refinement, not a claim of monotone or maximal token packing.
            while high - end > 1:
                middle = (end + high) // 2
                found = candidate(start, middle)
                if found is None:
                    high = middle
                else:
                    end, result = middle, found
        return end, result

    def overlap_start(start, end):
        """Choose a checked suffix bounded by overlap_tokens, with forward progress."""
        if overlap_tokens == 0:
            return end
        low, high = start + 1, end
        if measured(low, end) <= overlap_tokens:
            return low
        while high - low > 1:
            middle = (low + high) // 2
            if measured(middle, end) <= overlap_tokens:
                high = middle
            else:
                low = middle
        return high

    spans = []
    start = covered = total_bytes = 0
    hint = len(text)
    while covered < len(text):
        if len(spans) == MAX_PASSAGES:
            raise PlanningError("document exceeds 128 passages; split the source explicitly")
        # Grow the next proposal from the last fitted span instead of repeatedly
        # tokenizing the entire remaining document. Exact checks decide acceptance.
        end = min(len(text), max(covered + 1, start + hint))
        found = fitting(start, covered + 1, end)
        while found is None and start < covered:
            start = min(covered, start + max(1, (covered - start) // 2))
            # Less overlap is allowed, but it must be measured again: NFC/BPE
            # can make even a shorter substring use more tokens.
            if measured(start, covered) > overlap_tokens:
                continue
            found = fitting(start, covered + 1, end)
        if found is None:
            raise PlanningError("cannot fit a nonblank passage with forward progress under token limits")
        end, (body, rendered) = found
        overlap = measured(start, covered) if start < covered else 0
        if overlap > overlap_tokens:
            raise PlanningError("chosen overlap exceeds token policy")
        if spans and overlap < minimum_overlap_tokens:
            raise PlanningError("cannot retain the minimum window overlap within the serving budget")
        total_bytes += offsets[end] - offsets[start]
        if total_bytes > MAX_PASSAGE_BYTES:
            raise PlanningError("overlapping passages exceed 4 MiB")
        spans.append({"start_byte": offsets[start], "end_byte": offsets[end],
                      "body_tokens": body, "input_tokens": rendered, "overlap_tokens": overlap})
        hint = 2 * (end - start)
        covered = end
        if covered < len(text):
            start = overlap_start(start, end)
    return {"format": FORMAT, "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "passage_tokens": passage_tokens, "overlap_tokens": overlap_tokens,
            "max_input_tokens": max_input_tokens, "spans": spans}
