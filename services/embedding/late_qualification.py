"""Independent tokenizer recounts of plans and token coordinates observed over Rust HTTP."""
import hashlib

from .identity import ProviderReceipt
from .qualification import require
from .verified_jina import PROMPTS


def recheck(provider, text, report):
    """Use direct tokenizer calls, without the planner or pooling boundary helpers."""
    plan = report["plan"]
    raw = text.encode("utf-8")
    require(hashlib.sha256(raw).hexdigest() == plan["source_sha256"], "source changed")
    expected = ProviderReceipt.from_dict(plan["receipt"])
    with provider._operation(expected):
        tokenizer = provider._backend._tokenizer

        def encode(value, rendered, offsets=False):
            return tokenizer((PROMPTS["retrieval.passage"] if rendered else "") + value,
                             add_special_tokens=rendered, padding=False, truncation=False,
                             split_special_tokens=False, return_offsets_mapping=offsets)

        for family in ("passages", "windows"):
            previous = None
            for span in plan[family]:
                start, end = span["start_byte"], span["end_byte"]
                value = raw[start:end].decode("utf-8")
                require(value.strip() and len(encode(value, False)["input_ids"]) == span["body_tokens"],
                        "body token count differs")
                require(len(encode(value, True)["input_ids"]) == span["input_tokens"] <= plan["max_input_tokens"],
                        "rendered token count differs")
                if previous is None:
                    require(start == 0 and span["overlap_tokens"] == 0, "invalid first span")
                else:
                    require(previous["start_byte"] < start <= previous["end_byte"] < end, "source coverage changed")
                    overlap = raw[start:previous["end_byte"]].decode("utf-8")
                    require(len(encode(overlap, False)["input_ids"]) == span["overlap_tokens"], "overlap count differs")
                    if family == "windows":
                        require(span["overlap_tokens"] >= plan["minimum_window_overlap_tokens"], "minimum overlap lost")
                previous = span
            require(previous is not None and previous["end_byte"] == len(raw), "source tail lost")
        ranges = report["token_ranges"]
        require(len(ranges) == len(plan["assignments"]) == len(plan["passages"]), "incomplete token ranges")
        prefix = len(PROMPTS["retrieval.passage"])
        for passage, index, observed in zip(plan["passages"], plan["assignments"], ranges):
            window = plan["windows"][index]
            require(window["start_byte"] <= passage["start_byte"] < passage["end_byte"] <= window["end_byte"],
                    "passage outside its window")
            value = raw[window["start_byte"]:window["end_byte"]].decode("utf-8")
            start = prefix + len(raw[window["start_byte"]:passage["start_byte"]].decode("utf-8"))
            end = prefix + len(raw[window["start_byte"]:passage["end_byte"]].decode("utf-8"))
            offsets = encode(value, True, True)["offset_mapping"]
            selected = [i for i, (left, right) in enumerate(offsets) if left < right and left < end and right > start]
            require(selected and selected == list(range(observed["token_start"], observed["token_end"])),
                    "returned token coordinates differ from actual original-character offsets")
    return {"passed": True, "checked_passages": len(ranges), "checked_windows": len(plan["windows"]),
            "source_bytes": len(raw)}
