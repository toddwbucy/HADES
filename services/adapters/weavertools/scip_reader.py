"""Minimal SCIP index reader, enough to prove the alignment path works.

SCIP is protobuf. Rather than take a protobuf dependency to answer "is this
index usable", this walks the wire format directly for the three fields that
matter:

    Index.documents            = 2   (repeated Document)
    Document.relative_path     = 1   (string)
    Document.symbols           = 3   (repeated SymbolInformation)
    SymbolInformation.symbol   = 1   (string)

Everything else is skipped by wire type. If this can enumerate documents and
symbols, the real ingest can too, and the only thing a proper parser buys is
convenience.
"""

import sys
from collections import Counter
from pathlib import Path

INDEX = Path(sys.argv[1] if len(sys.argv) > 1 else "weavertools.scip")


def varint(buf, i):
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def fields(buf, start, end):
    """Yield (field_number, wire_type, payload_or_value, next_index)."""
    i = start
    while i < end:
        key, i = varint(buf, i)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:
            val, i = varint(buf, i)
            yield fnum, wtype, val
        elif wtype == 2:
            ln, i = varint(buf, i)
            yield fnum, wtype, buf[i : i + ln]
            i += ln
        elif wtype == 5:
            yield fnum, wtype, buf[i : i + 4]
            i += 4
        elif wtype == 1:
            yield fnum, wtype, buf[i : i + 8]
            i += 8
        else:
            raise ValueError(f"wire type {wtype} at {i}")


def main():
    buf = INDEX.read_bytes()
    print(f"index: {INDEX.name}  {len(buf) / 1048576:.1f} MB\n")

    docs = 0
    symbols = 0
    paths = []
    by_crate = Counter()
    kernel_hits = []

    for fnum, wtype, payload in fields(buf, 0, len(buf)):
        if fnum != 2 or wtype != 2:
            continue  # not Index.documents
        docs += 1
        path = None
        doc_syms = 0
        for f2, w2, p2 in fields(payload, 0, len(payload)):
            if f2 == 1 and w2 == 2:
                path = p2.decode("utf-8", "replace")
            elif f2 == 3 and w2 == 2:
                doc_syms += 1
                if doc_syms == 1:
                    for f3, w3, p3 in fields(p2, 0, len(p2)):
                        if f3 == 1 and w3 == 2:
                            break
        symbols += doc_syms
        if path:
            paths.append((path, doc_syms))
            parts = path.split("/")
            if len(parts) > 1 and parts[0] == "crates":
                by_crate[parts[1]] += doc_syms
            if path.endswith(".cu") or "kernels" in path:
                kernel_hits.append(path)

    print(f"documents indexed : {docs}")
    print(f"symbols defined   : {symbols}\n")

    print("symbols per crate:")
    for crate, n in by_crate.most_common():
        print(f"  {crate:<22} {n}")

    print(f"\nlargest documents:")
    for path, n in sorted(paths, key=lambda p: -p[1])[:6]:
        print(f"  {n:>5}  {path}")

    print(f"\nCUDA / kernel files in the index: {kernel_hits or 'none'}")
    print(
        "\nVERDICT: index is walkable and the alignment path is unblocked."
        if docs and symbols
        else "\nVERDICT: index did not parse as expected."
    )
    return 0 if docs and symbols else 1


if __name__ == "__main__":
    raise SystemExit(main())
