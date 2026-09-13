"""Write the WeaverTools conformance extraction into a HADES graph.

The extractor in this package computes assertions, terms, axioms and the
`cites` edges that join code to the documents claiming it, and deliberately
imports no database driver: it is a pure function over a repository. This is the
other half, kept separate for the same reason.

**Why this exists.** `hades ingest` builds two halves of a graph and no bridge:
code files with symbols and call edges, documents with chunks and vectors, and
nothing between them. A query can find code and can find prose, and cannot ask
whether the code satisfies what the prose claims. The `//! conforms:` headers are
the join, 478 of them across this corpus, and nothing in HADES reads them.

**The join attaches to HADES's own file nodes.** A `cites` edge runs from
`codebase_files/<file_key>` to `wt_assertions/<key>`, not from a private copy of
the source list, so the result is one graph rather than three collections sharing
a database. That is also the failure this is written to avoid: two halves that
are co-resident and unjoined look identical to a joined graph until someone
traverses.

**Notes are written, not printed.** The extractor reports 48 sources owing a
header, a malformed node id, and tags naming instruments nothing cites. Left in
a terminal those become a graph that reads as complete. They land in
`wt_ingest_report` so a coverage query can find them.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weavertools.extractor import ingest  # noqa: E402

# One collection per node kind and one per relation, which is the notation the
# 2026-08-08 graph used. The content does not migrate (its rows are stale against
# 413 assertions where it held 242) but the shape is worth keeping.
NODE_COLLECTIONS = {
    "assertion": "wt_assertions",
    "document": "wt_documents",
    "vocabulary": "wt_vocabulary",
    "crate": "wt_crates",
    "term": "wt_terms",
    "axiom": "wt_axioms",
    "artifact": "wt_artifacts",
    "system": "wt_systems",
}
REPORT = "wt_ingest_report"

# `source` nodes are deliberately absent: a source file already exists in the
# graph as `codebase_files`, put there by `hades ingest`, and duplicating it
# would leave two nodes for one file and a join that proves nothing.
CODE_FILES = "codebase_files"


def arango(db: str, path: str, body=None, method="POST"):
    url = f"http://127.0.0.1:8529/_db/{db}/_api/{path}"
    data = None
    if body is not None:
        data = ("\n".join(json.dumps(d) for d in body) if isinstance(body, list)
                else json.dumps(body)).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}")


def key_for(ident: str) -> str:
    """An ArangoDB key for an extractor ident, keeping the ident readable.

    ArangoDB keys permit letters, digits and `_ - . @ ( ) + , = ; $ ! * ' %`, so
    a slash or a space has to go. Replaced rather than hashed, because a key a
    human can read is worth more here than one that round-trips.
    """
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", ident)


def file_key(rel_path: str) -> str:
    """The key `hades ingest` gives a file, so an edge can point at its node."""
    return rel_path.replace(".", "_").replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--repo", default="/opt/weavertools/WeaverTools")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs, code = ingest(Path(args.repo))

    # Collections first, ignoring "duplicate name" so a re-run is safe.
    relations = {e.relation for e in docs.edges} | {e.relation for e in code.edges}
    edge_collections = {rel: f"wt_{rel.replace('-', '_')}_edges" for rel in relations}
    if not args.dry_run:
        for name in list(NODE_COLLECTIONS.values()) + [REPORT]:
            arango(args.db, "collection", {"name": name, "type": 2})
        for name in edge_collections.values():
            arango(args.db, "collection", {"name": name, "type": 3})

    # Nodes.
    by_collection: dict[str, list[dict]] = collections.defaultdict(list)
    for node in docs.nodes:
        target = NODE_COLLECTIONS.get(node.kind)
        if target is None:
            print(f"  no collection for node kind {node.kind!r}, skipping", file=sys.stderr)
            continue
        by_collection[target].append({
            "_key": key_for(node.ident), "ident": node.ident, "kind": node.kind,
            "path": node.path, "line": node.line, "title": node.title,
            "body": node.body, "tag": node.tag, "lang": node.lang,
            "basis": "declared",
        })

    # Edges. A cites edge's source is a file HADES already ingested.
    missing_sources: list[str] = []
    for name, edges in (("documents", docs.edges), ("code", code.edges)):
        for e in edges:
            if e.relation == "cites":
                src_id = f"{CODE_FILES}/{file_key(e.src)}"
            else:
                src_id = f"{NODE_COLLECTIONS.get('assertion')}/{key_for(e.src)}"
                for kind, coll in NODE_COLLECTIONS.items():
                    if any(n.ident == e.src and n.kind == kind for n in docs.nodes):
                        src_id = f"{coll}/{key_for(e.src)}"
                        break
            dst_id = None
            for kind, coll in NODE_COLLECTIONS.items():
                if any(n.ident == e.dst and n.kind == kind for n in docs.nodes):
                    dst_id = f"{coll}/{key_for(e.dst)}"
                    break
            if dst_id is None:
                dst_id = f"{NODE_COLLECTIONS['assertion']}/{key_for(e.dst)}"
            by_collection[edge_collections[e.relation]].append({
                "_key": key_for(f"{e.src}--{e.relation}--{e.dst}")[:254],
                "_from": src_id, "_to": dst_id,
                "relation": e.relation, "basis": e.basis,
            })

    print(f"{'collection':36} {'documents':>10}")
    for name in sorted(by_collection):
        print(f"  {name:34} {len(by_collection[name]):>10,}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    for name, rows in sorted(by_collection.items()):
        resp = arango(args.db, f"import?collection={name}&type=documents&onDuplicate=replace",
                      rows)
        if resp.get("errors"):
            print(f"  {name}: {resp.get('created', 0)} created, "
                  f"{resp.get('errors')} errors, {resp.get('details', [])[:2]}",
                  file=sys.stderr)

    # What the extraction could not vouch for, written where a query can find it.
    arango(args.db, f"document/{REPORT}", {
        "_key": "latest",
        "repo": args.repo,
        "document_notes": docs.notes,
        "code_notes": code.notes,
        "dangling_documents": [vars(e) for e in docs.dangling],
        "dangling_code": [vars(e) for e in code.dangling],
        "warning": "counts in *_notes describe claims the extraction could not "
                   "resolve. A coverage query that ignores them will read "
                   "unenforced claims as enforced.",
    })
    print(f"\nwrote {sum(len(v) for v in by_collection.values()):,} documents plus one report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
