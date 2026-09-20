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
import base64
import collections
import json
import os
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

# The same argument, for the markdown. A `declared-in` edge ends at the
# `documents` row the ingest created, so `wt_documents` holds only the 13 records
# the corpus declares as `kind: document` and not one minted node per file.
DOC_FILES = "documents"


def _endpoint() -> str:
    """Where ArangoDB is, from the same environment the CLI reads.

    TCP rather than the Unix socket the Rust client prefers, because `urllib`
    cannot speak a Unix socket and this adapter has no other HTTP client. Host
    and port come from `ARANGO_HOST` and `ARANGO_PORT` so the endpoint is not
    hardcoded, and credentials from `ARANGO_USERNAME` and `ARANGO_PASSWORD`, so
    an instance with authentication enabled is reachable and one with it disabled
    keeps working unchanged.
    """
    host = os.environ.get("ARANGO_HOST", "127.0.0.1")
    port = os.environ.get("ARANGO_PORT", "8529")
    return f"http://{host}:{port}"


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _auth_header() -> dict[str, str]:
    """Basic auth, and only where it cannot be read off the wire.

    The endpoint is plain HTTP, so credentials sent to anything but loopback
    cross the network in the clear. Refused rather than sent: a script that
    quietly leaks a database password is worse than one that stops. Point it at a
    loopback endpoint and tunnel if the instance is remote.
    """
    password = os.environ.get("ARANGO_PASSWORD")
    if not password:
        return {}
    host = os.environ.get("ARANGO_HOST", "127.0.0.1")
    if host not in LOOPBACK:
        raise SystemExit(
            f"refusing to send ARANGO_PASSWORD to {host} over plain HTTP. "
            f"Use a loopback endpoint (an SSH tunnel, for instance) or unset the "
            f"password if the instance has authentication disabled."
        )
    user = os.environ.get("ARANGO_USERNAME", "root")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep credentials and writes at the explicitly configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def arango(db: str, path: str, body=None, method="POST"):
    """One request, returning the parsed body whether it succeeded or not.

    The caller checks. An HTTPError body carries ArangoDB's own `errorMessage`,
    which is more use than the exception, and several calls here are expected to
    fail (creating a collection that exists).
    """
    url = f"{_endpoint()}/_db/{db}/_api/{path}"
    data = None
    if body is not None:
        data = ("\n".join(json.dumps(d) for d in body) if isinstance(body, list)
                else json.dumps(body)).encode()
    headers = {"Content-Type": "application/json"} | _auth_header()
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.build_opener(_RejectRedirects()).open(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        with e:
            if 300 <= e.code < 400:
                return {"error": True, "code": e.code,
                        "errorMessage": "HTTP redirect refused by adapter endpoint policy"}
            return json.loads(e.read().decode() or "{}")


def code_keys_by_path(db: str) -> dict[str, str]:
    """Repo-relative path -> the `codebase_files` key the ingest gave it.

    Two jobs. It **bounds the conformance pass** to the files the graph holds, so a
    citation in a file the ingest never took is reported rather than turned into an
    edge with no source. And it supplies the key instead of re-deriving it, so the
    `_from` of every `cites` edge is the key the ingest wrote rather than this
    module's guess at how the ingest writes keys. The document half was re-deriving
    its key the same way until the path-and-key mismatch it invites was found, and
    this is the other half of that.
    """
    resp = arango(db, "cursor", {
        "query": "FOR f IN codebase_files RETURN [f.path, f._key]",
        "batchSize": 5000,
    })
    pairs = list(resp.get("result") or [])
    while resp.get("hasMore"):
        resp = arango(db, f"cursor/{resp['id']}", method="PUT")
        pairs.extend(resp.get("result") or [])
    return {path: key for path, key in pairs if path}


def document_keys_by_rel(db: str) -> dict[str, str]:
    """Relative path -> the `_key` `hades ingest` gave that document.

    A `declared-in` edge points at the markdown that declares the record, and
    that markdown is already in the graph as a `documents` row with its text, its
    chunks and a vector. Looked up by `source_rel` rather than re-deriving the
    key, because the derivation is the CLI's (`derive_doc_key` drops the
    extension, normalizes separators and strips a trailing version suffix) and a
    second implementation of it here would agree until it did not. A path absent
    from this map is reported, not guessed at.
    """
    resp = arango(db, "cursor", {
        "query": "FOR d IN documents FILTER d.source_rel != null "
                 "RETURN [d.source_rel, d._key]",
        "batchSize": 5000,
    })
    pairs = list(resp.get("result") or [])
    while resp.get("hasMore"):
        resp = arango(db, f"cursor/{resp['id']}", method="PUT")
        pairs.extend(resp.get("result") or [])
    return {rel: key for rel, key in pairs}


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

    # The scopes come from the graph, and the extraction is bounded by them. This
    # is the single decision about what is in play: `hades ingest` made it from
    # `.hadesignore`, and this reads the result rather than re-deciding.
    doc_keys = document_keys_by_rel(args.db)
    code_keys = code_keys_by_path(args.db)
    if not doc_keys and not code_keys:
        print(
            f"{args.db} holds no codebase_files and no documents rows, so nothing "
            f"is in scope. Run `hades ingest` over the tree first: a conformance "
            f"pass against an empty graph writes edges whose endpoints do not "
            f"exist.",
            file=sys.stderr,
        )
        return 1

    docs, code = ingest(Path(args.repo), set(doc_keys), set(code_keys))

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

    # Edges. A cites edge's source is a file HADES already ingested, so the node
    # has to be there. `.toml` is the case that makes this more than a formality:
    # the extractor reads manifests for citations while `ingest_routing` leaves
    # `.toml` unrouted, so a plain `hades ingest` creates no node for one and the
    # edge would point at nothing. `hades ingest --unparsed-ext toml` creates it.
    missing_sources: list[str] = []
    missing_documents: list[str] = []
    # One pass over the node list instead of one per edge endpoint. The lookups
    # below ran `any()` over 491 nodes twice for each of 1,637 edges.
    kind_of = {n.ident: n.kind for n in docs.nodes}
    for name, edges in (("documents", docs.edges), ("code", code.edges)):
        for e in edges:
            if e.relation == "cites":
                # Scoping should make this unreachable. Kept as an assertion
                # rather than removed: it is the check that would catch the
                # scope and the graph parting company again.
                key = code_keys.get(e.src)
                if key is None:
                    missing_sources.append(e.src)
                    key = file_key(e.src)
                src_id = f"{CODE_FILES}/{key}"
            else:
                src_coll = NODE_COLLECTIONS.get(kind_of.get(e.src), NODE_COLLECTIONS["assertion"])
                src_id = f"{src_coll}/{key_for(e.src)}"
            if e.relation == "declared-in":
                # The target is a path, and the node it names is the ingested
                # markdown rather than anything this adapter writes.
                doc_key = doc_keys.get(e.dst)
                if doc_key is None:
                    missing_documents.append(e.dst)
                    doc_key = key_for(e.dst)
                dst_id = f"{DOC_FILES}/{doc_key}"
            else:
                dst_coll = NODE_COLLECTIONS.get(kind_of.get(e.dst), NODE_COLLECTIONS["assertion"])
                dst_id = f"{dst_coll}/{key_for(e.dst)}"
            # `via` is in the key because it is part of the edge's identity: see
            # the note on `Edge.via`. Three seams between one pair of crates
            # collapsed into one row without it.
            identity = f"{e.src}--{e.relation}--{e.dst}"
            if e.via:
                identity = f"{identity}--via-{e.via}"
            by_collection[edge_collections[e.relation]].append({
                "_key": key_for(identity)[:254],
                "_from": src_id, "_to": dst_id,
                "relation": e.relation, "basis": e.basis,
                "via": e.via, "tag": e.tag,
            })

    # The scope is a decision the graph made, so what it excluded is printed
    # rather than left in a report nobody opens. A file holding declarations that
    # nothing read is the one case where a low count looks like a clean run.
    excluded = [n for n in docs.notes + code.notes if n.startswith("out of scope")]
    if excluded:
        print(
            f"\n{len(excluded)} file(s) hold declarations or citations and are not "
            f"in the graph, so they were not read:",
            file=sys.stderr,
        )
        for note in excluded[:10]:
            print(f"    {note.split(': ', 1)[-1]}", file=sys.stderr)
        print(
            "    These are excluded by .hadesignore or were never ingested. That is "
            "a scope decision, not an error, but it is why a count may be lower "
            "than the tree suggests.",
            file=sys.stderr,
        )

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
    # overwriteMode=replace, because a fixed `_key` POSTs into a 409 on every run
    # after the first and this function returns the error body rather than
    # raising, so the second run would have silently kept the first run's notes.
    report = arango(args.db, f"document/{REPORT}?overwriteMode=replace", {
        "_key": "latest",
        "repo": args.repo,
        "document_notes": docs.notes,
        "code_notes": code.notes,
        "dangling_documents": [vars(e) for e in docs.dangling],
        "dangling_code": [vars(e) for e in code.dangling],
        "cites_sources_with_no_file_node": missing_sources,
        "declared_in_targets_with_no_document_row": sorted(set(missing_documents)),
        "warning": "counts in *_notes describe claims the extraction could not "
                   "resolve. A coverage query that ignores them will read "
                   "unenforced claims as enforced.",
    })
    if report.get("error"):
        print(f"  report write failed: {report.get('errorMessage')}", file=sys.stderr)
    if missing_sources:
        print(
            f"\n{len(missing_sources)} cites edge(s) name a file with no graph node, "
            f"so they dangle. Ingest those files first, e.g. with --unparsed-ext:",
            file=sys.stderr,
        )
        for src in sorted(set(missing_sources))[:10]:
            print(f"    {src}", file=sys.stderr)
    if missing_documents:
        print(
            f"\n{len(set(missing_documents))} declared-in edge(s) name a markdown file "
            f"with no `documents` row, so they dangle. Run the ingest over the tree "
            f"first:",
            file=sys.stderr,
        )
        for rel in sorted(set(missing_documents))[:10]:
            print(f"    {rel}", file=sys.stderr)
    print(f"\nwrote {sum(len(v) for v in by_collection.values()):,} documents plus one report")
    return 1 if missing_sources or missing_documents or report.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
