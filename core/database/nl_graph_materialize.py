"""Materialize NL graph edges from embedded cross-reference fields.

Reads documents from NL vertex collections, extracts references from
fields defined in ``nl_graph_schema.EdgeCollectionDef``, and inserts
native ArangoDB edges into the corresponding edge collections.

The materializer is data-driven — it uses the schema definitions to
determine which collections to scan, which fields to extract, and where
to write edges. No edge-type-specific code is needed.

Special handling:
    - ``paper_edges`` collection uses paired ``from_node``/``to_node`` fields
    - ``chain`` fields produce sequential edges (chain[0]→chain[1]→...)
    - Array fields produce one edge per element
    - String fields produce one edge

Usage:
    from core.database.nl_graph_materialize import NLGraphMaterializer

    materializer = NLGraphMaterializer(client, database="NL")
    stats = materializer.materialize_all()
    print(stats)

CLI:
    poetry run hades --database NL db graph materialize
    poetry run hades --database NL db graph materialize --edge nl_axiom_basis_edges
    poetry run hades --database NL db graph materialize --dry-run
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.database.nl_graph_schema import (
    CROSS_PAPER,
    LINEAGE_CHAIN,
    NL_GRAPH_SCHEMA,
    EdgeCollectionDef,
)

logger = logging.getLogger(__name__)


@dataclass
class MaterializeStats:
    """Statistics for a materialization run."""

    edges_created: int = 0
    edges_skipped: int = 0
    collections_scanned: int = 0
    collections_missing: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges_created": self.edges_created,
            "edges_skipped": self.edges_skipped,
            "collections_scanned": self.collections_scanned,
            "collections_missing": self.collections_missing,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 1),
        }


class NLGraphMaterializer:
    """Materializes edges from NL document cross-reference fields.

    Args:
        client: ArangoHttp2Client instance
        database: Target database name (default "NL")
    """

    def __init__(self, client: Any, database: str = "NL") -> None:
        self._client = client
        self._db = database

    def _existing_collections(self) -> set[str]:
        """Get set of collection names that exist in the database."""
        resp = self._client.request("GET", f"/_db/{self._db}/_api/collection")
        if resp.get("error"):
            raise RuntimeError(f"Failed to list collections: {resp.get('errorMessage')}")
        return {c["name"] for c in resp.get("result", [])}

    def _ensure_edge_collection(self, name: str, existing: set[str]) -> None:
        """Create an edge collection if it doesn't exist."""
        if name in existing:
            return
        resp = self._client.request(
            "POST",
            f"/_db/{self._db}/_api/collection",
            json={"name": name, "type": 3},
        )
        if resp.get("error") and resp.get("errorNum") != 1207:  # 1207 = duplicate name
            raise RuntimeError(f"Failed to create edge collection {name}: {resp.get('errorMessage')}")
        existing.add(name)
        logger.info("Created edge collection %s", name)

    def _scan_collection(self, collection: str, fields: list[str]) -> list[dict[str, Any]]:
        """Scan a collection returning docs that have at least one of the given fields."""
        filter_clauses = " OR ".join(f"d.{f} != null" for f in fields)
        return_fields = ", ".join(f"{f}: d.{f}" for f in fields)
        aql = f"FOR d IN {collection} FILTER {filter_clauses} RETURN {{_id: d._id, _key: d._key, {return_fields}}}"
        return self._client.query(aql, batch_size=50000)

    def _resolve_ref(self, ref: str) -> str | None:
        """Resolve a reference to a full ArangoDB document ID.

        Handles both full IDs (collection/key) and bare keys.
        Returns None if the reference is invalid.
        """
        if not ref or not isinstance(ref, str):
            return None
        # Already a full ID
        if "/" in ref:
            return ref
        # Bare key — can't resolve without collection context
        return None

    def _build_edges_standard(
        self,
        edge_def: EdgeCollectionDef,
        existing: set[str],
        dry_run: bool = False,
    ) -> MaterializeStats:
        """Build edges for a standard edge definition (single or array field)."""
        stats = MaterializeStats()
        start = time.monotonic()

        from_collections = [c for c in edge_def.from_collections if c in existing]
        stats.collections_missing += len(edge_def.from_collections) - len(from_collections)

        edges: list[dict[str, Any]] = []

        for coll in from_collections:
            try:
                docs = self._scan_collection(coll, [edge_def.source_field])
            except Exception as e:
                stats.errors.append(f"{coll}: {e}")
                continue
            stats.collections_scanned += 1

            for doc in docs:
                from_id = doc["_id"]
                raw_refs = doc.get(edge_def.source_field)
                if raw_refs is None:
                    continue

                # Normalize to list
                if isinstance(raw_refs, list):
                    refs = raw_refs
                elif isinstance(raw_refs, str):
                    refs = [raw_refs]
                else:
                    continue

                for ref in refs:
                    to_id = self._resolve_ref(ref)
                    if to_id is None:
                        stats.edges_skipped += 1
                        continue

                    # Validate target collection exists
                    target_coll = to_id.split("/")[0]
                    if target_coll not in existing:
                        stats.edges_skipped += 1
                        continue

                    edge = {
                        "_from": from_id,
                        "_to": to_id,
                        "_key": f"{from_id.replace('/', '_')}__{to_id.replace('/', '_')}",
                        "source_field": edge_def.source_field,
                    }
                    edges.append(edge)

        if edges and not dry_run:
            self._ensure_edge_collection(edge_def.name, existing)
            try:
                result = self._insert_edges(edge_def.name, edges)
                stats.edges_created = result.get("created", 0)
                stats.edges_skipped += result.get("errors", 0)
            except Exception as e:
                stats.errors.append(f"Insert failed for {edge_def.name}: {e}")
        elif dry_run:
            stats.edges_created = len(edges)

        stats.duration_ms = (time.monotonic() - start) * 1000
        return stats

    def _build_edges_cross_paper(
        self,
        edge_def: EdgeCollectionDef,
        existing: set[str],
        dry_run: bool = False,
    ) -> MaterializeStats:
        """Build edges from paper_edges collection (paired from_node/to_node)."""
        stats = MaterializeStats()
        start = time.monotonic()

        if "paper_edges" not in existing:
            stats.collections_missing += 1
            stats.duration_ms = (time.monotonic() - start) * 1000
            return stats

        aql = "FOR d IN paper_edges FILTER d.from_node != null AND d.to_node != null RETURN d"
        try:
            docs = self._client.query(aql, batch_size=50000)
        except Exception as e:
            stats.errors.append(f"paper_edges: {e}")
            stats.duration_ms = (time.monotonic() - start) * 1000
            return stats

        stats.collections_scanned = 1
        edges: list[dict[str, Any]] = []

        for doc in docs:
            from_id = self._resolve_ref(doc.get("from_node", ""))
            to_id = self._resolve_ref(doc.get("to_node", ""))
            if not from_id or not to_id:
                stats.edges_skipped += 1
                continue

            from_coll = from_id.split("/")[0]
            to_coll = to_id.split("/")[0]
            if from_coll not in existing or to_coll not in existing:
                stats.edges_skipped += 1
                continue

            edge: dict[str, Any] = {
                "_from": from_id,
                "_to": to_id,
                "_key": doc["_key"],
                "source_field": "from_node/to_node",
            }
            for attr in edge_def.edge_attributes:
                if attr in doc:
                    edge[attr] = doc[attr]
            edges.append(edge)

        if edges and not dry_run:
            self._ensure_edge_collection(edge_def.name, existing)
            try:
                result = self._insert_edges(edge_def.name, edges)
                stats.edges_created = result.get("created", 0)
                stats.edges_skipped += result.get("errors", 0)
            except Exception as e:
                stats.errors.append(f"Insert failed for {edge_def.name}: {e}")
        elif dry_run:
            stats.edges_created = len(edges)

        stats.duration_ms = (time.monotonic() - start) * 1000
        return stats

    def _build_edges_lineage(
        self,
        edge_def: EdgeCollectionDef,
        existing: set[str],
        dry_run: bool = False,
    ) -> MaterializeStats:
        """Build sequential edges from lineage chain arrays.

        For a chain [A, B, C], creates edges: A→B and B→C.
        Also creates edges from the lineage doc to each chain member.
        """
        stats = MaterializeStats()
        start = time.monotonic()

        from_collections = [c for c in edge_def.from_collections if c in existing]
        stats.collections_missing += len(edge_def.from_collections) - len(from_collections)

        edges: list[dict[str, Any]] = []

        for coll in from_collections:
            try:
                docs = self._scan_collection(coll, ["chain"])
            except Exception as e:
                stats.errors.append(f"{coll}: {e}")
                continue
            stats.collections_scanned += 1

            for doc in docs:
                chain = doc.get("chain")
                if not chain or not isinstance(chain, list) or len(chain) < 2:
                    continue

                lineage_id = doc["_id"]
                lineage_key = doc["_key"]

                # Sequential edges: chain[i] → chain[i+1]
                for i in range(len(chain) - 1):
                    from_ref = self._resolve_ref(chain[i])
                    to_ref = self._resolve_ref(chain[i + 1])
                    if not from_ref or not to_ref:
                        stats.edges_skipped += 1
                        continue

                    from_coll = from_ref.split("/")[0]
                    to_coll = to_ref.split("/")[0]
                    if from_coll not in existing or to_coll not in existing:
                        stats.edges_skipped += 1
                        continue

                    edge = {
                        "_from": from_ref,
                        "_to": to_ref,
                        "_key": f"{lineage_key}__step_{i}",
                        "source_field": "chain",
                        "lineage_doc": lineage_id,
                        "chain_position": i,
                    }
                    edges.append(edge)

                # Membership edges: lineage_doc → each chain member
                for i, ref in enumerate(chain):
                    to_ref = self._resolve_ref(ref)
                    if not to_ref:
                        stats.edges_skipped += 1
                        continue
                    to_coll = to_ref.split("/")[0]
                    if to_coll not in existing:
                        stats.edges_skipped += 1
                        continue

                    edge = {
                        "_from": lineage_id,
                        "_to": to_ref,
                        "_key": f"{lineage_key}__member_{i}",
                        "source_field": "chain",
                        "chain_position": i,
                    }
                    edges.append(edge)

        if edges and not dry_run:
            self._ensure_edge_collection(edge_def.name, existing)
            try:
                result = self._insert_edges(edge_def.name, edges)
                stats.edges_created = result.get("created", 0)
                stats.edges_skipped += result.get("errors", 0)
            except Exception as e:
                stats.errors.append(f"Insert failed for {edge_def.name}: {e}")
        elif dry_run:
            stats.edges_created = len(edges)

        stats.duration_ms = (time.monotonic() - start) * 1000
        return stats

    def _insert_edges(self, collection: str, edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert edges into an edge collection, overwriting duplicates."""
        path = f"/_db/{self._db}/_api/import" f"?collection={collection}&type=documents&complete=false&overwrite=true"
        import orjson

        payload = b"\n".join(orjson.dumps(e) for e in edges)
        response = self._client._client.post(
            path,
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        result = response.json()
        if response.status_code >= 400 and not result.get("created", 0):
            raise RuntimeError(f"Import failed: {result.get('errorMessage', response.text)}")
        return result

    def materialize_edge(
        self,
        edge_def: EdgeCollectionDef,
        existing: set[str] | None = None,
        dry_run: bool = False,
    ) -> MaterializeStats:
        """Materialize a single edge collection definition.

        Args:
            edge_def: The edge collection definition to materialize.
            existing: Pre-fetched set of existing collections (optimization).
            dry_run: If True, count edges without inserting.

        Returns:
            MaterializeStats with results.
        """
        if existing is None:
            existing = self._existing_collections()

        # Route to specialized handler based on edge type
        if edge_def is CROSS_PAPER:
            return self._build_edges_cross_paper(edge_def, existing, dry_run)
        elif edge_def is LINEAGE_CHAIN:
            return self._build_edges_lineage(edge_def, existing, dry_run)
        else:
            return self._build_edges_standard(edge_def, existing, dry_run)

    def materialize_all(
        self,
        dry_run: bool = False,
        edge_filter: str | None = None,
    ) -> dict[str, Any]:
        """Materialize all (or filtered) edge collections.

        Args:
            dry_run: If True, count edges without inserting.
            edge_filter: If set, only materialize edge collections matching this name.

        Returns:
            Dict with per-collection stats and totals.
        """
        start = time.monotonic()
        existing = self._existing_collections()

        results: dict[str, Any] = {}
        total = MaterializeStats()

        # Group edge defs by collection name to avoid duplicate scans
        seen_names: set[str] = set()

        for edge_def in NL_GRAPH_SCHEMA.edge_collections:
            if edge_filter and edge_def.name != edge_filter:
                continue

            # For shared edge collection names (e.g., nl_hecate_trace_edges),
            # materialize each definition separately — they scan different fields
            label = f"{edge_def.name}:{edge_def.source_field}"

            logger.info("Materializing %s from %s", edge_def.name, edge_def.source_field)
            stats = self.materialize_edge(edge_def, existing, dry_run)

            results[label] = stats.to_dict()
            total.edges_created += stats.edges_created
            total.edges_skipped += stats.edges_skipped
            total.collections_scanned += stats.collections_scanned
            total.collections_missing += stats.collections_missing
            total.errors.extend(stats.errors)
            seen_names.add(edge_def.name)

        total.duration_ms = (time.monotonic() - start) * 1000

        return {
            "edge_collections": results,
            "totals": total.to_dict(),
            "dry_run": dry_run,
            "edge_collections_created": sorted(seen_names),
        }

    def create_named_graphs(self, drop_existing: bool = False) -> dict[str, Any]:
        """Register named graphs in ArangoDB via the Gharial API.

        Args:
            drop_existing: If True, drop existing graphs before recreating.

        Returns:
            Dict with creation results per graph.
        """
        results: dict[str, Any] = {}

        for graph_def in NL_GRAPH_SCHEMA.named_graphs:
            if drop_existing:
                self._client.request(
                    "DELETE",
                    f"/_db/{self._db}/_api/gharial/{graph_def.name}",
                    params={"dropCollections": "false"},
                )

            payload = graph_def.to_gharial_payload()
            resp = self._client.request(
                "POST",
                f"/_db/{self._db}/_api/gharial",
                json=payload,
            )

            if resp.get("error"):
                error_num = resp.get("errorNum", 0)
                if error_num == 1925:  # graph already exists
                    results[graph_def.name] = {"status": "already_exists"}
                else:
                    results[graph_def.name] = {
                        "status": "error",
                        "message": resp.get("errorMessage", "unknown"),
                    }
            else:
                results[graph_def.name] = {"status": "created"}

        return results
