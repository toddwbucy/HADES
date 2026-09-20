# Native database command contracts

Source cut: `e17303571bc84c272f63de8118a016004668e247` (PR #121, merged as `a563d84`).
This source review traces the **all 33 `db` leaves** through
Clap, `main.rs`, CLI adapters and the selected shared handlers/CRUD boundaries.
The [manifest](db-command-contracts.json) records every leaf and inspected source
hash. It is not a claim that all 33 commands have end-to-end runtime coverage.
The top-level `schema apply` command is separate from these database leaves.

## Common return and authority rules

Except export, JSON output wraps results in `success`, `command`, `data` and a
timestamp. JSONL uses one compact envelope, not one line per result document.
Table output presents selected data and is not an equivalent machine envelope.
Ordinary CLI errors propagate to stderr/nonzero; callers must not expect every
error to produce JSON. A successful health command can contain degraded health.

All routes use the explicitly selected effective database; create-database
explicitly overrides the request context to `_system`. Destructive CLI `-y`
checks are confirmation checks, not authorization. Shared dispatch itself does
not implement the daemon's transport access policy. ArangoDB ACLs remain the
write authority; the separate [access-boundary report](database-access-boundary.md)
and unresolved #74 apply. Nothing here certifies production credentials/ACLs.

## Reads and search

`F` below means `--format`, default `json`, accepting json/jsonl/table unless noted.
Limits listed are caller-facing defaults plus shared-handler bounds.

| Leaf | Inputs/defaults and routing | Result/error semantics |
|---|---|---|
| query | Text required by main; limit 10, valid 1–1000; optional profile, hybrid/structural; F. CLI profile flag precedes nonblank HADES_DEFAULT_COLLECTION. Shared DbQuery. | Query/profile/model/dimension/count/results. Rerank rejected; text capped at 64 KiB; admission/provider/database failures propagate. `--verbose` parses but is not forwarded. |
| aql | Required query; optional JSON-object bind and limit; explicit limit clamped to 1000; F. Shared DbAql. | Results/count; mutation keywords rejected after stripping strings/comments. Missing limit is not an implicit 1000-row cap. |
| list | Optional profile, limit 20 (clamped to 1000), paper filter; F. Shared DbList. | Metadata collection, documents/count/full_count. Unknown profile errors; missing collection yields empty result with note. Fields full_text/embedding/text/body are omitted; no CLI fields override. |
| stats | F; shared DbStats across registered static profiles. | Per-profile metadata/chunk/embedding counts and totals. Missing collections count as zero; other errors propagate. |
| recent | Limit 10 (clamped to 1000); F; shared DbRecent. | Default profile metadata only, sorted by created_at then revision descending. Missing collection returns empty with note. |
| health | Optional verbose=false; shared DbHealth. | Reader/writer status and version; degraded connectivity is data, not necessarily nonzero exit. Verbose adds collection counts/index totals and may fail on non-404 collection/index errors. |
| check | Required collection/key ID; shared DbCheck. | document_id/exists; not-found becomes false; other errors propagate. |
| collections | F; shared DbCollections. | Non-system collection names/types/counts and total; a concurrently missing collection counts as zero. |
| databases | F; direct database/user request. | Accessible database names/count. #118 requires a string-array response; malformed replies fail instead of becoming empty. |
| count | Required collection; shared DbCount; fixed JSON. | Collection/count; errors propagate, including missing collection. |
| get | Required collection/key; F; shared DbGet. | Full document. Not-found is an error, unlike check. |
| export | Required collection; optional output/limit; format defaults jsonl and other formats are rejected. Direct paged cursor. | Raw document lines and stderr count. Output file opened/truncated before query; earlier pages may remain on failure. #118 validates pages/cursor state and attempts known-cursor cleanup; no atomic-file or process-death cleanup guarantee. |
| index-status | Optional collection; otherwise registered embedding collections; F. Direct index listing. | Per-collection first vector index/total indexes. Missing collection yields no indexes; other errors propagate. |

## Writes

All use fixed JSON output. Mutations below describe source behavior; none was
executed against a live database for this review.

| Leaf | Inputs/defaults and routing | Result/error semantics |
|---|---|---|
| insert | Collection; --data before --input before stdin. Both explicit input options are accepted; data wins. Shared DbInsert. | Single object or document array. PR #121 preserves successful response shapes, rejects invalid input before write, reports per-item failure counts/diagnostics, and treats malformed outcomes as uncertain. Successful batch items are not rolled back. |
| update | Collection/key; --data or stdin. Shared DbUpdate uses PATCH. | Merge-patch response; missing document errors. Does not replace the whole document. |
| delete | Collection/key; requires --force/-y before pool creation. Shared DbDelete. | Deletes one document; missing document errors. No related-data cascade. |
| purge | Qualified ID; requires --force/-y. Shared DbPurge. | Known metadata profile cascade; codebase files additionally remove owned symbols and referencing standard edges. Returns deletion counts. Arbitrary collections rejected. This trace is not concurrency/transaction proof. |
| create | Name; type document by default, or edge. Shared DbCreateCollection. | Backend creation response; unknown type rejected. |
| create-database | Name; direct configured client with database set to _system. | created/name/backend response on HTTP success; not an administrative grant to the caller. No CLI database-drop command exists. |
| truncate | Collection; requires --force/-y. Direct CRUD. | Clears documents while retaining collection/indexes; reports truncated/collection/schema_referenced/response. Warning uses static registered collection names, not runtime schema inspection. |
| drop-collection | Collection; requires --force/-y. Direct CRUD. | Removes collection; same static warning policy; reports dropped/collection/schema_referenced/response. |
| create-index | Optional collection defaults to default-profile embeddings; dimension required by adapter; metric cosine by default, aliases l2/euclidean and dotproduct/innerProduct. Shared DbCreateIndex. | Vector index on embedding; auto nLists=max(1,count/15), defaultNProbe=10. Index metadata returned. Collection-name interpolation in the preliminary count query needs separate adversarial review; this table does not certify that input boundary. |

## Graph commands

All seven routes use shared dispatch and fixed JSON output. Main discards the
format strings declared for traverse, shortest-path, neighbors and list; even an
unsupported format string reaches the adapter without format validation.
Traversal directions accept exactly outbound/inbound/any. Vertex handles are
validated. An omitted graph resolves only when exactly one named graph exists;
zero or multiple graphs cause an error. Explicit graph names go to the query.

| Leaf | Inputs/defaults and routing | Result/error semantics |
|---|---|---|
| graph traverse | Start; direction outbound; depth 1–3; optional graph. Shared DbGraphTraverse, CLI supplies no limit. | Default 100 rows; shared limit capped at 1000. Maximum depth clamped to 20; minimum above 20 or above effective maximum rejected. Returns vertex/edge results, start, resolved graph and direction. |
| graph shortest-path | Source/target; optional graph. Shared DbGraphShortestPath; CLI hardcodes direction any. | Returns vertex/edge results, from/to, resolved graph and direction. No CLI result limit. |
| graph neighbors | Vertex; direction any; limit 20, capped at 1000; optional graph. Shared DbGraphNeighbors. | Depth-one traversal results, vertex and direction; resolved graph is omitted from this result. |
| graph list | Shared DbGraphList, Gharial reader request. | Graph names and edge definitions. Missing/non-array graphs becomes empty; missing names become unknown. Malformed-response handling needs a focused contract. |
| graph create | Name; optional JSON edge-definitions. Shared DbGraphCreate. | Name must be nonempty ASCII alphanumeric/underscore/hyphen. Explicit definitions must be an array; item shape is delegated to backend. Otherwise loads named definition from runtime schema. HTTP success reports created=true; payload outcome validation remains separate. |
| graph drop | Name; drop-collections=false; force required by both CLI and dispatch. Shared DbGraphDrop. | Same name validation; Gharial deletion with dropCollections flag. Returns dropped and collections_dropped. No transaction or recovery guarantee established here. |
| graph materialize | Optional edge filter; dry-run=false, register=false. Shared DbGraphMaterialize. | Loads runtime definitions; standard/lineage/cross_paper strategies; upserts chunks of 5000. Reports per-definition/totals counts and errors. Many scan/import/registration failures accumulate inside a successful result; earlier writes remain. Dry-run counts candidates and suppresses writes/registration. Unknown edge filter errors; requested registration tolerates graph-already-exists. |

## Database schema commands

These four routes use shared dispatch and fixed JSON. RuntimeSchema::load requires
an existing, nonempty hades_schema collection and metadata, checks relation
count/checksum and graph references, and has no static fallback. Legacy fallback
strings remain in callers but do not describe a successful load at this source cut.

| Leaf | Inputs/defaults and routing | Result/error semantics |
|---|---|---|
| schema init | Required --seed; only empty accepted. Shared DbSchemaInit. | Creates hades_schema if absent, then truncates it and inserts one metadata document (zero relations, feature_dim 2048). Existing definitions are reset without a force flag; truncate and insert are separate operations. Import errors fail but do not restore prior schema. Requires a focused reset/failure/recovery contract. |
| schema list | Shared DbSchemaList. | Edge-definition summaries, named-graph summaries and source=database. Loader failures propagate. |
| schema show | Required name; shared DbSchemaShow. | All matching edge definitions take precedence over a named graph with the same name. Unknown name errors. |
| schema version | Shared DbSchemaVersion. | Version, relation-order checksum, seed name, relation count and feature dimension. This is metadata inspection, not deployed binary/model provenance proof. |

## Evidence and remaining scope

[Response-shape contracts](cli-db-response-shapes.md) and
[insert outcome contracts](cli-insert-outcomes.md) supply private actual-CLI and
shared-service evidence for the identified false-success cases. Existing CRUD,
query, index, cursor and lifecycle fixtures supply their separately stated scope;
source traces do not turn them into complete CLI parameter-combination tests.

Reviewed mismatches: the CLI list adapter's “whole documents” comment disagrees
with shared field omission; recent's “across profiles” comment disagrees with its
single default profile; query verbose is accepted without a handler argument.
These are documented source limitations, not production incidents or resolved
behavioral changes. Format semantics, zero limits for list/recent/AQL, concurrent
purge behavior, index-name handling and other write-response validation remain
candidates for focused operational contracts. The other 47 native leaves and
full daemon/MCP/viewer parity remain separate audit scope.
