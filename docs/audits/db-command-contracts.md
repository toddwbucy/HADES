# Native database command contracts

Source cut: `e17303571bc84c272f63de8118a016004668e247` (PR #121 pending).
This source review traces the **22 non-graph/non-schema `db` leaves** through
Clap, `main.rs`, CLI adapters and the selected shared handlers/CRUD boundaries.
The [manifest](db-command-contracts.json) records every leaf and inspected source
hash. It is not a claim that all 22 commands have end-to-end runtime coverage.
The seven `db graph` and four `db schema` leaves remain outside this table.

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
candidates for focused operational contracts. The other 58 native leaves and
full daemon/MCP/viewer parity remain separate audit scope.
