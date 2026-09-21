# Schema application acknowledgment audit

## Reproduction and scope

Issue #135, part of epic #12. At source d250666, actual CLI schema application
of an empty YAML mapping reports applied:true, success:true, and exit 0 for three
private injected HTTP-success import replies: errors=1, missing counters, and
zero created/updated/errors despite one submitted metadata document. A valid
created=1 control succeeds first. All three cases execute before the regression
assertion fails (0.03s). The adjacent JSON preserves normalized actual responses
and four source/probe hashes.

The fixture acknowledges collection creation, then supplies the import reply.
No live database was used and no persistence is asserted. In particular this
does not prove that ArangoDB emits errors=1 with HTTP success under complete=true.
It proves acceptance of unconfirmed write outcomes at the client boundary.

## Source findings

The CLI reads the entire YAML. Dry-run validates and prints a plan without opening
a database pool; it does not inspect permissions, existing types, or guard state.
Core dry-run returns zero counters instead of the CLI plan.

Validation checks declared collection and seed key uniqueness, references and
endpoint types, relation-order duplicates, and known model types. It does not
establish valid database identifiers, nonempty endpoint sets, positive dimensions,
unique graph names, or existence of undeclared training relations. Repeated edge
names are rejected despite comments describing identity by name plus source field.

The in-use guard checks declared non-universal collections only. Universal codebase
collections are exempt; implicit hades_schema is not inspected unless declared.
Missing or nonnumeric count results become zero. Force skips this guard, and no
transaction connects guard reads with writes.

Execution ensures collections, replaces seed documents by key in sorted collection
order, writes edge/graph metadata, writes schema_meta, then creates named graphs.
Conflicts are treated as existing without type/definition readback. Import errors
and acknowledged counts are not validated; metadata import results are discarded.
Earlier effects remain on later failure without a structured partial report.
Empty YAML still replaces metadata. Omitted definitions are not removed, so apply
is not full reconciliation. The checksum covers relation order, not the whole schema.

## Required work

Validate document accounting at seed, edge-definition, graph-document and metadata
imports. Reject malformed, failed or unaccounted replies with stage diagnostics;
retain valid created/updated controls and explicitly disclose earlier effects.
Verify relevant backend semantics in a private database separately from injected
reply contracts. Guard exemptions, reconciliation, identifiers and existing-object
mismatches remain separate candidates. No production changes were made.

## Remediation and verified backend behavior

All four schema import stages now retain the raw response until acknowledgment
validation completes: error must be false; created, updated, errors, empty, and
ignored must be unsigned counters; errors/empty/ignored must be zero; checked
created+updated must equal the submitted count. Stage and collection diagnostics
warn that earlier operations may have committed and require inspection before retry.
The underlying complete=true/onDuplicate=replace import semantics are unchanged.

Ten CLI response tests pass, including all four stage failures, successful created
and updated controls, offline dry-run, in-use refusal, and 31 malformed/inconsistent
counter cases. Existing response contracts remain covered.

The real private ArangoDB 3.12.11 lifecycle suite passed all 17 tests. The new schema
case creates and reads back seed data, all three schema records and the named graph;
refuses an in-use apply; successfully replaces a seed under force while retaining
an unrelated document and skipping the existing graph. An invalid-key batch is
rejected with HTTP 400/error 1221, produces CLI exit 1 and empty stdout, and leaves
the prior seed value, unrelated document, collection count, and metadata revision
unchanged. The private server stopped with exit 0. This verifies atomic rejection
of that single complete=true import, not rollback of the whole schema operation.

The historical injected HTTP-success/error reply remains a client-boundary test;
it is not represented as normal backend behavior. Baseline JSON is preserved.
Guard exemptions, schema reconciliation, collection/graph mismatch checks and
broader recovery readiness remain outside this fix. Nothing was deployed.
