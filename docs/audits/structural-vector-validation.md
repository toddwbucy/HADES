# Stored structural-vector validation

Issue #85, under epic #12. This is repository remediation, not a deployment.

## Reproduction

At `22e0d7c`, private cursor fixtures exercising the real service envelope
showed that `graph_embed.embed` returned a stored `[null]` vector as success
and `graph_embed.neighbors` accepted that target. The two rejection tests
failed; a valid numeric control passed. Those baseline loops stopped at their
first invalid example, so they do not establish execution of every variant.

## Contract

Requested vectors must be nonempty arrays of numbers within the finite float32
range. Missing, non-array, empty, null/string/boolean components and out-of-range
numbers produce the existing `QUERY_FAILED` envelope. No new dimension cap is
introduced. Candidate vectors must satisfy the same numeric range and match the
target dimension; malformed candidates are excluded before dot-product scoring.
Target exclusion, global sorting, rounding and request limits retain their
existing behavior. Numeric validity does not certify normalization, shared
checkpoint provenance, graph freshness or retrieval quality.

## Verification

Three private cursor/service tests cover invalid shapes and components plus a
valid control. `structural_vectors_db` runs only through the isolated database
harness, which requires explicit private configuration and never discovers a
live endpoint. Its synthetic documents cover null, string, boolean, positive
and negative out-of-range values, wrong dimensions, non-array, empty and missing
vectors. The real query returns exactly two valid neighbors across two document
collections, in score order (0.75, 0.5), excluding the target. Both APIs reject
malformed requested vectors and accept a valid target.

The first local database run passed candidate filtering but failed because the
test sent the neighbor-only `limit` parameter to the strict lookup endpoint.
After correcting that fixture, the complete test passed on a fresh ArangoDB
3.12.11 instance. Both runs stopped their private process groups. The replay
used a Unix socket without TCP, one CPU, lowered priority, 8 GiB child limits
and a 300-second command deadline. No production database or service changed.
Selected source hashes are retained in `structural-vectors-db-result.json`;
they bind these sources, not every runtime dependency.

```bash
cargo test -p hades-core --test structural_vectors
python3 scripts/test_isolated_database.py --arangod /path/to/arangod \
  --contract structural_vectors_db
```

CI includes the service-free test and the real database target. Full final-head
CI and review remain required before merging this change.
