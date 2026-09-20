# Code-file identities and migration

Version 2 identifies each file by its **canonical absolute ingest root and exact
relative UTF-8 path**. Full SHA-256 hashes length-delimited components; a bounded
ASCII prefix is only a readability aid. Keys stay below ArangoDB's 254-byte limit,
including existing symbol, chunk, and embedding suffixes. Dots, underscores,
Unicode, long paths, and identical relative paths in different roots stay distinct.
Non-UTF-8 paths are rejected before ingest writes file data.

File documents record `file_key_version: 2`, `ingest_root`, and `path` together.
Ingest checks all discovered identities before processing files and checks each
identity again before replacement. Conflicting ownership or any legacy file
document rejects ingest, including `--force`. These checks do not provide a
transaction across concurrent ingestion jobs: serialize writers for a corpus.

Canonical aliases of one root retain identities. Moving a root, ingesting a
subdirectory instead of the original root, or ingesting a single file with a
different parent namespace creates different identities. Keep the ingest root
stable. Drift uses the same namespace and rejects legacy identities within its
comparison scope instead of emitting misleading retirement candidates. Attributed
version-2 rows must reproduce their stored key from root and path; missing or
mismatched metadata aborts drift. Unknown ownership stays out of retirement lists.

## Existing databases

This is a storage compatibility change. Existing data is not automatically
rekeyed. Read-only legacy lookup remains available, and explicitly reviewed
retirement by stored key remains available. Replacing a file key alone is unsafe:
symbol/chunk/embedding keys, edge endpoints, adapter citations, and external
references may all depend on it. Earlier collisions may already have destroyed
data; an old key cannot recover which source owned it.

Use an explicitly scheduled rebuild and cutover:

1. Inventory each original ingest root, source revision, language/analyzer options,
   schema, adapter configuration, model identity, and external key consumers.
   Verify a backup and restore in a separate environment before any cutover.
2. Create a new isolated database. Re-ingest the recorded sources with consistent
   roots using the new binary; rebuild domain edges and embeddings with verified
   model settings. Do not copy ambiguous old identities or relabel old vectors.
3. Compare file inventories, symbols/chunks, endpoint integrity, drift results,
   and representative retrieval results. Rebuild external references from source
   identities; review every unresolved old-key mapping. Retrain incompatible
   graph checkpoints following [training guidance](training-evaluation.md).
4. During a maintenance window, pause writers, capture/replay source changes,
   repeat validation, and switch the configured database and compatible binary
   together. Preserve the previous binary, configuration, and database for rollback.
5. Roll back both binary and database selection if validation fails. Reconcile any
   writes after cutover before rollback; do not delete the old database until the
   retention and acceptance review is complete.

No production rebuild, profile switch, or service restart is part of the audit
fix. The disposable regression target is `cargo test -p hades-cli --test
file_identity`; configure only a separate ArangoDB instance and set
`ARANGO_TESTS=1` so missing prerequisites fail.
