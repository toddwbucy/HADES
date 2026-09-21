# Graph-update destination selection

Tracking: epic #12, P2 #144. Baseline `91bfa01` preserves seven parser-level
false acceptances on source `7bd9f64`: null/empty rows, absent or mistyped flags,
contradictory absence and invalid identity states all produced no missing nodes.
The caller treats an empty selection as a successful missing-only no-op. This
was a private reproduction, not an observed production backend incident.

## Fix

The selection query echoes each requested key. The parser requires typed keys
and booleans plus an explicit ID value, exactly one row per requested key,
no duplicates or foreign keys, and qualified IDs matching the requested
collection/key. Absent documents require null IDs and `missing=false`.
Present documents require a matching string ID; only `missing=true` selects
one for update. Row ordering is unrestricted. Legitimate absent and already
embedded nodes retain existing no-op behavior. Query errors still propagate.

## Verification

- Five unit tests cover the original malformed shapes, valid mixed states,
  missing/duplicate/foreign responses, identity mismatches and valid no-op.
- The full 13-test actual CLI response suite passes. The new private Unix-peer
  fixture checks eight malformed response cases exit unsuccessfully without
  stdout, two valid no-ops report no service contact, and a real selected item
  reaches missing-checkpoint validation. A nonexistent checkpoint prevents
  accidental contact with the fixed training socket.
- Two existing training-alignment tests pass against disposable ArangoDB
  3.12.11 and a private CPU provider: actual missing-only export updates one row,
  preserves unselected rows/revisions, and the legitimate no-op preserves state.
  The separate full train/update lifecycle also passes. The owned server exits
  cleanly; live sockets and GPU devices are hidden by the fixture.
- CLI all-target Clippy with warnings denied, rustfmt and diff checks pass.

Run `cargo test --offline -p hades-cli --bin hades commands::graph_embed_update::tests`
and `cargo test --offline -p hades-cli --test db_response_shapes` for local
contracts. The real backend check uses `scripts/test_isolated_database.py
--arangod <private-binary> --training-alignment-python <CPU-venv>` after protobuf
generation. Source hashes and normalized outcomes accompany this report.

The added query field is internal to the CLI's database request, not a daemon
protocol change. This does not establish vector freshness, checkpoint-generation
consistency, snapshot isolation during concurrent updates, GPU correctness or
representative retrieval quality. No production component was changed or deployed.
