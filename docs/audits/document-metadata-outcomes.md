# Document ingestion metadata outcome audit

## Real-backend reproduction

At source 3dd1990, a private ArangoDB 3.12.11 collection validation rule accepts
the pipeline document without a source field but rejects the subsequent PATCH
setting source=local. Actual CLI ingestion exits 0 and reports outer success,
item success, completed=1 and failed=0 despite HTTP400/error1620. The rejection
appears only in warning logs. Readback confirms full text persists while
source_path and content_hash are absent; one chunk and embedding for this file
also persist (two each including the successful control).

A valid control first verifies source=local and a recorded content hash. Private
gRPC extraction and HTTP embedding peers provide deterministic content/vectors;
`--task code` is explicitly selected to match the shared synthetic embedding
fixture. This is not a production-model relevance measurement. The database,
files, sockets and peers are disposable; no production service is involved.
Normalized actual output, readback observations and source/probe hashes are
preserved in the adjacent JSON.

## Impact and required disposition

The item becomes successful before required source identity/content-hash metadata
is durable. This can undermine subsequent skip/collision behavior and causes
callers and checkpointing to trust an incomplete result. The committed pipeline
write must not be described as rolled back when its later PATCH fails.

Require failure to propagate into the per-item result, batch summary, outer
success flag and process exit, preserving other items' completed results. Retain
stage/document diagnostics and disclose already committed content. Test success,
metadata rejection, retry after rejection removal, mixed batches and checkpoint
resume so a failed metadata stage is not silently skipped as completed. Evaluate
whether metadata belongs in the pipeline transaction rather than claiming that
error propagation alone makes ingestion atomic. Metadata identity override and
batch/checkpoint scoping remain separate source-review candidates.
