# Status, orientation and analyzer command contracts

Source cut: `a9a152be592a3dc95a834ee27ceb9fd3cd986622`. This traces four native
leaves from Clap and main through adapters and selected shared handlers. The
[manifest](system-command-contracts.json) binds the inspected sources. No live
status query, analyzer invocation, installation or production mutation was run.

## Status and orientation

Both accept --format json (default), jsonl or table, validated before pool
construction. Their adapters use shared dispatch. JSON/JSONL return the common
success/command/data/timestamp envelope; table is a lossy presentation. Main
loads configuration and applies database/GPU overrides before routing. There is
no separate config command in the visible command inventory.

| Leaf | Inputs and route | Result/error contract |
|---|---|---|
| status | --verbose/-V defaults false; shared Status handler. | Effective database/socket summary, Arango reader/writer health and version, embedding metadata. A degraded database or unavailable embedder is successful result data, not necessarily nonzero exit. |
| orient | Optional --collection/-c; shared Orient handler. | Without a collection, enumerates static profiles, three counts per profile, five recent metadata rows and totals. With a collection, treats it as a raw collection name, returning count, one-document field sample, ten recent rows and indexes. |

Status probes the configured embedding endpoint inside a five-second timeout;
model_loaded is reported separately from running. The adapter does not impose
the search-specific embedding response cap. Verbose status adds static-profile
counts: missing collections become zero, other count errors propagate. Database
health checks shared connections once or separate reader/writer connections
concurrently. These are connectivity observations, not complete readiness or
binary/model provenance verification. Configuration output contains selected
endpoint/database fields, not a full configuration dump.

Orient count queries likewise tolerate not-found and propagate other errors.
However, its sample query, recent-document queries and index listing suppress
all errors into absent/empty data. Recent ordering uses processing_timestamp,
then revision, descending; returned fields are key/title/processing_timestamp.
This can conceal failures other than absence. The recent_docs comment mentions
only missing collections, which is narrower than the implementation. Focused
failure fixtures remain required before assigning a runtime finding.

## Analyzer management

| Leaf | Inputs and route | Result/error contract |
|---|---|---|
| tools status | Optional --workspace; explicit path canonicalized, otherwise cwd. CLI-only adapter, fixed JSON. | Resolves rust-analyzer and gopls via configured pin, managed directory, then PATH; probes from the workspace. Failed configured pins produce success:false and nonzero exit. Unconfigured failures remain inventory data with successful outer status. libclang is explicitly unprobed. |
| tools install | rust-analyzer or gopls required; optional --version, default latest; --allow-unverified defaults false. CLI-only adapter, fixed JSON. | Installs into HADES_TOOLS_DIR or HOME/.local/share/hades/tools, records manifest, then probes. Post-install failure prints success:false and exits nonzero; files may already have changed. Unknown tool rejected before filesystem changes. |

Shared analyzer probes use a ten-second deadline, 64 KiB per-stream cap and owned
process-group cleanup. gopls uses `version`; rust-analyzer uses `--version`.
Managed presence checks for a file, while executability is determined by probing.
An explicit bad pin does not fall back to another executable.

Rust-analyzer installation chooses a Linux/macOS x86_64/aarch64 release asset,
checks its SHA-256 against the release API digest, then decompresses and replaces
the executable via temporary-file rename. Missing digest requires explicit
allow-unverified; mismatched digest still fails. The digest and URL share the
same trust source. gopls runs `go install` with GOBIN set to the managed directory
and probes the resulting binary. The allow-unverified flag affects only the
rust-analyzer path. Post-install resolution passes no configured analyzer pin.

## Boundaries still requiring evidence

The compressed-download 256 MiB check occurs after buffering if Content-Length
is absent; release JSON and decompressed output have no explicit size cap in
this adapter. No explicit request/build timeout bounds the entire installation.
Binary and manifest replacements are separate operations, with fixed temporary
names and no observed installation lock. Manifest parse failure preserves the
manifest but can occur after binary replacement. These are source limitations,
not measured resource failures or concurrent-install incidents.

Existing helper tests cover digest parsing, gzip round-trip, executable rename,
manifest preservation and unknown-tool rejection. They do not establish full
network installation, crash recovery, concurrent installation or status failure
semantics. Combined with the 33-leaf database map, source contract tables cover
37 of 80 visible leaves; the other 43 and full transport/viewer parity remain
open. This count is not end-to-end test coverage.
