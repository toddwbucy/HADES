# Registered provider discovery

This internal [#225](https://github.com/toddwbucy/HADES/issues/225) slice connects
protected local registrations to actual provider observations and database-bound
clients. `ProfiledDocuments::connect(pool, &catalog)` now resolves the saved
provider ID itself. Its caller cannot inject an endpoint or observed receipt.
[Public search admission](profiled-search-admission.md) now supplies this catalog
from explicit installation configuration. [Foreground file ingestion](profiled-file-ingestion.md)
uses the same catalog. [Background document jobs](profiled-ingestion-jobs.md)
also carry verified profile pins; frontend setup remains pending.

## Installation-owned catalog

`persephone::provider_registry::ProviderCatalog` accepts a registry root and an
explicit list of `ProviderConnection` values from trusted installation code.
These are operational settings, not browser input. No provider, model, endpoint,
or environment fallback is selected. There may be zero to 128 configured entries;
IDs follow `[a-z][a-z0-9_-]{0,63}` and duplicates are refused.

The initial implementation supports same-account Unix providers only. Socket
paths must be absolute, without traversal or symlink ancestors. The socket and
its immediate 0700 parent belong to the service account. Ancestors must belong
to that account or root and cannot be shared writable directories, except for
root-owned sticky directories such as `/tmp`. Remote HTTP, cross-account socket
authorization, and credential resolution require later explicit contracts.

Timeouts are positive and at most 600 seconds. Connection response limits are
1 byte–8 MiB; discovery additionally caps descriptions and error bodies at
64 KiB. A timeout or failed observation never selects another provider.

## Inspection and explicit verification

`provider_ids()` lists only the allowlist. `inspect(id)` reads registered model,
revision, adapter label, artifact MD5, and file count without contacting a model.
Its `loaded_verified` value is always false. Listing/inspection do not scan
unconfigured registrations, download assets, or load GPU weights.

`verify(id)` performs these steps:

1. Read `records/<id>.json` using the existing `hades-model-registration-v1`
   envelope. Reject duplicate/unknown fields, inconsistent manifest MD5, unsafe
   snapshot names, and records over 2 MiB. The manifest retains its 1 MiB limit.
2. Check account ownership and permissions on the registry, metadata, and artifact
   directory. Open descendants relative to checked descriptors without following
   links; refuse nonregular metadata and hard links. Use the registrar's shared,
   nonblocking `.lock`; a busy registry requires an explicit retry. No files or
   permissions are changed.
3. Release the registry lock and send an empty `POST /v1/verify` to the configured
   private service. This is an explicit model-load/observation operation.
4. Parse the full typed description directly from bounded response bytes. Compare
   model ID/revision and artifact MD5 with registration; validate tokenizer assets,
   tasks, dimensions, capabilities, space identity, and receipt consistency for
   every advertised workload, including workloads the caller has not enabled.
5. Re-read the protected registration and compare file/directory identities and
   record timestamps/content. A replacement or change-and-restore during loading
   discards the observation. No lock is held across model loading.

The result is an opaque `VerifiedProvider`; it has no public constructor or
deserializer. Its manifest comes from registration, not HTTP. Read-only accessors
expose the checked portable description without paths. The common registration
reader treats adapter labels as metadata; adapter-specific asset validation and
actual weight hashing remain the registrar and verified loader's responsibility.
The production registrar currently supports Jina only. This is a trusted local
runtime boundary, not attestation against code controlling the service account.

## Workload observations

Discovery accepts two strict response shapes from the same empty `POST /v1/verify`:

| Description | Required root fields | Meaning |
| --- | --- | --- |
| Original, unversioned | `model`, `capabilities`, `space`, `receipt` | One text space; existing verified conventional and contextual Jina providers retain this wire format. |
| Workload-aware | `format: "hades-provider-description-v2"`, `model`, `workloads` | Explicit document and/or code observations from one loaded, registered model. |

V2 `workloads` permits only `documents` and `code`. Each present observation
contains exactly `capabilities`, `space`, and `receipt`, using the existing
capability, embedding-space, and receipt contracts. An absent or null workload is
unavailable; at least one must be present. Root fields from the two formats cannot
be mixed. Unknown fields, duplicate keys at every level, null required fields,
and unsupported versions are refused. The complete response still fits 64 KiB.

All observations share the root model identity, registered artifact manifest,
declared tokenizer dependencies, and one serving generation. Each receipt must
match its own space ID and the registered artifact MD5. Document and code spaces
may differ in task pair, prompts, adapters, pooling, and vector width. Code-only
observations are valid; they cannot construct a document profile.

Capabilities are checked within the selected workload. A document adapter's late
chunking or token-offset support does not authorize late code ingestion; code
specialization must be supported by the code observation. Different live token
ceilings remain operational limits and never rewrite the saved profile.

Serving capability does not enable database intent. Under V2, code as ordinary
text requires both an explicit code observation and a saved code workload with
`specialized_code: false`; that observation may reuse the document space and
receipt. A missing V2 code observation is always unavailable.

The original unversioned format retains compatibility with explicitly enabled
ordinary-text code profiles that match its exact space and supported chunking
capabilities. This preserves document-only traversals of profiles with code
enabled, including trees whose code files are ignored. It cannot enable code
intent, change task/prompt/representation, or authorize specialized code; a
specialized code observation requires V2. Managed code ingestion remains gated.

Rust callers inspect `description().model()` and
`description().workload(WorkloadKind::Documents)` or `WorkloadKind::Code`.
Serialization preserves the observed wire format. This is a consumer contract
with synthetic mixed/code-provider coverage; a real verified Jina code producer,
its qualification, and code graph persistence are subsequent #225 work.

## Database admission and lifetime

`verify_profile(profile, manifest)` checks the saved model against registration
before contacting the provider, then compares actual behavior and every enabled
workload against its own observation. Same dimensions are insufficient. Changed
tokenizer dependencies, spaces, tasks, or required late/code capabilities refuse
admission. Binding a document client also checks any enabled code workload.

A smaller live token ceiling is an operational change, not an embedding-space
change. It does not rewrite the saved profile: conventional planning still uses
the smaller of saved and live ceilings, and inference refuses overflow.

`VerifiedProvider::document_profile(generation, chunking)` creates initializing
revision one with explicit document intent and observed model metadata. Code
remains disabled. This constructs data only; it neither creates a database nor
marks readiness. It uses only the document observation's capabilities.
`client(...)` additionally requires a ready profile and captures the selected
workload's receipt for all planning/inference requests. A response carrying a
different workload space's receipt is refused even when model, dimension, and
load match. Ordinary-text code using the exact document space shares its receipt.

Document connection checks saved readiness, supported policy, and required
collections before provider contact, then checks the complete database snapshot
again after observation. A profile change during loading prevents admission.
Existing prepared-write and query guards remain in force.

There is no global observation cache or transparent receipt refresh. Clones
retain their original load pin. A compatible provider reload requires explicit
verification and new clients; older clients reject changed receipts. An idle
unload may invalidate an observation before its first request, which fails
explicitly. Registry inspection does not acquire or own a model lease.

## Validation and next integration

```bash
cargo test --locked -p hades-core --test provider_discovery_contract \
  --test bound_embedding_contract --test embedding_contract --test embedding_client
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_documents
```

The documented native `--arangod /path/to/arangod` alternative starts a private
Unix-only server. Tests use synthetic registrations, private sockets, and disposable
databases. They cover read-only inspection, unsafe paths/permissions, concurrent
registration, strict wire parsing, resource limits, compatibility, reloads,
database isolation, and policy drift during verification. Workload cases cover
mixed and code-only observations, code as ordinary text, capability isolation,
cross-workload receipt rejection, coherent load generations, and unchanged
document provenance through the original and V2 formats.

[Real-Jina qualification](verified-jina-qualification.md) now makes Rust read the
actual protected registration and observe `/v1/verify` independently before profile
initialization and document admission. Its observation must agree with the Python
qualification, including the receipt and manifest. The
[recoverable setup coordinator](recoverable-workspace-setup.md) now verifies
providers before database creation and again before readiness publication.
Public search and [foreground file ingestion](profiled-file-ingestion.md) now
enforce the saved profile, as do background document jobs. Frontend creation remains subsequent work.
