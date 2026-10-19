# Local model registration and checked Jina loading

This slice of [#225](https://github.com/toddwbucy/HADES/issues/225) builds on the
[identity contracts](embedding-identity-contract.md). It provides an administrative
registration command and an internal Jina loading boundary. HTTP configuration,
database setup, and existing workspaces do not activate this path yet.

## Register an approved local model

The local operator must approve the source and supply its resolved Hugging Face
commit ID. Registration does not download models or authenticate their publisher.
MD5 records the approved bytes and detects subsequent drift.

From `services/`, using an appropriate Python environment, set `JINA_REVISION`
to the approved snapshot's resolved 40-character commit ID:

```sh
python -m embedding.registry --registry "$HOME/.local/share/hades/model-registry" \
  register --provider local-jina \
  --source /path/to/approved/jina-snapshot \
  --revision "$JINA_REVISION"

python -m embedding.registry --registry "$HOME/.local/share/hades/model-registry" \
  inspect --provider local-jina
```

The command reports the model ID, revision, aggregate manifest MD5, and asset
count. `loaded_verified: false` means registration alone establishes no claim
about a running provider. It never changes a database or starts inference.

The first adapter accepts local Jina v4 bundles with a fast tokenizer, processor
configuration, safetensors weights or a complete shard index, adapter configuration
and weights, and locally resolvable `auto_map` code. It includes supported runtime
files recursively; cache-file symlinks are copied as regular files under their
logical paths. Directory symlinks and special files are rejected. Known cache
directories (`.git`, `.cache`, `__pycache__`), Markdown documentation, license/notice
files, `.gitattributes`, `.gitignore`, and Jina's evaluation `results.json` are
excluded. Unknown asset suffixes cause refusal rather than silent omission.
This is a Jina-specific selection policy, not a general model importer.

Required assets and configuration references are checked, including every listed
weight shard. Model-local Python remains trusted code; these checks are not a
static proof of arbitrary code's dependencies. Remote `auto_map` references are
refused. The loading process must start with both `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; runtime library settings are checked before registered
loading is accepted. Approved code and installed libraries remain part of the
trusted execution environment.

## Registry ownership and persistence

The registry and its `records/`, `artifacts/`, and `sessions/` directories require
mode `0700` and ownership by the operating account. Records are private `0600`
files. A process lock serializes registration; publishing a record is exclusive
and follows file/directory synchronization. A provider ID is append-only: changed
artifacts require a new ID. No automatic overwrite, adoption, or migration occurs.

Registration copies files into a private artifact tree, hashes copied bytes,
checks the source remained consistent, and removes write permissions. Copy or
validation failure removes the incomplete tree without publishing a record. A
crash before publication can leave an orphan artifact tree; do not delete such
trees while registration/loading is active. If publication durability is uncertain,
inspect the ID rather than retrying with different bytes. Cleanup/reconciliation
automation is not part of this slice.

Keep `records/` and referenced `artifacts/` together in backups. `sessions/`
contains disposable copies owned by active loaders and is not portable state.
After a process crash, stale session copies may be removed while that provider
is stopped. Do not reuse them as verified loads.

## Serving lifecycle

Internal callers can obtain a `RegisteredModel` from `ModelRegistry.load()` and
pass it as `registration=` to `JinaV4Embedder`. A load:

1. Copies the registered assets into a fresh private directory and recomputes
   their manifest MD5 against the saved registration.
2. Makes the copy read-only and records its complete file/directory inventory,
   inode identities, permissions, sizes, and modification/change timestamps.
3. Loads tokenizer and model from that same unique directory, using local-only
   acquisition and safetensors for the base weights.
4. Rechecks the inventory after loading. Only then does it publish model objects
   and a fresh serving-generation ID together.

Each registered inference operation checks the snapshot before and after its
complete batch/late-chunk operation. Metadata checks avoid rehashing large weights
per request. Mutation, replacement, insertion, or removal invalidates the loaded
model and provenance and prevents returning its vectors. The artifact lifecycle
lock serializes loading, inference, and unload, so cleanup cannot remove files
between batches. Unload clears provenance and deletes its private copy; the next
load gets a fresh directory and serving generation even for identical bytes.

The copies isolate normal model-cache changes. Read-only permissions plus checked
metadata detect accidental mutation of the private copy; this is not kernel-level
immutability or a sandbox against a hostile service-account process, root, or
malicious approved model code. Those actors control the runtime itself. The
contract targets trusted local provisioning and accidental artifact drift.

Registration and each resident load require a complete additional disk copy;
hard links are deliberately avoided. The default limit is 32 GiB per copy, with
`register --max-bytes` available for the administrative copy. No large model is
copied during tests. Capacity planning and real GPU qualification are still needed
before enabling this path in an installation.

## Remaining boundary and validation

`loaded_artifact` is checked internal load provenance, **not** a full provider
receipt: it does not claim a verified task, prompt, pooling, precision, or embedding
space. The next integration must derive those contracts, enforce expected receipts
on requests/responses, and connect the Rust client. Database persistence, guarded
ingest/query operations, provisioning recovery, and frontend setup remain pending.

CPU tests use synthetic artifact files and doubles only for model/tokenizer
acquisition; they execute the real load guards, inference entry points, pooling,
and unload lifecycle. Run in the [isolated CPU test environment](workstation-specific-tests.md):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest services/tests/test_model_registration.py \
  services/tests/test_registered_jina_loading.py \
  services/tests/test_embedding_identity.py -v
```

Coverage includes relocation, cache links, missing shards and adapters, changed
dependencies, concurrent registration, byte limits, partial-load cleanup,
mutation during inference, reload identity, and unload waiting for active work.
The installed Jina directory was inspected read-only for asset/configuration
compatibility; it was not hashed, registered, loaded, or used for inference.
