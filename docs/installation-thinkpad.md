# ThinkPad installation record

This records the first HADES installation outside Olympus, starting on
2026-10-18. The immediate objective is a fresh local ArangoDB build, fresh
research database, and working HADES CLI, browser, extraction, and embedding.
Data migration, context-window improvements in #208, and scanner development
are separate later work. Use this record to develop a reproducible installation
guide. The baseline installation, GPU embedding, and isolated ingestion/retrieval
workflow were verified on the same day after the operator patched and rebooted.

## System baseline

This table records the initial inspection. The post-reboot kernel and driver
versions are recorded under **NVIDIA driver versions must agree** below.

| Component | Observed value |
| --- | --- |
| HADES revision | `7423624e029c6f117e122456c06a721bf407cb82` |
| OS | CachyOS, Arch-based rolling distribution, x86_64 |
| CPU | Intel Core Ultra 7 265HX, 20 logical CPUs |
| RAM | 62 GiB reported by `free -h` |
| Project storage | 2.5 TiB available at initial inspection |
| Temporary storage | `/tmp` is a 32 GiB tmpfs; use project storage for builds |
| Home filesystem | Separate 473 GiB root volume; only 13 GiB free after initial dependency/model downloads |
| GPU | NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, PCI `10de:2c38` |
| Running kernel | `7.2.8-1-cachyos` |
| Installed primary kernel | `linux-cachyos 7.2.9-2` |
| Loaded NVIDIA module | `615.71.09` |
| Installed NVIDIA userspace | `nvidia-utils 615.78.08-2` |
| Rust | Repository pin `1.98.1`, installed through rustup |
| Protobuf compiler | `libprotoc 36.1` |
| C/C++ toolchain | GCC `16.2.1`, Clang `23.1.1` |
| Build tools | CMake `4.4.4`, Ninja `1.13.2` |
| Python | System `3.14.7`; uv-managed Python 3.12 available for HADES |
| Node | `26.11.1` |

No existing HADES or ArangoDB system units or `arangod` on PATH were found.
Keep the existing shared Python environment untouched; use a dedicated service
environment. The Ubuntu package commands in [installation.md](installation.md)
are not applicable to this host.

## Local build workspace

Dependency sources and raw build logs belong in `.local-install/`, excluded
from Git. Keep source revisions, reproducible commands, required patches, and
verified outcomes in this document. Never place credentials in the record.

The selected ArangoDB source is the official `v3.12.12.1` tag at
`c10dcc0881833737262b11f250151ae19a3ae920`, the highest 3.12 release tag found
on 2026-10-18. Verify vector-index support in the built server before accepting
the database installation. Source: <https://github.com/arangodb/arangodb>.

Persistent data and model storage uses `/home/todd/Projects/HADES-local`, outside
the source checkout, on the large Projects volume. `~/.local/share/hades` links
there, and `~/.cache/hades` links to its `cache/` directory. These paths must
survive checkout replacement and `cargo clean`. The initial model download used
the home volume; relocate only the HADES-created directories, leaving other
applications' data and shared caches untouched. Set future uv downloads to a
cache on the Projects volume as well.

## Commands and results

### HADES builds

From the checkout root:

```bash
cargo build --release --locked --workspace -j 6
target/release/hades --version
target/release/hades-viewer --version
cargo build --locked --workspace -j 6
```

Both release and debug builds passed; both executables report `0.3.0`. The
debug build supports the isolated database/browser harness.
No HADES source changes were needed for the release build.

Release executables are installed in `~/.local/bin/`. Their SHA-256 hashes are
recorded in `.local-install/hades-binaries.sha256`.

### Python services and model assets

```bash
uv venv --python 3.12 services/.venv
UV_CACHE_DIR=/home/todd/Projects/HADES-local/cache/uv \
  uv pip install --python services/.venv/bin/python \
  -c services/constraints-jina-v4.txt -e './services[dev]'
UV_CACHE_DIR=/home/todd/Projects/HADES-local/cache/uv \
  uv pip check --python services/.venv/bin/python
```

Python `3.12.13` was selected. Installation and dependency consistency checks
passed for 134 packages, including PyTorch `2.14.1+cu130`, Transformers `4.52.4`,
PEFT `0.15.2`, gRPC `1.84.0`, and the repository-pinned Docling packages. Editable installation
generated the protobuf bindings. The first installation used uv's default cache;
the command above selects the larger volume for subsequent runs.
The complete observed versions are preserved in
[installation-thinkpad-python.txt](installation-thinkpad-python.txt). This is an
environment inventory, not a replacement for the repository's hashed CPU CI lock.

The initial unconstrained install selected Transformers `4.57.6` and PEFT
`0.21.2`. A short embedding returned a vector but emitted missing-adapter-key
warnings. Do not treat a successful HTTP response or the expected vector shape
as proof that model adapters loaded correctly. Pinning Transformers `4.52.4`
and PEFT `0.15.2` resolved the warnings; all 1,518 adapter tensors were compared
exactly against the pinned checkpoint, and a short CPU inference produced a
finite 2,048-dimensional vector. Package consistency and CLI PDF extraction
also passed after changing the dependencies.

Reproduce the model check using the services environment:

```bash
HF_HOME="$HOME/.cache/hades/huggingface" HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  services/.venv/bin/python scripts/install/check_jina_model.py \
  --model "$HOME/.local/share/hades/models/jina-v4/853c867b65b749f3c3c72a06868140d842e04f06" \
  --device cpu
```

Repeating with `--device cuda:0` after driver recovery passed all 1,518 adapter
checks and produced a finite 2,048-dimensional vector in 24.24 seconds,
including first-use CUDA kernel compilation and model loading.

Jina v4 was downloaded with `huggingface_hub.snapshot_download`, pinning revision
`853c867b65b749f3c3c72a06868140d842e04f06`, into
`~/.local/share/hades/models/jina-v4/<revision>`. Configure the embedder with that
local path, avoiding an unpinned model fetch during service startup. For the
running service, use this alias to the pinned snapshot:

```bash
ln -s "$HOME/.local/share/hades/models/jina-v4/853c867b65b749f3c3c72a06868140d842e04f06" \
  "$HOME/.local/share/hades/models/jinaai--jina-embeddings-v4"
```

Set `HADES_EMBEDDER_MODEL` to that absolute alias path. The Rust embedding client
recognizes `jinaai/jina-embeddings-v4` and local paths ending in
`jinaai--jina-embeddings-v4`; a directory named only by the snapshot hash fails
its model-identity check. The alias preserves the exact pinned weights.

CPU extraction passed on synthetic PDF and DOCX fixtures using
`DoclingExtractor(device='cpu', use_ocr=False)`, with
`HF_HOME=~/.cache/hades/huggingface`. Both returned nonempty text and
`docling-blocks-v1` reading representations. The smoke-test results are in
`.local-install/fixtures/extraction-result.json`. This verifies text extraction,
not scanned-page OCR or GPU inference.

### ArangoDB source setup

```bash
git clone --depth 1 --branch v3.12.12.1 --recurse-submodules \
  --shallow-submodules https://github.com/arangodb/arangodb.git \
  .local-install/arangodb
sudo pacman -S --needed openmp openblas
```

The source and pinned submodules downloaded successfully. The operator installed
`openmp 23.1.1-1.1` and `openblas 0.3.34-1` with pacman. Package installation
requires an interactive sudo password; do not record passwords in logs or this
guide.

The initial GCC configure failed in upstream `CheckCompilerVersion.cmake`:
only Clang initializes the minimum compiler version, and GCC leaves an invalid
comparison. Upstream requests Clang 19 or newer; use the installed Clang 23.
Clang configuration initially stopped at missing OpenMP. With both packages
installed, configuration passes. Complete configure/build logs are preserved in
`.local-install/logs/`.

Compatibility fixes are required with the host's libstdc++ 16. Apply the
recorded patches to the exact pinned source and RocksDB submodule before
configuring:

```bash
git -C .local-install/arangodb apply \
  ../../scripts/install/patches/arangodb-3.12.12.1-cachyos.patch
git -C .local-install/arangodb/3rdParty/rocksdb apply \
  ../../../../scripts/install/patches/rocksdb-392a91d-cachyos.patch
```

The jemalloc patch replaces the removed private `std::__throw_bad_alloc()`
helper with standard `throw std::bad_alloc()`. The logger needs an explicit
`<unistd.h>` include. Several ArangoDB and RocksDB headers need explicit
`<cstdint>` declarations, and cluster backup code needs its version header.
The Community graph build also needs guards around references to the
Enterprise-only `SmartGraphProvider`; Community configurations retain their
existing cursor queue implementations.
The Community `replaceOffsetInfo` fallback in `HandleViews.cpp` also needs
explicit `aql::` qualifiers for `Variable` and `functions`.
Preserve these patches rather than relying on unrecorded
edits in the dependency checkout. When repairing an already configured tree,
rerun CMake so jemalloc's copied build sources receive the patch.

The successful configure command is:

```bash
cmake -S .local-install/arangodb -B .local-install/arangodb-build-clang -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -DMAKE_JOBS=-j16 \
  -DCMAKE_INSTALL_PREFIX=/home/todd/.local/opt/arangodb-3.12.12.1 \
  -DUSE_ENTERPRISE=OFF -DUSE_GOOGLE_TESTS=OFF -DUSE_FRONTEND=OFF \
  -DUSE_FAIL_ON_WARNINGS=OFF -DUSE_IPO=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=/home/todd/Projects/HADES/services/.venv/bin/python
cmake --build .local-install/arangodb-build-clang --parallel 8
cmake --install .local-install/arangodb-build-clang
```

Keep V8 support enabled for the standard server/client tools. `USE_FRONTEND=OFF`
omits ArangoDB's bundled web administration application, not the HADES browser.
Disable upstream warning-as-error handling for the newer compiler and IPO to
bound linking cost. The CMake policy floor accommodates bundled older projects.
Use LLVM's `lld` linker: the initial GNU linker failed to resolve symbols
already present in the static ICU and ArangoDB libraries due to archive
ordering. `MAKE_JOBS=-j16` bounds nested third-party make builds; the outer
eight-worker Ninja limit alone does not control them. The nested limit was
raised from eight after checking CPU and memory headroom. Upstream configuration rewrites
`ARANGO-VERSION` from `3.12.12.1` to `3.12.12-1`; this generated normalization is
not part of the compatibility patch.

Build and user-local installation passed. The installed server reports
`3.12.12-1`, build repository `c10dcc088`, Clang `23.1.1`, RocksDB `9.6.0`,
V8 `12.1.165`, and FAISS `1.15.0`. Executables are under the prefix's `bin/`
and `sbin/`; JavaScript runtime assets are under `share/arangodb3/js`.

## Local configuration and services

Configuration lives in `~/.config/hades/`, with mode 0600 files and a private
directory. `hades.yaml` points to the laptop's database and extractor sockets
under `~/.local/share/hades/run/`, and embedding at `127.0.0.1:8087/v1`.
The fresh database is `Conveyance_KG`, initialized with `--seed empty`.
Initial statistics confirm zero research documents, chunks, and embeddings.
The [service configuration snapshot](installation-thinkpad-services.md) records
the exact units, environment settings, and YAML overrides, with secrets omitted.

`daemon.env` supplies the explicit configuration path, database name, and
database credentials. Separate restricted files hold the ArangoDB root password,
JWT secret, MCP token, and viewer password. Secrets are generated locally and
never printed in the installation log. For interactive CLI use in a trusted
shell, load the generated environment with:

```bash
set -a
. "$HOME/.config/hades/daemon.env"
set +a
```

User units are installed under `~/.config/systemd/user/`:

| Unit | Current state and configuration |
| --- | --- |
| `hades-extractor.service` | Enabled and running; dedicated Python environment, CPU, OCR disabled, `extractor.env` |
| `hades-embedder.service` | Enabled and running; pinned Jina alias, GPU 0, batch 1, verified 4,096-token profile in `embedder.env` |
| `hades-daemon.service` | Enabled and running; explicit database, Unix socket, loopback guided-reading MCP on port 10444 |
| `hades-viewer.service` | Enabled and running; loopback port 8088, matching reading socket/database, password-file authentication |
| `hades-arangodb.service` | Enabled and running; authenticated Unix-only server, `arangod.conf`, data on the Projects volume |

The extractor startup was verified in the user journal, and the installed
Rust CLI successfully extracted the synthetic PDF
through its Unix socket:

```bash
HADES_CONFIG="$HOME/.config/hades/hades.yaml" \
  "$HOME/.local/bin/hades" --db Conveyance_KG \
  extract .local-install/fixtures/install-fixture.pdf
```

The extract command does not require the research database to exist. These
user services run as the logged-in owner, avoiding Olympus-specific Unix groups
and root-managed unit assumptions. The four initially enabled services started
successfully after reboot/login. The embedder was enabled afterward and its
explicit service restart and GPU requests passed.

The installed daemon passed `scripts/check_daemon_ready.py` against
`Conveyance_KG`. The viewer returned HTTP 401 without credentials and HTTP 200
with its password. Authenticated MCP initialization and tool discovery passed
at `http://127.0.0.1:10444/mcp`. These checks do not establish GPU readiness.

Open `http://127.0.0.1:8088/reading` for guided reading. HTTP Basic authentication
accepts any username; the local password is in `~/.config/hades/viewer-password`.
Both browser pages were checked in Chromium without JavaScript errors.

### Service readiness and restarts

`After=hades-daemon.service` orders process startup but does not wait for its
socket. The first combined restart caused the viewer to exit once and recover
on its retry. The viewer unit now includes this readiness step:

```ini
ExecStartPre=/usr/bin/python3 /home/todd/Projects/HADES/scripts/check_daemon_ready.py --socket /home/todd/.local/share/hades/run/hades.sock --database Conveyance_KG --timeout 60
```

A subsequent combined database/daemon/viewer restart passed with zero viewer
restarts. Authentication, MCP discovery, and the stored empty research schema
also survived the operator's subsequent reboot/login.

### Database initialization

Initialize only empty storage, using `ARANGODB_DEFAULT_ROOT_PASSWORD` from the
private root-password file and `arangod --configuration <path> --database.init-database true`.
Keep the password in the process environment, never command arguments or logs.
Enable `authentication` and `authentication-unix-sockets`, supply the private
JWT key file, and use `endpoint = unix://<absolute socket path>`.
Set top-level `vector-index = true`.

Create the `hades` database user through the authenticated `_system` user API,
grant `_system` and `*` read/write access for this local development account,
then load `daemon.env` and run:

```bash
hades --db _system db create-database Conveyance_KG
hades --db Conveyance_KG db schema init --seed empty
hades --db Conveyance_KG db stats
```

Use `LC_ALL=C.UTF-8` during initialization and in `hades-arangodb.service`.
The initial shell used this locale while systemd inherited `en_US.UTF-8`,
causing a fatal stored-language mismatch. Aligning the service locale fixed
startup without resetting data or disabling language checks.
Use current ArangoSearch settings: `commit-threads = 2`,
`consolidation-threads = 2`, and `execution-threads-limit = 2`.
The old `threads`/`threads-limit` options did not bound execution threads.

## Findings and fixes

### NVIDIA driver versions must agree

Initially, host-level `nvidia-smi` failed with `Driver/library version mismatch`.
The loaded module and installed userspace differed, and the installed kernel
was newer than the running kernel. After the operator patched and rebooted,
the running kernel was `7.2.9-2-cachyos`; both NVIDIA kernel and userspace
reported `615.78.08`. The RTX PRO 5000 Blackwell Laptop GPU exposes 24,463 MiB
of memory. The initial sandbox-level error was less specific; host-level
inspection established the mismatch and confirmed its resolution.

### GPU HTTP and retrieval verification

The `thinkpad-initial` profile uses `cuda:0`, fp16, batch size 1, and a
4,096-token ceiling. Eight tokens are reserved for task prompts, leaving
4,088 tokens counted with special tokens for raw input.

| HTTP case | Result | Elapsed | Sampled total GPU memory peak |
| --- | --- | --- | --- |
| Short passage, first request after service restart | One finite 2,048-dimensional vector | 5.79 s | 9,235 MiB |
| 4,088-token input | One finite vector | 0.57 s | 12,305 MiB |
| Same input with late chunking | 13 finite vectors | 0.82 s | 12,305 MiB |
| 4,200-token input | HTTP 400, `PE_INPUT_TOO_LARGE`, ceiling 4,088 | 0.02 s | 12,305 MiB |

Memory was sampled with `nvidia-smi` every 100 ms and includes desktop use and
allocator reservations; it is not an exact tensor-allocation measurement.
These are smoke-test observations, not a throughput benchmark or maximum-window
search. The larger-context work in #208 remains separate.

The installed release CLI then ingested three synthetic text documents through
the real CPU extractor and GPU embedder on a separate Unix-only ArangoDB server.
The test used `with_temp_db`, initialized an empty schema, and asked three
semantic questions. Each ranked its matching document first: archive revision
history (cosine 0.632), bread fermentation (0.862), and Saturn/Titan (0.760).
Creation of a 2,048-dimensional vector index also passed. The disposable
database was removed; `Conveyance_KG` still contains zero documents, chunks,
and embeddings. Reports are under `.local-install/verification/`.

### Host tuning observations

ArangoDB starts successfully but recommends increasing `vm.max_map_count` from
1,048,576 to 1,280,000 and changing transparent huge pages from `always` to
`madvise` or `never`. No system-wide tuning was applied during this installation.
Reassess these settings when validating larger local workloads.

## Reproducing integration checks

These commands launch separate disposable databases, never `Conveyance_KG`:

```bash
CARGO_TARGET_DIR="$PWD/target" python3 scripts/test_isolated_database.py \
  --arangod "$HOME/.local/opt/arangodb-3.12.12.1/sbin/arangod" \
  --command-timeout 1800
CARGO_TARGET_DIR="$PWD/target" python3 scripts/test_isolated_database.py \
  --arangod "$HOME/.local/opt/arangodb-3.12.12.1/sbin/arangod" \
  --reading-e2e-python "$PWD/services/.venv/bin/python" \
  --reading-node /usr/bin/node \
  --reading-playwright "$PWD/.local-install/browser/node_modules/playwright" \
  --reading-chromium /usr/bin/chromium \
  --reading-model-cache "$HOME/.cache/hades/huggingface" \
  --command-timeout 1800
```

The browser harness uses Playwright `1.64.1` installed locally under
`.local-install/browser/`, with the system Chromium executable. Its PDF/DOCX
browser/MCP flows and saved-reading recovery with extraction offline passed.
Preserve logs, JSON reports, and screenshots from `/tmp/hades-tests-*` under
`.local-install/verification/` before reboot; `/tmp` is volatile.

## Verification status

- Host inventory: recorded.
- ArangoDB source build, installation, and vector-index creation: passed.
- HADES release build and executable version checks: passed.
- Service-free Rust unit tests: 786 passed, 0 failed, 1 ignored, using
  `cargo test --locked --workspace --lib --bins -j 6`.
- Fresh authenticated database and empty schema: passed.
- Isolated database harness: passed all phases, including 69 database contracts,
  77 codebase command tests, 51 CLI lifecycle tests, and 20 graphable unit tests.
  The missing-socket prerequisite check also failed as required, proving that
  database tests do not silently skip a missing server.
- Browser pages, authentication, and local MCP discovery: passed.
- Isolated PDF/DOCX browser/MCP workflow and offline recovery: passed.
- CPU extraction of synthetic PDF and DOCX: passed.
- Focused Python embedding/extraction contracts after dependency pinning:
  57 passed (`test_embedding_tensors.py`, `test_embedding_size_fields.py`,
  `test_embedding_ownership.py`, and `test_docling_backend.py`).
- Pinned Jina CPU compatibility: passed, including all 1,518 adapter tensors.
- GPU adapter loading, HTTP embedding, late chunking, size rejection, and
  installed-CLI ingestion/semantic retrieval: passed.
- Service restart persistence: passed. Reboot/login startup passed for ArangoDB,
  extractor, daemon, and viewer. Embedder enabled afterward and restart-tested.

## Codex chat deployment after PR #218

On 2026-10-18, merge commit `557477973665c5dc53ee9ef11df02f282f63c4dc`
was built with:

```bash
cargo build --release --locked -p hades-cli -p hades-frontend -j 6
```

The release `hades` and `hades-viewer` binaries were installed atomically into
`~/.local/bin/`. Prior binaries and HADES YAML were preserved in
`~/.local/share/hades/backups/pr218-20261018T180539Z/`. Both application services
were restarted; ArangoDB, extraction, and embedding stayed running.

Codex chat uses `/usr/bin/codex` version 0.163.0, the existing authenticated
`~/.codex` history, and a dedicated `~/.local/share/hades/reading-workspace/`.
The MCP token is supplied through a private daemon-only environment file and
systemd drop-in. Exact configuration is in the
[service snapshot](installation-thinkpad-services.md#codex-reading-chat).

Installed checks confirmed matching release hashes and running executable
paths, all five services enabled/active, daemon database readiness, authenticated
browser HTTP 200, unauthenticated HTTP 401, and MCP discovery. Both browser
pages loaded without JavaScript errors. The reading chat API reported enabled
and successfully listed existing Codex conversations. These deployment checks
did not resume a conversation, send a model message, or write research records;
they are distinct from the isolated chat contracts described in
[Reading with Codex](codex-reading.md#limits-and-validation).

Local evidence is saved in `.local-install/reading-chat-deployment.json`,
`.local-install/installed-reading-chat-check.json`,
`.local-install/installed-services-check.json`, and browser screenshots under
`.local-install/browser/`. These files are intentionally outside Git.

## Reading desk and permission deployment after PRs #219 and #220

On 2026-10-18, merged main commit
`15e2c9565a9c051ca8a2b17c625cc6005b3aa0fa` was built from the isolated checkout
`/tmp/hades-deploy-219-220`, preserving the development checkout and its local
installation notes:

```bash
CARGO_TARGET_DIR=/home/todd/Projects/HADES/target \
  cargo build --release --locked -p hades-cli -p hades-frontend -j 6
```

Both release executables were backed up to
`~/.local/share/hades/backups/pr219-220-20261018T200827Z/` and atomically replaced
in `~/.local/bin/`. The viewer and daemon were stopped for the replacement, then
the daemon was started and checked for database readiness before starting the
viewer. Configuration was unchanged. ArangoDB, the extractor, and the embedder
kept their existing process IDs throughout the deployment.

The preflight found six saved readings, three saved conversations, and no active
conversation turns. Installed checks confirmed all five services enabled/active,
matching running release binaries, authenticated HTTP 200, unauthenticated HTTP
401, MCP discovery, and Codex history discovery. All six reading identities and
source hashes remained present. The browser loaded without JavaScript errors,
showed removal/archive controls, and widened both side panels. At 1800px, chat
expanded from 576px to 684px and the reading desk from 220px to 324px.

Refresh an existing browser tab to load the new assets. In a saved conversation,
**Always allow reads in this conversation** enables remembered read approval;
**Revoke remembered read permission** removes it. Installation does not enable
that preference automatically. Use **Resume** if the conversation was disconnected
by the daemon restart. Reading removal/restoration and duplicate prevention are
described in [Guided document reading](guided-reading.md).

Deployment checks issued no research mutations or model messages. Functional
mutation tests ran separately against disposable databases before merging.
Local evidence is in `.local-install/reading-desk-deployment.json`,
`.local-install/installed-reading-desk-check.json`, and
`.local-install/browser/installed-reading-desk-wide.png`. The deployment manifest
is also stored with the backup binaries. These local artifacts are outside Git.

## Current operating state

All five user services are enabled and active. Guided reading is available at
`http://127.0.0.1:8088/reading`; the embedder serves
`http://127.0.0.1:8087/v1`, and MCP uses `http://127.0.0.1:10444/mcp`.
The model unloads after 900 idle seconds and reloads on demand. Check
`systemctl --user status hades-embedder.service` and `/v1/models` when diagnosing
embedding availability. Codex chat is enabled in the reading desk; open a saved
reading and select **Chat** to create or explicitly resume a conversation.
No Olympus data has been restored.
