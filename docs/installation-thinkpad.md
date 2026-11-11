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
| `hades-embedder.service` | Enabled and running; pinned Jina alias, GPU 0, batch 1, 20,000-token ceiling in `embedder.env` ([bisection](#sequence-ceiling-bisection-2026-11-09)) |
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
search. The larger-context work in #208 remains separate. On 2026-11-09 the
ceiling was raised to 20,000 tokens after the
[sequence ceiling bisection](#sequence-ceiling-bisection-2026-11-09).

The installed release CLI then ingested three synthetic text documents through
the real CPU extractor and GPU embedder on a separate Unix-only ArangoDB server.
The test used `with_temp_db`, initialized an empty schema, and asked three
semantic questions. Each ranked its matching document first: archive revision
history (cosine 0.632), bread fermentation (0.862), and Saturn/Titan (0.760).
Creation of a 2,048-dimensional vector index also passed. The disposable
database was removed; `Conveyance_KG` still contains zero documents, chunks,
and embeddings. Reports are under `.local-install/verification/`.

### Sequence ceiling bisection, 2026-11-09

On 2026-11-09 the embedder's sequence ceiling was bisected on the RTX PRO 5000
(24,463 MiB), running the `52d04d4` services code, with real prose rather than
repeated tokens. The corpus was the 113 `docs/**/*.md` files at `52d04d4`,
concatenated in sorted order: 215,490 tokens, sha256 `691ef97f62e9e131...`. Each
candidate ran in a fresh service process. A drop-in, `ceiling.conf`, appended a
second `EnvironmentFile` setting `HADES_EMBEDDER_MAX_TOKENS`, and `/v1/models` had
to report the candidate before any request. A drop-in `Environment=` line would
not work, because `EnvironmentFile` values override it and `embedder.env` sets the
ceiling.

Each candidate sent three `retrieval.passage` requests with batch size 1: the
corpus prefix at the ceiling minus 8 tokens, the same prefix late-chunked with
500-token chunks and 200-token overlap, and a prefix at the ceiling plus 200 tokens,
which must return 400 `PE_INPUT_TOO_LARGE`. `nvidia-smi` sampled whole-card memory
every 100 ms. A candidate passed only if the first two requests both returned 200
with finite 2,048-dimensional vectors and the unit did not restart.

| Ceiling | Result | Late vectors | Sampled peak |
| --- | --- | --- | --- |
| 12,000 | Pass | 40 | 18,663 MiB |
| 16,000 | Pass | 53 | 21,231 MiB |
| 20,000 | Pass | 66 | 22,491 MiB |
| 22,000 | Pass | 73 | 23,823 MiB |
| 23,000 | Pass | 76 | 24,041 MiB |
| 24,000 | Fail, 503 `PE_BACKEND_OOM` on both requests | None | 23,301 MiB |

Candidates ran in the order 12,000, 16,000, 20,000, 24,000, 22,000, 23,000; the
bisection stopped at a 1,000-token gap. The late request starts from the plain
request's reserved peak, held by the PyTorch caching allocator, so the late
figures are floors rather than independent peaks; the column shows the higher of
the two. At 24,000 the plain request failed at a sampled 20,515 MiB and the late
request at 23,301 MiB, each on a single 6.79 GiB allocation. The OOM returned 503
without killing the process. Every over-ceiling request returned 400
`PE_INPUT_TOO_LARGE`, and no candidate restarted the unit.

The service bound 2.6 to 2.7 s after each restart and loaded the model in 5.7 to
7.2 s, inside the first request. That request took 24.6 s at 12,000, with cold
caches, and 10.7 to 17.0 s afterwards; late-chunked requests took 3.0 to 10.2 s.
Before each load the card showed 131 MiB in use by the desktop. An idle embedder
process with the model unloaded holds about 360 MiB more for its CUDA context
(497 MiB total, observed earlier on 2026-11-09). The records, including the corpus
file list and full hash, are in
`docs/evidence/thinkpad-ceiling-bisect-2026-11-09.json`.

The operator chose 20,000 tokens as the operating ceiling on 2026-11-09. Its peak
of 22,491 MiB leaves about 2 GiB of headroom on a card that also drives the
desktop.
23,000 passed with 422 MiB to spare and is not an operating value.

Clients read the ceiling from `/v1/models` and need no configuration change. With
this laptop's `chunking.prechunk` settings (1,000-token overlap, 256-token safety
margin, no cap), the prechunk window becomes 20,000 - 256 = 19,744 tokens. The
client YAML's `embedding.model.max_tokens` is not read by the client and does not
limit requests.

To apply the ceiling, confirm the checkout and that no conversation is active, then
replace the bisection override with the `embedder.env` value:

```bash
cd ~/Projects/HADES
git diff --stat 52d04d4e5d66625466ea2b1b2c60fda19ed06f2e HEAD -- services/
git status --short -- services
python3 .local-install/preflight-224.py
sed -i 's/^HADES_EMBEDDER_MAX_TOKENS=.*/HADES_EMBEDDER_MAX_TOKENS=20000/' \
  ~/.config/hades/embedder.env
grep '^HADES_EMBEDDER_MAX_TOKENS=' ~/.config/hades/embedder.env
rm -f ~/.config/systemd/user/hades-embedder.service.d/ceiling.conf \
  ~/.config/systemd/user/hades-embedder.service.d/ceiling.env
systemctl --user daemon-reload
systemctl --user restart hades-embedder.service
curl -s --retry 60 --retry-connrefused --retry-delay 2 http://127.0.0.1:8087/v1/models
systemctl --user cat hades-embedder.service
```

`diff` and `status` must both print nothing: the first proves the committed services
tree equals the verified `52d04d4` tree, whichever later docs-only commit `HEAD` is
on, and the second that nothing under `services/` is modified locally. `grep` must
print `HADES_EMBEDDER_MAX_TOKENS=20000`, `/v1/models` must
report a `max_seq_length` of 20000, and `systemctl --user cat` must list no
`ceiling.conf`. This restart also moves the embedder onto the `52d04d4` services
code, so it replaces [Restart the embedder for 52d04d4](#restart-the-embedder-for-52d04d4)
if that has not been run. The [rollback](#roll-back-52d04d4) worktree procedure is
unaffected: it changes only the working directory, and the ceiling in
`embedder.env` applies to both revisions.

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

## Deployment of main 52d04d4

Superseded by [Managed workspace, 2026-11-xx](#managed-workspace-2026-11-xx), which
installs main `09df08f` over the same `f930fbc` executables instead.

On 2026-11-09, main commit `52d04d4e5d66625466ea2b1b2c60fda19ed06f2e`, the merge
of PR #203, was built and verified in a separate worktree. The commit adds merged
PRs #203 and #226 through #248, except #232, to the installed build.

The executables it replaces were built from
`f930fbca1c5362f0e8e337ceeb1a1f9d21173cec` (PR #224) and installed on 2026-10-19.
Their hashes match `.local-install/reading-desk-deployment.json`, and their backup
is `~/.local/share/hades/backups/pr224-20261019T055443Z/`.

```bash
CARGO_TARGET_DIR=~/Projects/hades-targets/deploy-main \
  cargo build --release --locked -p hades-cli -p hades-frontend
```

On the same commit and target directory, `cargo fmt --all -- --check` and
`cargo clippy --workspace --all-targets --locked -- -D warnings` passed.
`cargo test --workspace --lib --bins --locked` passed 837 tests with 1 ignored,
and the service-free `tests/` targets run by `.github/workflows/ci.yml` passed
173 tests. Both executables report version 0.3.0.

| Executable | SHA-256 |
| --- | --- |
| `hades` | `d25cffb4772c7a17be8b4b5e789e2d927914c93f916a754321055c59a6141c49` |
| `hades-viewer` | `e84d25bcb693f2a7e5e657621364a937dd89abffb1071dc386a2e39c20b2ff9b` |

### Supporting services for 52d04d4

The extractor and embedder units run Python from the development checkout,
`~/Projects/HADES/services` and its `.venv`, not from the installed executables.
Replacing the executables therefore restarts neither. The viewer's readiness
step, `scripts/check_daemon_ready.py`, is unchanged.

Between `f930fbc` and `52d04d4`, `services/extraction`, the Python requirements,
constraints and package files, and `services/systemd` are unchanged. The embedder
changed: `services/embedding/http_server.py` and `jina_v4.py` were modified and
13 modules were added, with no new dependencies. Every package they import is
already in `services/.venv`. Over HTTP, the only behaviour change is that each
model load first releases cached CUDA memory; the new registered-loading path is
not reachable from the service configuration. `deploy/systemd/README.md` and
`hades-embedder-profile` also changed, but this laptop runs the non-templated
`hades-embedder.service` and does not install the profile switcher.

The development checkout was fast-forwarded to `52d04d4` on 2026-11-09, while the
embedder process started on 2026-10-18 still runs the earlier code. Restart the
embedder in this deployment, so the running process matches the code on disk;
otherwise a later automatic restart would switch code without a record. Do not
reinstall `services/.venv`. The extractor's code is unchanged and it needs no
restart.

### Install 52d04d4

Run these from the development checkout, in one shell, one at a time, and stop at
the first failure. The hash table above is the source of truth for the build:
`sha256sum -c` fails on any mismatch, before anything is stopped. The preflight
must report zero active conversations.

```bash
cd ~/Projects/HADES
BUILD="$HOME/Projects/hades-targets/deploy-main/release"
printf '%s  %s\n' \
  d25cffb4772c7a17be8b4b5e789e2d927914c93f916a754321055c59a6141c49 "$BUILD/hades" \
  e84d25bcb693f2a7e5e657621364a937dd89abffb1071dc386a2e39c20b2ff9b "$BUILD/hades-viewer" |
  sha256sum -c -
snapshot() {
  python3 -c 'import json, runpy
s = runpy.run_path(".local-install/preflight-224.py")["snapshot"]()
print(json.dumps(s, sort_keys=True))'
}
BACKUP="$HOME/.local/share/hades/backups/main-52d04d4-$(date -u +%Y%m%dT%H%M%SZ)"
install -d -m 700 "$BACKUP"
cp -p ~/.local/bin/hades ~/.local/bin/hades-viewer ~/.config/hades/hades.yaml "$BACKUP/"
snapshot > "$BACKUP/before.json"
python3 .local-install/preflight-224.py
systemctl --user stop hades-viewer.service
systemctl --user stop hades-daemon.service
install -m 755 "$BUILD/hades" ~/.local/bin/.hades.new
install -m 755 "$BUILD/hades-viewer" ~/.local/bin/.hades-viewer.new
mv -f ~/.local/bin/.hades.new ~/.local/bin/hades
mv -f ~/.local/bin/.hades-viewer.new ~/.local/bin/hades-viewer
systemctl --user daemon-reload
systemctl --user start hades-daemon.service
python3 scripts/check_daemon_ready.py --socket ~/.local/share/hades/run/hades.sock \
  --database Conveyance_KG --timeout 60
systemctl --user start hades-viewer.service
```

Both services are stopped because a running daemon would keep executing the
replaced file. Both new files are staged before either rename, and each `mv` is a
same-directory rename, so `~/.local/bin` never holds a partial executable.

### Post-install checks for 52d04d4

Run these in the same shell, before anyone opens a reading chat: the final
comparison includes live conversation state.

```bash
sha256sum ~/.local/bin/hades ~/.local/bin/hades-viewer
for unit in hades-daemon hades-viewer; do
  readlink "/proc/$(systemctl --user show "$unit.service" -p MainPID --value)/exe"
done
systemctl --user is-active hades-arangodb hades-extractor hades-embedder \
  hades-daemon hades-viewer
python3 .local-install/check-installed-services.py
node .local-install/check-installed-reading-desk.cjs
node .local-install/check-installed-workspace-chat.cjs
snapshot > "$BACKUP/after.json" && cmp "$BACKUP/before.json" "$BACKUP/after.json"
```

The hashes must match the table above. Each `readlink` must print the
`~/.local/bin` path without `(deleted)`, and all five units must be active. `cmp`
must print nothing, which confirms that readings, conversation identities,
permissions and saved exchanges are unchanged. `check-installed-workspace-chat.cjs`
chooses its conversation from `.local-install/preflight-224.json` (2026-10-19), and
fails if that conversation has since been removed. Do not use
`check-installed-reading-chat.py` for this deployment: it compares the installed
executables with `~/Projects/HADES/target/release`, which still holds the `f930fbc`
build. The first two commands above replace its executable check.

### Restart the embedder for 52d04d4

Run this after the post-install checks, in the same shell. The first two commands
confirm that the checkout holds exactly the verified services code, and both must
print nothing; stop otherwise. `diff` proves the committed services tree equals the
`52d04d4` tree, whichever later docs-only commit `HEAD` is on, and `status` that
nothing under `services/` is modified locally.

```bash
cd ~/Projects/HADES
git diff --stat 52d04d4e5d66625466ea2b1b2c60fda19ed06f2e HEAD -- services/
git status --short -- services
python3 .local-install/preflight-224.py
systemctl --user restart hades-embedder.service
curl -s --retry 60 --retry-connrefused --retry-delay 2 http://127.0.0.1:8087/v1/models
cp -p .local-install/verification/gpu-http.json "$BACKUP/gpu-http-before.json"
services/.venv/bin/python .local-install/check-gpu-http.py
```

The retry window allows up to two minutes for the service to bind, and
`/v1/models` answers before the model is loaded. It must report `cuda:0` and the
configured ceiling: 4096 before the
[20,000-token change](#sequence-ceiling-bisection-2026-11-09), 20000 after it.
`check-gpu-http.py` asserts the 4,096 ceiling, so run it only before that change.
It sends four embedding requests, including one over the ceiling that must fail
with `PE_INPUT_TOO_LARGE`, and rewrites `.local-install/verification/gpu-http.json`,
which is why the earlier report is copied first.

### Roll back 52d04d4

The restore uses the backup directory that holds the pre-deployment executables,
chosen by content rather than date, because a repeated install can leave a newer
backup holding the new build or a mix. These are the `f930fbc` hashes recorded for
the PR #224 install in `.local-install/reading-desk-deployment.json`, and they
matched `~/.local/bin` on 2026-11-09 before this install:

| Executable | SHA-256 before this deployment (`f930fbc`) |
| --- | --- |
| `hades` | `cf0573ebbb42893cb7f1ccf5f2a69caf506df57cbe422e0fcd705e31c3ec6b42` |
| `hades-viewer` | `6efac5881ed6b5ea7ac30c0a7a6131979e6b224fc14618104d9a354c6cac2c72` |

Find the directory first, from any shell. The last line must print its path before
you continue; if it prints nothing, no backup holds that build, so stop:

```bash
BACKUP=""
for dir in ~/.local/share/hades/backups/main-52d04d4-*; do
  printf '%s  %s\n' \
    cf0573ebbb42893cb7f1ccf5f2a69caf506df57cbe422e0fcd705e31c3ec6b42 "$dir/hades" \
    6efac5881ed6b5ea7ac30c0a7a6131979e6b224fc14618104d9a354c6cac2c72 "$dir/hades-viewer" |
    sha256sum -c --status - 2>/dev/null && BACKUP="$dir" && break
done
test -n "$BACKUP" && echo "$BACKUP"
```

To restore the executables, stop the viewer and then the daemon, and repeat the two
`install` and two `mv` commands with `$BACKUP/hades` and `$BACKUP/hades-viewer` as
sources. Then start the daemon, check its readiness, and start the viewer, as
above. Reading records written after the install stay in `Conveyance_KG`.

Restoring the executables leaves the embedder on the `52d04d4` services code in
the development checkout. Before this deployment, the embedder process, started
2026-10-18 08:39:52 CDT, ran services code from the initial clone,
`7423624e029c6f117e122456c06a721bf407cb82`. Its Python and package files under
`services/` are identical to `f930fbc`; only `services/README.md` and
`services/constraints-jina-v4.txt` differ, and neither is loaded at runtime. After
the embedder restart above, return it to that code by serving it from a worktree
at `f930fbc`, leaving the development checkout untouched:

```bash
git -C ~/Projects/HADES worktree add ~/Projects/hades-worktrees/embedder-f930fbc \
  f930fbca1c5362f0e8e337ceeb1a1f9d21173cec
mkdir -p ~/.config/systemd/user/hades-embedder.service.d
cat > ~/.config/systemd/user/hades-embedder.service.d/rollback.conf <<'EOF'
[Service]
WorkingDirectory=%h/Projects/hades-worktrees/embedder-f930fbc/services
EOF
systemctl --user daemon-reload
systemctl --user restart hades-embedder.service
curl -s --retry 60 --retry-connrefused --retry-delay 2 http://127.0.0.1:8087/v1/models
readlink "/proc/$(systemctl --user show hades-embedder.service -p MainPID --value)/cwd"
```

The unit runs `python -m embedding.http_server`, which searches its working
directory first, so `embedding` loads from the worktree even though the shared
venv's editable install points at the development checkout. `readlink` must print
the worktree's `services` directory. The environment file, venv and model weights
are shared and unchanged in either direction. To return to main, remove
`rollback.conf`, run `systemctl --user daemon-reload`, and restart the unit. Other
drop-ins in that directory, such as a ceiling override, are independent of
`rollback.conf`; `systemctl --user cat hades-embedder.service` lists every drop-in
in effect.

## Managed workspace, 2026-11-xx

Draft for the operator's run. Replace `2026-11-xx` with the run date when the run is
recorded; until then this section is the plan, not a record. It carries out Part C of
the managed database design,
[db-setup-door.md](specs/db-setup-door.md#part-c-the-laptop-runbook-and-first-corpus-docs-operator-steps):
install main `09df08f`, register the Jina snapshot, switch the embedder to the verified
provider, create `Conveyance_KG_v2` with `hades db setup`, back it up, move the reading
desk into it, ingest the first corpus, and back it up again. `Conveyance_KG` stays
until the [retirement gate](#retirement-gate) is passed.

The steps are an installation script,
[`scripts/install/thinkpad-managed-workspace.sh`](../scripts/install/thinkpad-managed-workspace.sh),
with a standard-library helper beside it, `managed_workspace.py`, for the JSON checks
and the backup copy. This section says how to run it, what to look at after each step,
and what phase 2 records.

### Running the installer

Run it from a checkout that holds the script, as the account that owns the user
units. It changes to the checkout it installs into, `HADES_MW_CHECKOUT`
(`~/Projects/HADES`), before any step, because the private checks open
`.local-install/` relative to it. `check` is read-only and always safe: it prints the
resolved values, the
preconditions and each step's state. Then run the steps one at a time, each first with
`--dry-run`, or run `all`:

```bash
scripts/install/thinkpad-managed-workspace.sh check
scripts/install/thinkpad-managed-workspace.sh --dry-run install-main
scripts/install/thinkpad-managed-workspace.sh install-main
scripts/install/thinkpad-managed-workspace.sh all
```

| Step | What it does |
| --- | --- |
| `install-main` | installs the `09df08f` executables; holds `services/` at the script's commit |
| `register` | registers the approved snapshot in `$DATA/model-registry` |
| `configure` | private roots, the two YAML sections, `embedder-verified.env` |
| `switch-embedder` | starts the verified provider unit; disables the legacy one |
| `setup-db` | `hades db setup run`, then `inspect` and `diagnose` |
| `backup-1` | first combined backup: database, setup ledger, registry |
| `move-desk` | final `Conveyance_KG` backup, then the reading-desk move |
| `ingest` | ingests the Markdown and text of `docs/`; asks two questions |
| `backup-2` | second combined backup, verified in isolation |

How it behaves:

- Every step tests its own postcondition first and prints `already done` without
  acting. A missing precondition is a refusal in one line, exit 1. A file that exists
  with other content (the YAML sections, the env file, the unit, a drop-in) is refused,
  never replaced. Re-running `all` after a failure resumes at the first undone step.
- `--dry-run` prints every command with its resolved values and runs nothing;
  preconditions that are not met yet are reported and the printing goes on. In printed
  commands, `helper` is `python3 scripts/install/managed_workspace.py`, `DATA` is the
  resolved data directory, and `~` is the home directory.
- `install-main`, `switch-embedder` and `move-desk` stop or switch services and ask for
  confirmation before they write or stop anything; `--yes` skips the question. If one
  of them fails while services are stopped or switched, it says so on exit and names
  the way back.
- Values are in one block at the top of the script, each overridable by its
  `HADES_MW_*` variable. Paths come from `$HOME` and `readlink -f ~/.local/share/hades`:
  the provider catalog, the registry and the setup ledger refuse a symlink ancestor, so
  everything they read is written from the resolved directory.
- The script loads `daemon.env`, which supplies the database account and the socket
  overrides. The setup ledger hashes the endpoint, so `hades db setup` always runs with
  it loaded.
- Each run that acts writes its evidence to a new directory,
  `$DATA/managed-workspace/<UTC timestamp>/`, with `commands.log` listing every command
  run. Step markers live in `$DATA/managed-workspace/state/`. The record holds copies of
  `hades.yaml` and `daemon.env`, which carry credentials, so it stays private: phase 2
  copies only selected, sanitized values into `docs/evidence/`.

The fixed values come from the design and
[the verified-server qualification](evidence/verified-server-thinkpad-2026-11-10.json):

| Value | Setting |
| --- | --- |
| Provider ID | `laptop-jina` |
| Model revision | `853c867b65b749f3c3c72a06868140d842e04f06` |
| Expected manifest MD5 | `446160668af00248832ed8cf5d8d4432` (2026-10-21 and 2026-11-10 evidence) |
| Installation ID | `laptop-2026-11` |
| Database | `Conveyance_KG_v2` |
| Operation ID | `conveyance-kg-v2-1` |
| Generation | `2026-11-10-g1`, a label; a retry repeats it exactly |
| Late policy | context cap 16,384; window overlap 1,000; passages 500 with 200 overlap |
| Provider ceiling | 20,000 tokens, idle timeout 0 |

### install-main

Main commit `09df08febe62570a883563030e9c1cf4d9f9aabc`, the merge of PR #252, replaces
the `f930fbc` executables installed on 2026-10-19. It carries every change merged since
then, including the verified provider service, the widened setup inventory and
`hades db setup` (PRs #252 to #255). The [52d04d4 deployment](#deployment-of-main-52d04d4)
was not installed; this step uses its procedure with the new commit and hashes.

The build used a detached worktree at `09df08f`, toolchain 1.98.1 and
`RUSTFLAGS=-Dwarnings`:

```bash
CARGO_TARGET_DIR=~/Projects/hades-targets/deploy-main \
  cargo build --release --locked -p hades-cli -p hades-frontend
```

On the same commit and target directory, `cargo fmt --all -- --check` and
`cargo clippy --workspace --all-targets --locked -- -D warnings` passed.
`cargo test --workspace --lib --bins --locked` passed 849 tests with 1 ignored, and the
23 service-free `tests/` targets run by `.github/workflows/ci.yml` passed 184 tests.
Both executables report version 0.3.0.

| Executable | SHA-256 |
| --- | --- |
| `hades` | `633b68a1a9c1ce919d2c22ac227e07cb48ccb5bc071ff1926057e9ab44774b4c` |
| `hades-viewer` | `22ac0c56a12a16b79af8f7c2c84ae9116efbeaa6a5f5e825559f30f3f494131f` |

The step checks the build's hashes and refuses unless `~/.local/bin` holds either the
`f930fbc` executables, which the rollback restores, or this build. It then follows the
#249 procedure: a backup directory `~/.local/share/hades/backups/main-09df08f-*` with
the old executables, `hades.yaml` and a `before.json` snapshot; the preflight, which
must report zero active conversations; the viewer and the daemon stopped; a staged
install; the daemon's readiness check; and the
[post-install checks](#post-install-checks-for-52d04d4) with the new hashes. It records
the backup directory as the marker `install-main.backup` before stopping anything, and
`install-main.checked` once the checks pass. A re-run after a failure finds the new
executables and the backup marker, starts the daemon and the viewer if they are
stopped, and repeats the checks. If the executables were installed some other way, the
step refuses and names the marker to write after running the checks by hand. Look at:

- the hash and executable checks (`install-sha256.txt`, `install-exe.txt`, no
  `(deleted)`), all five units active, and a silent `cmp` of the before and after
  snapshots. `check-installed-workspace-chat.cjs` chooses its conversation from
  `.local-install/preflight-224.json` (2026-10-19) and fails if that conversation has
  since been removed.

The step then holds the checkout's `services/` and `scripts/check_daemon_ready.py` at
the commit the script runs from, not at `09df08f`: the change that added this script
also changed `services/` (the verified server's SIGTERM handling), and the verified
unit runs `python -m embedding.verified_server` from the checkout while its template
comes from the script's own tree. That commit must contain `09df08f`, whose executables
are installed, with those paths committed; otherwise the step is blocked. When the
script runs from another tree, the step fetches `origin` and fast-forwards the checkout
to that commit, refusing local changes under those two paths only. It is done when
`09df08f` is an ancestor of `HEAD` and those paths equal the script's commit, committed
and in the working tree. Between `ca37c93` and the script's commit the legacy
embedder's modules and the Python dependencies are unchanged, so the running legacy
embedder needs no restart. Do not reinstall `services/.venv`.

To roll back the executables, follow [Roll back 52d04d4](#roll-back-52d04d4) with
`main-09df08f-*` as the loop's glob; the `f930fbc` hashes are the same.

### register

Registration copies the snapshot's runtime files, 7.4 GiB, into the registry and
freezes them read-only; the step requires 10 GiB free. Each start of the verified
service makes another private copy under `model-registry/sessions/`. Look at
`register.json`: one JSON line with `"loaded_verified": false`, and the helper's
`manifest MD5 matches`. The registry is append-only. If the MD5 differs the step
refuses, the record stays, and a corrected registration needs a new provider ID
(`HADES_MW_PROVIDER`) in every later step.

### configure

Creates `$DATA/setup-ledger` and `$DATA/run/verified` with mode 0700 and appends the
`registered_providers` and `setup` sections to the HADES YAML, after copying it to
`hades.yaml.before-configure`. If either section already exists with other values the
step refuses: edit the YAML by hand. A read-only `hades db setup inspect` must then exit
1 with `"receipt": null` and leave the ledger empty (`configure-inspect.json`). Last, it
writes `~/.config/hades/embedder-verified.env` with `HADES_EMBEDDER_IDLE_TIMEOUT=0` and
`HADES_EMBEDDER_MAX_TOKENS=20000`; the unit reads it after `embedder.env`, so its values
win. An idle unload would invalidate every receipt a client holds.

### switch-embedder

This is the verified unit's first start as a systemd unit: the 2026-11-10
qualification ran the module directly. The step installs the unit from the repository
template with its four placeholders substituted, refusing an installed unit that
differs. The template omits `PrivateTmp=`: in a user manager it implies a user namespace
in which root-owned ancestors read as uid 65534, and the server refuses such a socket
parent. Under the elected design only one embedder runs, so the legacy unit is
disabled, not only stopped, and a reboot cannot start two loads on one card. Look at:

- `status.json`: `{"model":"jinaai/jina-embeddings-v4","space":"late","loaded":true}`.
  The service loads the model before it binds the socket, about 20 seconds in the
  qualification; the retry window allows two minutes.
- `unit-first-start.journal`, the journal of this start, for the phase 2 record.
- `hades-status.json`: `verified_providers` holds `laptop-jina`, space `late`, loaded.
  The legacy `embedder` block reports its TCP endpoint unavailable, which is expected.
- `verify.json`, which the helper checks for the revision, the manifest MD5, the late
  capabilities and a 20,000-token ceiling.

The step is done when the units are switched and the marker
`switch-embedder.verified` records a passing verification. A re-run with the units
already switched asks nothing and repeats only the checks. If a check fails after the
switch, the step prints how to switch back and refuses.

While the verified unit holds the card, these legacy paths fail with a connection error:
`hades embed`, `hades codebase ingest` and code embedding, `hades ingest` into a legacy
database, and semantic search of a legacy database.
Guided reading, the reading desk, Codex chat and graph reads do not use the embedder.
The older sections of this record still say
`systemctl --user restart hades-embedder.service`; after this step that would start the
disabled legacy unit beside the verified service, which never idles out. Switch units
only this way. To switch back, which in turn stops managed ingest and search:

```bash
systemctl --user disable --now hades-embedder-verified.service
systemctl --user enable --now hades-embedder.service
curl -s --retry 60 --retry-connrefused --retry-delay 2 http://127.0.0.1:8087/v1/models
```

### setup-db

Runs `hades db setup run` with the values above, then `inspect` and `diagnose`, into
`setup-run.json`, `setup-inspect.json` and `setup-diagnose.json`. The helper requires
`"status": "ready"` and `"step": "complete"` from `run` and `inspect`, the same
`request_sha256` from both, and `"action": "none"` from `diagnose`.

On a re-run with a saved receipt that is not ready, the step reads `diagnose`. For
`retry_identical_request` it repeats the exact request, which the ledger admits under
the same operation ID. For anything else, `inspect_required` included, it refuses: keep
the ledger and the database untouched. Do not start a new operation ID for the same
database name: the first operation holds the name. A refusal before the ledger
(configuration, status, space check, verification or an absent database socket) prints
only an error on stderr and leaves the ledger untouched; fix the cause and re-run.

### Before backup-1: extend the backup driver

The combined backups are made by the private driver `.local-install/backup-workspace.py`,
the only tool that writes the `manifest.json` the verifier reads, followed by the
helper's ledger and registry copy. As of this draft the driver backs up `Conveyance_KG`
only, requires the reading-desk collections and sends its preflight to
`Conveyance_KG`. Extend it as listed under
[driver changes](operations/thinkpad-backup-restore.md#driver-changes). `check` reports
`backup driver takes --database` once the driver's source declares a `"--database"`
option, a text match rather than a run of the driver, and `backup-1` refuses until
then. Then
dry-run the driver against `Conveyance_KG` and confirm the backup still passes the
verifier. This is the operator's lane.

### backup-1

Backs up `Conveyance_KG_v2` before any ingestion, as the
[managed workspace backup](operations/thinkpad-backup-restore.md#managed-workspace)
describes, and records the backup directory as a step marker. Look at
`backup-1-managed.json` and the helper's `files verified`. The isolated verifier is not
run on this backup: it requires the reading-desk collections, which arrive with
`move-desk`. `backup-2` is verified.

### move-desk

The reading desk resolves each reading graph through ArangoDB's graph API before every
write, so the move copies the `hades_reading_*` collections, every reading graph's
`reading_<id>_nodes` and `reading_<id>_edges` collections, and the graph definitions in
`_graphs`. The new database stays idle until the restore is compared: the daemon
creates reading collections lazily, so it is pointed at the new database last. The
step runs in phases and records each as a marker, so a re-run resumes after the last
completed one:

1. The final `Conveyance_KG` backup, with the driver: the source of the move and the
   last backup before retirement. The driver restarts the services, so the step stops
   the viewer and the daemon again at once. From here until the services restart,
   nothing may write to either database; if the step fails meanwhile, it says the
   services are stopped and how to resume or undo.
2. The move list, built twice, from the quiet database (`moving.txt`) and from the
   dump's structure files (`moving-dump.txt`), which must agree. The collections that
   stay behind are printed (an empty `hades_schema` is expected). Every `_graphs` key
   must start with `reading_`, and every collection a definition names must be on the
   list. Until the restore is recorded, the moving collections must also match the
   final backup's manifest: counts first (`counts-source.json`, `counts-backup.json`),
   then a content digest per collection (`contents-source.json`,
   `contents-backup.json`). The digest is the manifest's own: SHA-256 of the documents
   sorted by `_key`, without `_rev`, as canonical JSON. Counts alone cannot see a
   record updated in place, as the desk's chat reconcile updates exchanges. A reading
   changed after the backup, or a resume days later over a changed desk, fails here;
   the step removes the final backup's markers, so the re-run takes a fresh one.
3. The final backup verified in isolation (`conveyance-kg-final-verify-result.json`);
   the verifier's staging directory under `/tmp` is removed afterwards.
4. The restore, from a staged copy of the dump, because `arangorestore` writes a
   progress file into its input directory: the reading collections with
   `--overwrite false`, so a name clash fails, then `_graphs` data-only. The step first
   requires the new database to hold no reading collections and no graph definitions.
   If an earlier run restored the collections, it accepts them only with counts and
   content digests equal to the source's: with identical graph definitions it restores
   nothing, and with an
   empty `_graphs` it restores the definitions alone. Anything else is refused, with
   the instruction to drop the reading collections and definitions from the new
   database by hand. The staged copy is removed on exit, also after a failure. The root
   password reaches `arangorestore` from `~/.config/hades/arango-root-password` through
   an environment variable, never on the command line. Counts, content digests and
   graph definitions (whole, except `_rev`) of both databases must then compare equal,
   and `diagnose` must still report
   `"action": "none"`.
5. The daemon and the viewer pointed at the new database: drop-ins named
   `managed-database.conf`, generated from the installed units with only the database
   arguments changed (the daemon keeps `--mcp-db-prefix`), and
   `HADES_DATABASE=Conveyance_KG_v2` in `daemon.env`, after a copy to the record. The
   services restart, with the daemon's readiness check on the new database.
6. The desk check: the active and archived reading lists and the source archive of
   both databases must be identical (`desk-before.json`, `desk-after.json`).

Then open `http://127.0.0.1:8088/reading`, confirm the same readings are listed, open
one reading, and make one pin. The pin proves the graph definitions resolve, because
the desk checks the definition before every write. `Conveyance_KG` keeps its copy
unchanged; nothing writes to it after this step.

To undo the move, remove both `managed-database.conf` drop-ins, restore `daemon.env`
from the record's `daemon.env.before-move-desk`, run `systemctl --user daemon-reload`,
and restart the daemon and then the viewer. Desk changes made in the new database stay
there.

### ingest

The first corpus is the Markdown and text under the checkout's `docs/` tree, named
explicitly, because a tree ingest would also route the tree's JSON evidence files and
PNG screenshots to the extractor as documents. The step refuses a tree with
uncommitted changes, records the commit (`corpus-commit.txt`) and the file list
(`corpus-files.txt`), and runs `hades ingest --root docs/` over the list; `--root` keys
each document by its path under `docs/`, so the many `README.md` files do not collide.
The helper requires `completed` to equal the file count, `failed` 0 and
`embedding_profile_state` `ready`. Then it asks two questions with known answers, five
results each:

| Question | Known answer | Expected sources |
| --- | --- | --- |
| `query-inventory.json` | `hades_setup`, `hades_workspace`, `documents`, `chunks`, `embeddings`, then `hades_ingest_jobs`; once initialized also `hades_schema`, `hades_reading_*` and the reading graph collections | `specs/recoverable-workspace-setup.md`, `specs/db-setup-door.md` |
| `query-ceiling.json` | 20,000 tokens, ruled 2026-11-09 from a real-prose bisection (23,000 passed, 24,000 ran out of memory) | `installation-thinkpad.md`, `installation-thinkpad-services.md`, `specs/db-setup-door.md` |

A question passes when one of its expected sources is among the five results; the step
prints the sources in rank order. Once the helper accepts the ingest envelope the step
marks the corpus ingested (`state/ingest.ingested`), and it is done (`state/ingest.done`)
only when both questions pass. A missed question fails the step and leaves it pending,
"ingested, query gates not passed", so `backup-2` and `all` stop there. Read the query
files first; a re-run asks both questions again without ingesting again, which the
managed path would refuse for every file already stored.

### backup-2

Requires 15 GiB available in `/tmp`, a 32 GiB tmpfs shared with other work: the
verifier stages every manifested file there, the registered artifacts (7.4 GiB)
included, and restores a database. It takes the combined backup as `backup-1` does,
verifies it in isolation (`backup-2-verify-result.json`), and removes the verifier's
staging directory. A pass proves the records restore in isolation. Keep both backups,
and the final `Conveyance_KG` backup from `move-desk`.

### Phase 2 record

After the run, phase 2 copies selected values from the record directories, never the
directories themselves:

- the run date, replacing `2026-11-xx` in this section and the files linking to it;
- `docs/evidence/db-setup-receipt-thinkpad-2026-11-xx.json`: the receipt from
  `setup-run.json`, `setup-inspect.json` and `setup-diagnose.json`;
- the registration result and the manifest MD5 comparison;
- the verified unit's first start, from `unit-first-start.journal`;
- the move: the moved collections, the count and graph comparisons, the final
  `Conveyance_KG` verifier result, `diagnose` after the move, and the desk check;
- the first ingest: the corpus commit, the counts, and each question's sources in rank
  order;
- the two backup directories' names and the `backup-2` verifier result (counts only);
- the retirement decision below, and corrections to anything the run showed wrong.

### Retirement gate

A pass in `backup-2` does not prove that the managed workspace can be brought back from
that backup. Per the model-bound recovery contract's
[restore boundary](operations/model-bound-recovery.md#restore-and-migration-boundary),
a restored database has a different database incarnation even at the same name and
endpoint, the copied setup receipt cannot authorize it, and ownership rebinding is not
supported. A restore of `Conveyance_KG_v2` today therefore recovers its records, not a
ready managed workspace.

`Conveyance_KG` is retired only after the operator accepts that limitation explicitly,
recorded here with the date, or after a supported rebinding procedure exists, which is
outside this work. The drop is by hand, outside HADES, because HADES has no
drop-database command (#118). The installer never drops a database; it prints this gate
after `backup-2` and stops.

Operator decision: pending.

## Current operating state

All five user services are enabled and active. Guided reading is available at
`http://127.0.0.1:8088/reading`; the embedder serves
`http://127.0.0.1:8087/v1`, and MCP uses `http://127.0.0.1:10444/mcp`.
The model unloads after 900 idle seconds and reloads on demand. Check
`systemctl --user status hades-embedder.service` and `/v1/models` when diagnosing
embedding availability. Codex chat is enabled in the reading desk; open a saved
reading and select **Chat** to create or explicitly resume a conversation.
No Olympus data has been restored.
