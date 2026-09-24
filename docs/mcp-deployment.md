# MCP Endpoint Deployment

How the MCP endpoint is actually run, and what it depends on. The wire
protocol and access tiers are specified in `daemon-protocol.md`; this
document is operational and records measured values from the olympus
deployment on 2026-09-12.

## What the endpoint is

`hades daemon --mcp-bind` serves a **curated 12-tool agent surface** over
streamable HTTP, so a remote agent session consumes HADES without a local
binary. The CLI has roughly 93 subcommands. The MCP surface exposes twelve,
and that ratio is the point: every tool definition sits in the client's
context for the whole session, so the surface has to earn its size.

| tool | what it does |
|---|---|
| `orient` | collections, counts, shape of a database |
| `db_query` | semantic search — **embeds the query, needs the embedder** |
| `db_get` / `db_list` / `db_count` | document access |
| `graph_neighbors` / `graph_traverse` | graph walks, bounded depth |
| `smell_report` | recorded code smells for a file |
| `task_list` / `task_show` / `task_create` / `task_update` | kanban board |

Every call is serialized to the daemon frame format and routed through the
same parse/authorize/dispatch path the Unix socket uses, under
`ConnectionPolicy::agent_only`. No tool here can reach an Admin-tier command
even if one were mounted by mistake.

## Running it

```bash
hades daemon \
  --database WeaverTools_v3 \
  --socket   ~/.local/share/hades/run/hades.sock \
  --mcp-bind <lan-ip>:10443 \
  --mcp-token-file ~/.config/hades/mcp-token \
  --mcp-dbs WeaverTools_v3,hades_memories
```

- `--mcp-bind` accepts loopback or RFC1918 only, and startup fails closed
  when the token file is missing or empty.
- `--mcp-dbs` scopes what the endpoint serves. Anything not listed is
  refused. Writes stay ACL-gated on the ArangoDB user regardless.
- The endpoint is **plain HTTP, not TLS**, whatever port it runs on. The
  bearer token crosses the network in the clear, which is acceptable on a
  trusted LAN and is the reason to prefer an SSH tunnel otherwise:
  `ssh -N -L 10443:127.0.0.1:10443 <host>`.

Generate the token with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`,
mode 0600.

**Firewall.** Binding is not reaching. A LAN bind still needs the port
opened, and testing from the server itself will not catch this because
loopback traffic bypasses the filter:

```bash
sudo ufw allow from <your-subnet> to any port 10443 proto tcp comment 'hades mcp, lan'
```

Verify from the machine that will actually connect, never from the host.

**Client registration:**

```bash
claude mcp add --transport http hades http://<lan-ip>:10443/mcp \
  --header "Authorization: Bearer <token>"
```

## Provisioning: letting a client build its own graph

Off by default. With it on, an MCP client can create a database and ingest a
tree into it, which is how another session stands up its own graph without
shell access to the host.

```
hades daemon \
  --mcp-bind 192.168.0.10:10443 \
  --mcp-token-file ~/.config/hades/mcp-token \
  --mcp-db-prefix bident_ \
  --mcp-ingest-root /opt/HADES \
  --mcp-ingest-root /opt/weavertools
```

Three tools appear: `create_database`, `ingest_start`, `ingest_status`. They are
advertised on every endpoint and authorized on none by default, so a client
without provisioning gets `ACCESS_DENIED` naming what would have been permitted
rather than concluding the capability does not exist.

**Both flags are load-bearing, and both fail closed.**

`--mcp-db-prefix` bounds what may be created. Without it, no database may be,
even with the other flag set. It also widens the read allowlist for matching
names, because a database the endpoint just created is not on `--mcp-dbs` and
would otherwise be refused the moment the client tried to use it.

`--mcp-ingest-root` bounds what may be read. Ingest hands the daemon a path on
its *own* filesystem, so without this a bearer token could have it read
`~/.ssh`, `/etc`, or the token file itself, embed the contents, and query them
back out through `db_query`. That is exfiltration wearing a retrieval interface.
Paths are canonicalized before the check, so `..` and symlinks cannot escape a
listed root, and a path that does not exist is refused rather than guessed at.

**What this does not change.** The tier ceiling stays at Agent, so raw AQL,
`db.purge`, `db.insert` and `db.graph.drop` remain unavailable to the endpoint.
Provisioning grants two commands, not a promotion.

**What it assumes.** That ArangoDB is enforcing its own access control. If the
instance runs with `authentication = false`, the daemon can reach every database
on it and `--mcp-dbs` plus these prefixes are the only scope that exists. Stand
up the dedicated `hades` user with per-database grants before turning
provisioning on for anything you would mind a LAN client reaching.

## The embedder dependency

`db_query` embeds the query before searching, so **semantic search fails
without the embedder** while the other eleven tools keep working. The daemon
starts fine without it and reports the failure per call, so a broken embedder
looks like "search is broken" rather than "the endpoint is down".

Check it with `hades embed service status`. The CLI expects
`http://localhost:8087/v1`.

Configuration lives in `services/systemd/embedder.conf`, installed to
`/etc/hades/embedder.conf` and sourced as an `EnvironmentFile`. Bare
`KEY=value` lines: systemd silently discards any line with an `export`
prefix, and `systemctl show -p Environment` will not reveal the problem
because it reports only inline `Environment=` directives and never expands
`EnvironmentFile=`. Check `/proc/<pid>/environ` instead.

### Sequence length is a hardware property, and the numbers are lower than they look

Bisected on olympus GPU 2 (RTX 2000 Ada, 16 GiB) on 2026-09-12, fresh
process, `expandable_segments`, batch of one:

| tokens | result |
|---|---|
| 30000 | OOM, peak 15,856 MiB of 16,380 |
| 24000 | OOM, peak 15,796 MiB |
| 16384 | OOM |
| 16000 | OOM |
| **15000** | **OK, last size that completes** |
| 12000 | OK |

The model is 9,216 MiB resident, leaving roughly 6 GiB for activations, and
one forward pass over ~16k tokens does not fit in that.

**Two traps in those numbers.**

Jina v4 advertises 32768 and `MAX_TOKENS` was hardcoded to it. That is the
architectural limit, not what fits on a given card.

And Yeomna's embedder advertised **16384 on this same card**, which reads
like evidence that 16k works. It is not. That was a *refusal threshold* — the
service rejected longer inputs with "N tokens exceeds max_tokens 16384" and
chunked what it accepted, so it never ran 16k through the model in one pass
either.

`HADES_EMBEDDER_MAX_TOKENS` defaults to **14000**, below the bisected 15000,
because the shipped `HADES_EMBEDDER_BATCH_SIZE=4` has four sequences sharing
the activation headroom a single-sequence probe measured. **16000 is not a
safe round number, it is the first size that fails.**

### Preflight: the service refuses a busy card rather than OOMing on it

Without a check, a card that already has another job on it produces a service
that starts, reports ready, accepts a request, and only then dies inside the
forward pass with a CUDA OOM. That reads like a bad input rather than a busy
GPU, which is the expensive kind of wrong.

`JinaV4Embedder._preflight_device()` runs before the weights load and refuses
with a message naming the processes holding the device:

```
refusing to load on cuda:0: 9778 MiB free of 48538 MiB, need 11264 MiB
(model 9216 MiB plus 2048 MiB working headroom). Currently on this device:
2986361, 38494 MiB, python. Free the device, lower HADES_EMBEDDER_MAX_TOKENS,
or point CUDA_VISIBLE_DEVICES at a card with room.
```

**The default check detects contention, it does not predict OOM**, and the
distinction matters. A full activation estimate — 0.45 MiB per token, derived
from the 32,768-token run peaking at 23,906 MiB — comes to 15,966 MiB at
`MAX_TOKENS=15000`, which exceeds the 15,489 MiB free on an *idle* 16 GiB
card. A check built that way refuses configurations that demonstrably work.
Peak depends on allocator behaviour that varies run to run, so it is not
predictable from a resident-size constant. The default therefore asks only
whether the card can hold the model with room to work, which is exactly what
separates a free card from an occupied one. The true edge is enforced by
`MAX_TOKENS` being set from measurement.

When a blunt guarantee is wanted instead, set the flat floor:

```
HADES_EMBEDDER_MIN_FREE_VRAM_MIB=24576   # "24 GiB or refuse"
```

which overrides the computed check entirely. That is the right setting for
the large-document instance, since the 32,768-token run peaked at 23,906 MiB
and no drift in an estimate can quietly erode an explicit floor.

Verified 2026-09-12: idle 16 GiB card passes (15,489 free against 11,264
needed), A6000 carrying a 38 GiB job refuses (9,778 free against 11,264).

### Oversized documents: move them to an A6000

For a document that needs more, move it rather than raising the 16 GiB card's
ceiling. **Measured on olympus GPU 1 (RTX A6000, 49 GiB) on 2026-09-12: a
full 32,768-token input embeds successfully, peaking at 23,906 MiB.** That is
under half the card, so the limit there is jina v4's architectural 32768
rather than the hardware.

Run it as a second instance on its own port, leaving the GPU 2 service in
place for ordinary traffic:

```bash
CUDA_VISIBLE_DEVICES=1 \
HADES_EMBEDDER_DEVICE=cuda:0 \
HADES_EMBEDDER_PORT=8088 \
HADES_EMBEDDER_MAX_TOKENS=32768 \
HADES_EMBEDDER_MODEL=<model-path> \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m embedding.http_server
```

The two instances share the model directory read-only and do not interfere.
Point the oversized job at `http://localhost:8088/v1` and leave everything
else on 8087.

This is what makes the WeaverTools Specs reachable: at 25k-47k estimated
tokens they exceed the 16 GiB card at any setting, and the five largest
exceed 32768 even here, so those still need segmentation. The six between
16k and 32k embed whole on an A6000 and nowhere else.

### Model location

`HADES_EMBEDDER_MODEL=<model-path>`

Naming the path directly makes `HF_HOME` irrelevant to this service. The
duplicate under `~/.cache/huggingface` was removed 2026-09-12 after sha256
verification that both weight shards and the adapter matched the pinned
`853c867b` snapshot. `core/config/embedders/jina_v4.yaml` in the Python HADES
was repointed at the same path at that time, because it referenced the model
by HF name and would otherwise have silently re-downloaded 7.4 GB against a
cold cache.

Restricting GPU visibility at the driver level is preferred over selecting a
device in the application, since it makes the other cards invisible rather
than merely unselected:

```
CUDA_VISIBLE_DEVICES=2
HADES_EMBEDDER_DEVICE=cuda:0     # relative to the above
```

## Olympus deployment

| | |
|---|---|
| binary | `~/.local/bin/hades` (cargo target is on `fastpool` and dies on `cargo clean`) |
| daemon unit | `~/.config/systemd/user/hades-daemon.service` |
| daemon env | `~/.config/hades/daemon.env` |
| embedder config | `~/.config/hades/embedder.conf` |
| MCP token | `~/.config/hades/mcp-token`, 0600 |
| ArangoDB | built from source at `~/git/arangodb`, instance at `~/.local/share/arangodb3` |

`loginctl enable-linger todd` if the daemon should survive logout, otherwise
systemd stops user services when the last session closes and remote clients
lose the endpoint.

### Explicit degraded enrichment (#179)

`hades ingest <directory> --allow-degraded-enrichment` accepts semantic requests
that still fail after retry. Daemon `ingest.start` and MCP `ingest_start` expose
the optional boolean `allow_degraded_enrichment` (default `false`).
`ingest.start` remains **Provisioning**; `ingest.status` remains **Agent**.

Without the override these requests make the terminal ingest envelope
`success: false`. With it, an otherwise successful run reports `success: true`,
`data.enrichment_degraded: true`, and the full `data.failed_requests` list
(file, symbol, request and reason), also retained in the per-analyzer reports.
Clean runs report `enrichment_degraded: false` and an empty list.
The persisted job records the chosen override and captures this same envelope
in `result`; `ingest_status` / `ingest.status` return that record.

Like codebase ingest's `--allow-analysis-downgrade`, this explicit acceptance
retains prior affected semantic edges and refreshes eligible content. The new
option accepts request failures only: missing analyzers, storage failures and
other code/document failures still fail the run. It does not authorize a tier
downgrade. Existing bounded job capture still fails visibly on output overflow;
no successful degraded job silently truncates its failed-request list.
