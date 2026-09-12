# deploy/systemd — running HADES as a service

Templates for the daemon, the embedder, and an ArangoDB instance. Copy the
`.example` files, edit, install. **Nothing deployment-specific is stored in
this repository**: addresses, paths and credentials all come from the
environment at execution time.

| template | install to | configures |
|---|---|---|
| `hades-daemon.service` | `~/.config/systemd/user/` | daemon and optional MCP endpoint |
| `daemon.env.example` | `~/.config/hades/daemon.env` | ArangoDB connection, MCP exposure |
| `embedder.conf.example` | `~/.config/hades/embedder.conf` | embedder device, model path, token ceiling |
| `arangod.conf.example` | your ArangoDB instance directory | storage, endpoints, vector index |

```bash
systemctl --user daemon-reload
systemctl --user enable --now hades-daemon
```

`loginctl enable-linger $USER` if the daemon should survive logout. Otherwise
systemd stops user services when your last session closes, and any remote
client loses the endpoint.

## The MCP endpoint is off by default

`HADES_MCP_BIND` is unset in the template. With no bind address the daemon
serves its Unix socket and opens **no network listener at all**. Leave it
there unless you need remote access.

If you do enable it, two things are true and neither is obvious.

**The endpoint is plain HTTP, not TLS**, whatever port you choose. The bearer
token crosses the network in the clear. On anything but a trusted network,
keep the bind on loopback and reach it through an SSH tunnel:

```bash
ssh -N -L 10443:127.0.0.1:10443 <host>
```

**Binding is not reaching.** If the host runs a firewall, a bind to a
non-loopback address still needs that port opened, and testing from the server
will not tell you — loopback traffic bypasses the filter, so the endpoint
answers locally while every other machine times out. Open it for your local
subnet only, never globally, with whatever your distribution provides (`ufw`,
`firewall-cmd`, `nft`). Then verify from the machine that will actually
connect, not from the server.

## Configuration traps worth knowing before you hit them

**`EnvironmentFile` wants bare `KEY=value`.** systemd silently discards any
line carrying an `export` prefix, and `systemctl show -p Environment` will not
reveal the problem, because it reports only inline `Environment=` directives
and never expands `EnvironmentFile=`. Check `/proc/<pid>/environ` instead. Keep
a separate shell-sourced file with `export` lines if you want one for
interactive use.

**`experimental-vector-index = true`** must be set in `arangod.conf` or the
faiss-backed vector index is unavailable. That capability is much of why
ArangoDB is the store, so it is not optional in practice.

**The embedder's token ceiling is hardware, not preference.** See
`services/embedding/jina_v4.py` for the measurements. jina v4 advertises
32768, which is an architectural limit rather than what fits on a given card,
and batch size multiplies activation pressure. An OOM during ingest usually
means lower the batch size rather than doubt the ceiling.
`HADES_EMBEDDER_MIN_FREE_VRAM_MIB` sets a flat floor when you want a busy GPU
refused up front rather than discovered mid-request.

**Restrict GPU visibility at the driver level** rather than selecting a device
in the application, so other cards are invisible instead of merely unselected:

```
CUDA_VISIBLE_DEVICES=2
HADES_EMBEDDER_DEVICE=cuda:0    # relative to the above
```

## Not covered here

Building ArangoDB. If your distribution packages a recent enough version, use
it. Building from source against a very new toolchain needs local patches —
`docs/mcp-deployment.md` records the obstacles encountered and what each one
turned out to be.
