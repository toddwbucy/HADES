# deploy/systemd — running HADES as a service

Templates for the daemon, the embedder, and an ArangoDB instance. Copy the
`.example` files, edit, install. **Nothing deployment-specific is stored in
this repository**: addresses, paths and credentials all come from the
environment at execution time.

| template | install to | configures |
|---|---|---|
| `hades-daemon.service` | `~/.config/systemd/user/` | daemon and optional MCP endpoint |
| `daemon.env.example` | `~/.config/hades/daemon.env` | ArangoDB connection, MCP exposure |
| `hades-embedder@.service` | `~/.config/systemd/user/` | embedder, one instance per load profile |
| `embedder-profile-gpu*.conf.example` | `~/.config/hades/embedder-profiles/<name>.conf` | one load profile per GPU: card, ceiling, batch, VRAM floor |
| `hades-embedder-profile` | anywhere on `PATH` | switch the embedder between profiles |
| `arangod.conf.example` | your ArangoDB instance directory | storage, endpoints, vector index |

```bash
systemctl --user daemon-reload
systemctl --user enable --now hades-daemon
```

## Load profiles: the ceiling is a property of the card

The embedder is one templated unit instantiated per load profile, named for the
card it loads on. The profile fixes the device, the sequence ceiling, the batch
size and the VRAM floor, and every ceiling in the examples is a measurement on
that card rather than a preference.

```bash
hades-embedder-profile list     # profiles and their ceilings
hades-embedder-profile gpu1     # switch and persist; waits for matching metadata
hades-embedder-profile          # what is active, as the service reports it
```

**Every profile binds the same port.** That is the point: switching cards is a
stop and a start, and no client configuration changes, because clients read the
ceiling back from `GET /v1/models` (`max_seq_length`, plus `profile` and
`physical_device`). It also means two profiles cannot be live at once, since the
shared endpoint cannot host two listeners. This is not a GPU allocation guard:
use the selector to stop the previous instance before starting another.

## Persistent selection and maintenance procedure

The single persistently enabled `hades-embedder@NAME.service` is the saved
selection. The selector removes other persistent and runtime enablement links,
including failed instances, and enables the selected profile for the next user
manager boot. Selecting the already-active profile reconciles links without
restarting it. A lock prevents overlapping selector runs; do not run competing
manual `systemctl` commands during a switch. Multiple active instances cause a
refusal requiring operator review.

Switching still runs the service's normal device/VRAM preflight. The selector
requires an active unit and matching `/v1/models` profile metadata with positive
dimension and context limits. Metadata is **not an inference test**. Its wait is
120 seconds by default (`HADES_PROFILE_READY_TIMEOUT=SECONDS`, 1–999); each HTTP
attempt can add up to three seconds. Review unit journals if startup fails.

A failed switch or caught interruption attempts to restore previous enablement
and the previous active unit. Failure returns nonzero even when restoration
succeeds. `ROLLBACK INCOMPLETE` requires manual recovery; prior inference health
is not implied by successful `systemctl start`. If the prior boot configuration
already enabled multiple profiles, rollback restores that known-bad snapshot:
review and rerun the selector before rebooting. SIGKILL or power loss can interrupt
any transition; inspect active units and enablement before recovery.

For a running server, schedule and carry out these steps explicitly:

1. Record current active/enabled instances using `hades-embedder-profile show`
   and save the installed selector, template, and profile files privately. Confirm
   the chosen card's resource budget and keep the previous profile available.
2. Install the reviewed selector. Installing it does not change any service.
   Pause inference-dependent work for an actual profile change; selecting the
   already-active profile only reconciles boot enablement.
3. Run `hades-embedder-profile NAME`. Verify exactly one active instance and one
   persistently enabled instance. Inspect its journal for preflight failures and
   perform an agreed small inference request before resuming clients.
4. During a separate maintenance reboot, confirm the same single profile starts;
   check both metadata and inference again. Mocked reboot tests establish the
   selection logic, not this machine's full boot environment.
5. If verification fails, select the previous profile with the reviewed selector
   and repeat health checks. If rollback is incomplete, explicitly stop the failed
   target before starting the prior unit; restore exactly one boot selection.
   Preserve logs and revert the installed script/template if needed.

This procedure has not been executed on the audited live server. Its observed
active GPU2/failed-but-enabled GPU1 state remains unchanged by repository work.

Run the isolated selector regressions (fake systemctl/curl, no GPU/network):

```bash
python3 -m unittest discover -s scripts/tests -p test_embedder_profile.py -v
```

**Measure the ceiling on your own hardware, with your own documents.** The
example numbers are from olympus and two of them were wrong before they were
measured properly. A synthetic probe put the 16 GiB card at 15,000 tokens, and
real documents through the running service put it between 11,926 (passes
repeatedly) and 12,196 (out of memory). A synthetic probe put the 48 GiB card's
32,768-token peak at 23,906 MiB, and a real 31,871-token document peaked at
27,526 MiB. Real inputs cost more than probes, in both directions that matter:
the small card admits less and the large card needs a higher floor.

An input above the ceiling is refused with `PE_INPUT_TOO_LARGE`, naming its token
count, rather than truncated. Pre-chunking oversized inputs is the caller's job.

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
