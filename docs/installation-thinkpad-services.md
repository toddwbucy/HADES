# ThinkPad service configuration snapshot

Recorded 2026-10-18 after CPU/database/browser/GPU verification. See
[the installation record](installation-thinkpad.md) for prerequisites, source
patches and validation results. These are the actual local
paths; adapt them for another machine. Store configuration under a mode-0700
directory, with mode-0600 configuration/environment/secret files and mode-0644
user unit files. Generate new secrets for every installation.

The data path `~/.local/share/hades` resolves to
`/home/todd/Projects/HADES-local`. Create its `run/`, `arangodb/data/`, and
`arangodb/apps/` directories before starting services. The source checkout and
Python environment must remain at the paths used by these development units.
The browser readiness helper also runs from this checkout.

## HADES YAML overrides

Copy `config/hades.yaml` to `~/.config/hades/hades.yaml` and update these values;
retain other repository defaults. Use explicit `--db Conveyance_KG` commands.

```yaml
database:
  host: 127.0.0.1
  sockets:
    readonly: /home/todd/.local/share/hades/run/arango.sock
    readwrite: /home/todd/.local/share/hades/run/arango.sock
extraction:
  service:
    socket: /home/todd/.local/share/hades/run/extractor.sock
embedding:
  service:
    socket: http://127.0.0.1:8087/v1
    timeout_ms: 120000
  model:
    max_tokens: 4096
  batch:
    size: 1
    size_small: 1
gpu:
  device: cuda:0
```

The embedding configuration passed GPU inference, late chunking, and explicit
oversized-input rejection at this ceiling. This is not the GPU's maximum window.
This YAML is an override excerpt, not a complete replacement configuration.

## `~/.config/hades/arangod.conf`

```ini
vector-index = true

[server]
endpoint = unix:///home/todd/.local/share/hades/run/arango.sock
authentication = true
authentication-unix-sockets = true
jwt-secret-keyfile = /home/todd/.config/hades/arango-jwt-secret
minimal-threads = 4
maximal-threads = 8
io-threads = 2
statistics = false

[database]
directory = /home/todd/.local/share/hades/arangodb/data

[javascript]
startup-directory = /home/todd/.local/opt/arangodb-3.12.12.1/share/arangodb3/js
app-path = /home/todd/.local/share/hades/arangodb/apps

[foxx]
queues = false

[log]
output = -
level = info

[rocksdb]
block-cache-size = 536870912
total-write-buffer-size = 268435456
write-buffer-size = 67108864
max-background-jobs = 4

[query]
memory-limit = 268435456
global-memory-limit = 1073741824

[arangosearch]
commit-threads = 2
consolidation-threads = 2
execution-threads-limit = 2
```

## `~/.config/hades/daemon.env`

```ini
HADES_CONFIG=/home/todd/.config/hades/hades.yaml
HADES_DATABASE=Conveyance_KG
ARANGO_USERNAME=hades
ARANGO_PASSWORD=<GENERATE_A_NEW_LOCAL_PASSWORD>
ARANGO_RO_SOCKET=/home/todd/.local/share/hades/run/arango.sock
ARANGO_RW_SOCKET=/home/todd/.local/share/hades/run/arango.sock
HADES_EXTRACTOR_SOCKET=/home/todd/.local/share/hades/run/extractor.sock
HADES_EMBEDDER_SOCKET=http://127.0.0.1:8087/v1
```

## `~/.config/hades/extractor.env`

```ini
HADES_EXTRACTOR_SOCKET=/home/todd/.local/share/hades/run/extractor.sock
HADES_EXTRACTOR_DEVICE=cpu
HADES_EXTRACTOR_OCR=false
HF_HOME=/home/todd/.cache/hades/huggingface
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1
```

## `~/.config/hades/embedder.env`

```ini
CUDA_VISIBLE_DEVICES=0
HADES_EMBEDDER_DEVICE=cuda:0
HADES_EMBEDDER_HOST=127.0.0.1
HADES_EMBEDDER_PORT=8087
HADES_EMBEDDER_FP16=true
HADES_EMBEDDER_BATCH_SIZE=1
HADES_EMBEDDER_MAX_TOKENS=4096
HADES_EMBEDDER_PROFILE=thinkpad-initial
HADES_EMBEDDER_IDLE_TIMEOUT=900
HADES_EMBEDDER_MODEL=/home/todd/.local/share/hades/models/jinaai--jina-embeddings-v4
HF_HOME=/home/todd/.cache/hades/huggingface
OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=1
```

## Secret files

Generate independent values for `arango-root-password`, `arango-jwt-secret`,
`mcp-token`, and `viewer-password` under `~/.config/hades/`. The installation
used `secrets.token_urlsafe(32)` for passwords/tokens and
`secrets.token_hex(32)` for the ArangoDB JWT key. Also generate the separate
`ARANGO_PASSWORD` in `daemon.env`; use that same value when creating the
`hades` ArangoDB account. None of these values are included in this snapshot.

## `~/.config/systemd/user/hades-arangodb.service`

```ini
[Unit]
Description=HADES local ArangoDB 3.12.12.1
After=network.target

[Service]
Type=simple
ExecStart=/home/todd/.local/opt/arangodb-3.12.12.1/sbin/arangod --configuration /home/todd/.config/hades/arangod.conf
Environment=LC_ALL=C.UTF-8
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4
Restart=on-failure
RestartSec=10
UMask=0077
LimitNOFILE=65536
TimeoutStopSec=120

[Install]
WantedBy=default.target
```

## `~/.config/systemd/user/hades-extractor.service`

```ini
[Unit]
Description=HADES local CPU extraction
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/todd/Projects/HADES/services
EnvironmentFile=/home/todd/.config/hades/extractor.env
ExecStart=/home/todd/Projects/HADES/services/.venv/bin/python -m extraction.server
Restart=on-failure
RestartSec=10
UMask=0077
LimitNOFILE=65536

[Install]
WantedBy=default.target
```

## `~/.config/systemd/user/hades-embedder.service`

```ini
[Unit]
Description=HADES local Jina v4 embedding
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/todd/Projects/HADES/services
EnvironmentFile=/home/todd/.config/hades/embedder.env
ExecStart=/home/todd/Projects/HADES/services/.venv/bin/python -m embedding.http_server
Restart=on-failure
RestartSec=10
UMask=0077
LimitNOFILE=65536

[Install]
WantedBy=default.target
```

## `~/.config/systemd/user/hades-daemon.service`

```ini
[Unit]
Description=HADES local research daemon
After=network.target hades-arangodb.service

[Service]
Type=simple
WorkingDirectory=/home/todd/Projects/HADES/services
EnvironmentFile=/home/todd/.config/hades/daemon.env
ExecStart=/home/todd/.local/bin/hades --db Conveyance_KG daemon --socket /home/todd/.local/share/hades/run/hades.sock --mcp-bind 127.0.0.1:10444 --mcp-token-file /home/todd/.config/hades/mcp-token --mcp-dbs Conveyance_KG --mcp-db-prefix Conveyance_KG --mcp-guided-reading
Restart=on-failure
RestartSec=10
UMask=0077
LimitNOFILE=65536

[Install]
WantedBy=default.target
```

## `~/.config/systemd/user/hades-viewer.service`

```ini
[Unit]
Description=HADES local research browser
After=network.target hades-daemon.service

[Service]
Type=simple
WorkingDirectory=/home/todd/Projects/HADES/services
EnvironmentFile=/home/todd/.config/hades/daemon.env
ExecStartPre=/usr/bin/python3 /home/todd/Projects/HADES/scripts/check_daemon_ready.py --socket /home/todd/.local/share/hades/run/hades.sock --database Conveyance_KG --timeout 60
ExecStart=/home/todd/.local/bin/hades-viewer serve --db Conveyance_KG --reading-db Conveyance_KG --reading-socket /home/todd/.local/share/hades/run/hades.sock --hades-bin /home/todd/.local/bin/hades --bind 127.0.0.1:8088 --password-file /home/todd/.config/hades/viewer-password
Restart=on-failure
RestartSec=10
UMask=0077
LimitNOFILE=65536

[Install]
WantedBy=default.target
```

## Activation

Initialize the empty ArangoDB storage and create the database account/schema
as described in the installation record before activating the application.
Then run `systemctl --user daemon-reload` and enable/start the ArangoDB,
extractor, daemon, and viewer units. Leave the embedder disabled until the GPU
probe passes, then enable/start it too. The ThinkPad now has all five units
enabled and active. Create the model alias shown in the installation record
before starting the embedder. These user services start with the owner's login session; no
system-wide service or user lingering was configured.
