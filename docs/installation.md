# System installation (Linux)

For the project overview and user-level build, see the [README](../README.md).
This guide covers the system daemon, database account, and optional Python services.
The browser viewer and guided-reading setup are documented separately in
[Knowledge workspace](frontend-workspace.md) and [Guided document reading](guided-reading.md).

The repository includes an Ubuntu 24.04 container harness in
`scripts/install/test/`. It starts the daemon manually, so passing that harness
is not evidence that systemd, GPU services, or a complete guided-reading install
works on a fresh host. Review service users, supplementary groups, filesystem
paths, and model configuration for the destination machine before enabling units.
Run commands below from the repository root unless a step says otherwise.

## 0. Prerequisites

These system-install steps target Ubuntu 24.04 and require root. Other
distributions need their own package and service setup.

**Install ArangoDB** following the [official docs](https://docs.arangodb.com/3.12/operations/installation/linux/).
The short version for Debian/Ubuntu:

```bash
# Stop if fetching, converting, or authenticating the repository fails.
set -e
key_file=$(mktemp)
curl -fsSL https://download.arangodb.com/arangodb312/DEBIAN/Release.key -o "$key_file"
sudo install -d -m 0755 /etc/apt/keyrings
sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/arangodb.gpg "$key_file"
rm -f "$key_file"
sudo chmod 0644 /etc/apt/keyrings/arangodb.gpg
echo 'deb [signed-by=/etc/apt/keyrings/arangodb.gpg] https://download.arangodb.com/arangodb312/DEBIAN/ /' \
  | sudo tee /etc/apt/sources.list.d/arangodb.list
sudo apt-get -o APT::Update::Error-Mode=any update && sudo apt-get install -y arangodb3
```

If APT reports an expired, missing, or invalid signing key, stop and consult
ArangoDB's current download instructions for the supported repository and key.
Do not disable signature verification to continue an installation. The keyring
must be readable by APT and explicitly selected by the source's `signed-by` option.

During `apt-get install` you'll be prompted to set a root password
for ArangoDB — note it, you'll use it in step 2.

HADES also needs a reachable Unix socket. Configure ArangoDB to expose the
socket named in `database.sockets` before continuing; installing the package
alone does not establish that endpoint. The
[ArangoDB configuration example](../deploy/systemd/arangod.conf.example) shows
the endpoint syntax. Adapt its paths and permissions rather than copying another
machine's storage layout.

**Install rustup.** The checkout selects its compiler through
[`rust-toolchain.toml`](../rust-toolchain.toml):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. ~/.cargo/env
```

**Install `protoc`** (needed by `hades-proto`'s build script):

```bash
sudo apt-get install -y protobuf-compiler
```

## 1. Build the binary

From a checkout of this repo:

```bash
cargo build --release --locked
```

This produces `hades` and `hades-viewer` in the release target directory
(`target/release` unless Cargo configuration overrides it).

## 2. Create the ArangoDB user

HADES connects to ArangoDB as a dedicated `hades` user (not `root`).
Write restrictions on specific databases are enforced by ArangoDB
ACLs on this user. MCP and the viewer add their own database scopes, which do
not replace those ACLs. Bootstrap
with arangosh, using the root password you set during install:

```bash
arangosh --server.endpoint unix:///run/arangodb3/arangodb.sock \
         --server.username root
# (enter root password at the prompt, then in the arangosh REPL:)
> const users = require("@arangodb/users");
> users.save("hades", "<pick-a-password>");
> users.grantDatabase("hades", "_system", "rw");
> users.grantDatabase("hades", "*", "rw");   // default for new DBs
> // Tighten specific production databases as needed:
> // users.grantDatabase("hades", "<production-db-name>", "ro");
```

Note the password you set for `hades` — you'll use it in step 5.

## 3. Install the systemd users + tmpfiles

These create the Unix `hades` user and group, plus the `/run/hades`
runtime directory. They must run **before** step 4 because the config
file's group ownership references `hades`.

```bash
sudo install -m 644 services/systemd/hades-sysusers.conf  /etc/sysusers.d/hades.conf
sudo install -m 644 services/systemd/hades-tmpfiles.conf  /etc/tmpfiles.d/hades.conf
sudo systemd-sysusers
sudo systemd-tmpfiles --create
```

After this, `getent group hades` should show the group exists.

## 4. Install the binary

```bash
sudo install -m 755 target/release/hades /usr/local/bin/hades
sudo install -m 755 target/release/hades-viewer /usr/local/bin/hades-viewer
mkdir -p ~/.local/bin && install -m 755 target/release/hades ~/.local/bin/hades
```

The systemd daemon execs `/usr/local/bin/hades`; interactive shells
usually pick up `~/.local/bin/hades` first. If you only update one,
the other will silently run an older build.

## 5. Install the system config

```bash
sudo mkdir -p /etc/hades
sudo install -m 640 -o root -g hades config/hades.yaml /etc/hades/hades.yaml
```

Set `HADES_CONFIG=/etc/hades/hades.yaml` in the daemon environment explicitly;
the username (`hades`) and socket paths live there. Passwords are
never stored in YAML — they come from the daemon's env (step 6). The
username can also be overridden out-of-band with the `ARANGO_USERNAME`
env var, parallel to `ARANGO_PASSWORD` / `ARANGO_HOST` / `ARANGO_PORT` /
`ARANGO_RO_SOCKET` / `ARANGO_RW_SOCKET`.

## 6. Set the daemon environment

`/etc/hades/daemon.conf` is sourced by `hades-daemon.service` via
`EnvironmentFile=`. Create it with:

```bash
sudo tee /etc/hades/daemon.conf <<'CONF'
HADES_CONFIG=/etc/hades/hades.yaml
ARANGO_PASSWORD=<password-you-set-for-the-hades-user>
HADES_DATABASE=_system
ARANGO_RO_SOCKET=/run/arangodb3/arangodb.sock
ARANGO_RW_SOCKET=/run/arangodb3/arangodb.sock
HADES_EMBEDDER_SOCKET=http://localhost:8087/v1
CONF
sudo chown root:hades /etc/hades/daemon.conf
sudo chmod 640 /etc/hades/daemon.conf
```

The daemon needs an explicit database at startup. `_system` is used here for
administration, not as a research destination. The Unix daemon serves that one
database. For guided reading, run a daemon against the chosen reading database
and configure the viewer with the matching database and socket. The optional MCP
transport has separate allowlist and provisioning rules.

## 7. Enable and start the daemon

Review the unit before installation. Its `SupplementaryGroups` includes
`weaver-admin` and `arangodb`; remove unused groups or supply the corresponding
local groups and socket permissions. A missing group prevents systemd startup.

```bash
sudo install -m 644 services/systemd/hades-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hades-daemon.service
sudo systemctl status hades-daemon.service --no-pager
```

You should see `Active: active (running)` and a log line like
`daemon listening socket="/run/hades/hades.sock"`.

## 8. Verify

`hades db stats` opens its own ArangoDB connection (it doesn't proxy
through the daemon socket), so the CLI needs the `hades` user's
password in its environment:

```bash
export ARANGO_PASSWORD='<password-you-set-for-the-hades-user>'
hades --db _system db stats
```

A clean JSON response listing the `_system` database means
authentication works and your CLI can reach ArangoDB. The
`ARANGO_PASSWORD` export takes precedence over anything in
`hades.yaml`; keep it in your shell environment (or in a profile
file restricted to your user) for ongoing CLI use.

The `hades-embedder.service` and `hades-extractor.service` units cover
the optional embedder (Jina V4) and Docling extractor services. Document ingestion
and guided-reading uploads require extraction; embedding and semantic search
require the embedder. Both need additional Python dependencies and model assets;
see [Python services](../services/README.md) for installation details.
The supplied units are templates. Install model assets separately and measure
context and batch limits on the target hardware. A running process alone does not
verify extraction or embedding.

## Updating the daemon's ArangoDB password

1. Rotate the password in arangosh:

```javascript
require("@arangodb/users").replace("hades", "<new-password>");
```

2. Edit `/etc/hades/daemon.conf` and replace the `ARANGO_PASSWORD=` value
(file is `root:hades 640`, so use `sudoedit` or `sudo $EDITOR`).
3. Restart the daemon:

```bash
sudo systemctl restart hades-daemon.service
sudo systemctl status hades-daemon.service --no-pager
```

For interactive CLI use (outside the daemon), export `ARANGO_PASSWORD` in your
shell — it takes precedence over anything in YAML.

