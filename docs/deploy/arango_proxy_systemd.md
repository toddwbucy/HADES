# Arango Proxy Systemd Units

This guide bootstraps the read-only and read-write HTTP/2 proxies via
systemd socket activation so they start alongside ArangoDB and enforce
the desired Unix-domain-socket permissions.

## Directory Layout

```
/etc/systemd/system/
  hades-arango-ro.socket
  hades-arango-ro.service
  hades-arango-rw.socket
  hades-arango-rw.service
```

The sockets live under `/run/hades/readonly/` and `/run/hades/readwrite/`
respectively; adjust the paths if your runtime directory differs.

## Socket Units

```ini
# /etc/systemd/system/hades-arango-ro.socket
[Unit]
Description=HADES Arango Read-only Proxy Socket
PartOf=arangodb3.service

[Socket]
ListenStream=/run/hades/readonly/arangod.sock
SocketUser=inference
SocketGroup=hades
SocketMode=0640
RemoveOnStop=yes

[Install]
WantedBy=sockets.target
```

```ini
# /etc/systemd/system/hades-arango-rw.socket
[Unit]
Description=HADES Arango Read-write Proxy Socket
PartOf=arangodb3.service

[Socket]
ListenStream=/run/hades/readwrite/arangod.sock
SocketUser=consolidation
SocketGroup=hades
SocketMode=0600
RemoveOnStop=yes

[Install]
WantedBy=sockets.target
```

## Service Units

```ini
# /etc/systemd/system/hades-arango-ro.service
[Unit]
Description=HADES Arango Read-only Proxy
Requires=arangodb3.service
After=arangodb3.service

[Service]
ExecStart=/usr/local/bin/hades-arango-ro-proxy
Environment=LISTEN_SOCKET=/run/hades/readonly/arangod.sock
Environment=UPSTREAM_SOCKET=/run/arangodb3/arangodb.sock
User=arangodb
Group=arangodb
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/hades-arango-rw.service
[Unit]
Description=HADES Arango Read-write Proxy
Requires=arangodb3.service
After=arangodb3.service

[Service]
ExecStart=/usr/local/bin/hades-arango-rw-proxy
Environment=LISTEN_SOCKET=/run/hades/readwrite/arangod.sock
Environment=UPSTREAM_SOCKET=/run/arangodb3/arangodb.sock
User=arangodb
Group=arangodb
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Adjust `ExecStart` paths if the proxy binaries live elsewhere (e.g.
`/opt/hades/bin/roproxy`). Both proxies should be compiled statically or
shipped alongside the deployment.

## Activation Steps

```bash
sudo mkdir -p /run/hades/readonly /run/hades/readwrite
sudo chown inference:hades /run/hades/readonly
sudo chown consolidation:hades /run/hades/readwrite
# Replace the user/group pairs above if your deployment uses different service accounts.

sudo systemctl daemon-reload
sudo systemctl enable --now hades-arango-ro.socket hades-arango-rw.socket
```

`systemctl status hades-arango-ro.socket` will show the socket path and
owner. The services auto-start when a client connects. To restart
ArangoDB and the proxies together:

```bash
sudo systemctl restart arangodb3.service
```

Because `PartOf=arangodb3.service` is set on the socket units, this also
tears down and recreates the proxy sockets.

## Optional: Peer Credential Checks

If you need belt-and-suspenders protection, wrap the proxy binary with a
small launcher that verifies `SO_PEERCRED` on accepted connections and
rejects unexpected UIDs/GIDs before proxying traffic.
