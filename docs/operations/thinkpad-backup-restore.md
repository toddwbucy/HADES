# ThinkPad workspace backup and recovery

This procedure was validated on 2026-10-18 with ArangoDB 3.12.12-1 and HADES
commit `15e2c9565a9c051ca8a2b17c625cc6005b3aa0fa`. It covers the local
`Conveyance_KG` workspace. Adapt paths from the
[service configuration](../installation-thinkpad-services.md).

For workspaces created with a saved embedding profile, also follow the
[model-bound recovery contract](model-bound-recovery.md). This older workstation
rehearsal does not cover setup ledgers, provider registrations or ownership
reconciliation after restoring a managed database. The
[Managed workspace](#managed-workspace) section below adds the ledger and the registry
to the artifact.

## Backup scope

Preserve a logical database dump including system collections: `_graphs` carries
named graph definitions. Reading source bytes, converted documents, reading
positions, corrections, highlights, archived readings, graph material, conversation
bindings and saved exchanges are database records. Original reading uploads are
stored in `hades_reading_sources.content_base64`; their keys are SHA-256 hashes.

Also preserve `~/.config/hades/`, HADES user units and drop-ins, and the matching
`hades`/`hades-viewer` binaries. These files contain credentials: keep the backup
directory mode 0700 and files mode 0600. Do not commit or attach the backup.

Full provider transcripts are separate. Copy the Codex session JSONL files
referenced by `hades_reading_conversations.thread_id`, preserving their relative
paths under `~/.codex/`. Verify stable hashes while copying. This rehearsal did
not copy Codex authentication or unrelated conversation history.

Model weights, downloadable caches, source builds and live RocksDB files are not
part of this logical backup. Preserve external ingestion originals separately;
the reading archive only covers sources uploaded through guided reading.

## Create a consistent artifact

1. Finish active HADES turns and ingestion. Prevent other CLI/MCP clients from
   writing during the backup. Record collection counts, canonical document
   hashes, schemas and indexes. Exclude only top-level `_rev` from document
   comparisons; preserve application revisions and all content.
2. Stop `hades-viewer.service`, then `hades-daemon.service`. Leave ArangoDB running.
   Arrange unconditional restart of both application services if any step fails.
3. Use matching `arangodump` with an explicitly selected socket and database:

   ```bash
   umask 077
   backup_base="$HOME/.local/share/hades/backups"
   mkdir -p "$backup_base"
   backup_dir=$(mktemp -d "$backup_base/workspace-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
   chmod 700 "$backup_dir"
   arangodump --configuration none \
     --server.endpoint unix:///home/todd/.local/share/hades/run/arango.sock \
     --server.database Conveyance_KG --server.username root \
     --threads 1 --include-system-collections true \
     --output-directory "$backup_dir/dump"
   ```

   Enter the password at its prompt; do not place it literally on the command
   line. Automated runs can use `--server.password '@HADES_BACKUP_PASSWORD@'`
   with a privately supplied environment value.
4. Copy the configuration, units, binaries and referenced transcripts. Compare
   the database inventory again, including every system collection (notably
   `_analyzers` and `_aqlfunctions`); reject the artifact if it changed. Save hashes
   for every copied file. Restart the daemon, check readiness, then start the
   viewer. Refresh the browser; a saved conversation may require **Resume**.

The recorded run used the private local driver
`.local-install/backup-workspace.py`. Its `manifest.json` contains `files` (relative
path to SHA-256), `collections` (count, canonical SHA-256, type, schema,
computedValues and indexes), `provider_transcripts`, and
`empty_bindings_without_provider_transcript`. It also records exclusions and the
installed revision. The integrity checker rejects missing, changed, symlinked,
escaping and unmanifested inputs before launching a restore. This is evidence
for this workstation procedure, not a general automated backup installer.

## Validate in isolation

Never restore over the active research database as a test. The verifier creates
a new mode-0700 `/tmp/hades-recovery-*` directory, starts a bounded private
ArangoDB server on its own Unix socket, and restores into
`hades_recovery_fixture`. It accepts no target endpoint. HADES receives a new
configuration with chat disabled and nonexistent extraction/embedding endpoints.

```bash
python3 scripts/verify_workspace_backup.py \
  --backup "$backup_dir" \
  --arangod ~/.local/opt/arangodb-3.12.12.1/sbin/arangod \
  --arangorestore ~/.local/opt/arangodb-3.12.12.1/bin/arangorestore \
  --hades ~/.local/bin/hades
```

The verifier checks document contents, named graph records, collection schemas,
index definitions, original source hashes and file hashes. It opens and exports
every reading through the installed HADES CLI. Private processes stop on success
or failure; restore data and logs remain in the private directory for inspection.
The probe does not run model inference or resume any provider conversation.

## Recorded result and limits

Backup: `~/.local/share/hades/backups/workspace-20261018T214440Z/`.
Recovery evidence: `/tmp/hades-recovery-g_yzgmo2/result.json`.
The [sanitized result](../specs/thinkpad-recovery-result.json) contains counts only.

- All 26 collections, including system collections, and 46 documents matched, including graph records,
  schemas and indexes. All 76 backed-up files passed SHA-256 verification.
- Three original source hashes matched. All six readings, including removed
  desk items, opened and exported through HADES.
- Three conversation bindings and two saved exchanges matched. Both existing
  provider transcript files were copied and verified.
- One pre-existing empty binding has no exchanges and no provider transcript;
  this backup cannot supply history that was never present on disk.
- Live services passed readiness/authentication checks after the backup.

The restore command took approximately 0.27 seconds for this small database;
this is not full recovery time or a performance target. User accounts/permissions
in `_system` were not restored, and provider authentication/resumption was not
tested. A real replacement host must recreate database users/grants, install the
matching services and models, adapt paths, authenticate Codex, and explicitly
verify conversation discovery before resuming work.

This is one on-device backup. Off-device storage, encryption for transport,
retention, scheduling, freshness monitoring and recovery objectives remain open.
Do not treat local binary rollback backups as database recovery coverage.

## Managed workspace

This section extends the procedure above to a database created by `hades db setup`,
the first being `Conveyance_KG_v2` in the
[managed workspace runbook](../installation-thinkpad.md#managed-workspace-2026-11-xx).
It was written before its first run; the runbook records the run. A database-only
backup cannot restore the setup ownership record, so one artifact holds three parts
taken in one quiet window: the database dump, the complete setup ledger, and the model
registry's records with the artifacts they name. The
[model-bound recovery contract](model-bound-recovery.md) says why each part matters
and where restore stops.

Each artifact meets three requirements:

- Every writer is quiesced, the viewer and then the daemon are stopped, the inventory
  is recorded before and compared after, referenced transcripts are copied and hashed,
  and `manifest.json` is written, exactly as in
  [Create a consistent artifact](#create-a-consistent-artifact). Writers include
  `hades db setup` and registry registration: run neither during a backup.
- The ledger root is copied complete, including its empty `.lock` file. From the
  registry, `records/` and every `artifacts/<snapshot>` that a record names are copied.
  `sessions/` is excluded as disposable, and restore recreates it empty and private.
  The registry's own `.lock` is not copied either; the registry recreates it, mode
  0600, on first use.
- Ownership and modes are preserved on both copies and checked against an inventory of
  the source, and every copied file is listed in the manifest's `files`, because the
  verifier hashes only manifested files and rejects unmanifested ones.

The verified provider service may keep running. It reads the registry when it starts
and writes only under `sessions/`, which is excluded.

### Steps

The private driver `.local-install/backup-workspace.py` makes the artifact, as it does
for `Conveyance_KG`: it quiesces, stops the viewer and the daemon, dumps the database
with system collections, copies configuration, units, binaries and transcripts,
compares the inventory, writes `manifest.json` and restarts the services. For a
managed database it must also run the ledger and registry copy below after the dump
and before it hashes the manifest, with `backup_dir` set to its backup directory. The
block is the reference for that copy; it was dry-run on a synthetic ledger and
registry, not yet on this laptop. `DATA` is the resolved directory because the ledger
and registry paths in the configuration are resolved:

```bash
DATA=$(readlink -f ~/.local/share/hades)
REG="$DATA/model-registry"
inventory() {
  python3 - "$@" <<'EOF'
import hashlib, json, os, stat, sys
root, parts = sys.argv[1], sys.argv[2:] or ["."]
rows = []
def add(path):
    info = os.lstat(path)
    row = {"path": os.path.relpath(path, root), "mode": oct(stat.S_IMODE(info.st_mode)),
           "uid": info.st_uid, "gid": info.st_gid,
           "type": "dir" if stat.S_ISDIR(info.st_mode)
           else "file" if stat.S_ISREG(info.st_mode) else "other"}
    if row["type"] == "file":
        with open(path, "rb") as stream:
            row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    rows.append(row)
add(root)
for part in parts:
    parent = os.path.dirname(os.path.normpath(part))
    while parent:
        add(os.path.join(root, parent))
        parent = os.path.dirname(parent)
    for top, dirs, files in os.walk(os.path.join(root, part)):
        add(top)
        for name in files:
            add(os.path.join(top, name))
unique = {row["path"]: row for row in rows}
print(json.dumps([unique[k] for k in sorted(unique)], indent=1))
EOF
}
mapfile -t snaps < <(python3 -c 'import json, pathlib, sys
records = pathlib.Path(sys.argv[1]).glob("*.json")
print("\n".join(sorted({json.loads(p.read_text())["snapshot"] for p in records})))' \
  "$REG/records")
parts=(records "${snaps[@]/#/artifacts/}")
inv="$backup_dir/inventory"
install -d -m 700 "$inv" "$backup_dir/model-registry" \
  "$backup_dir/model-registry/artifacts"
inventory "$DATA/setup-ledger" > "$inv/ledger-source-before.json"
inventory "$REG" "${parts[@]}" > "$inv/registry-source-before.json"
cp -a "$DATA/setup-ledger" "$backup_dir/setup-ledger"
cp -a "$REG/records" "$backup_dir/model-registry/"
for snap in "${snaps[@]}"; do
  cp -a "$REG/artifacts/$snap" "$backup_dir/model-registry/artifacts/"
done
inventory "$backup_dir/setup-ledger" > "$inv/ledger-copy.json"
inventory "$backup_dir/model-registry" "${parts[@]}" > "$inv/registry-copy.json"
inventory "$DATA/setup-ledger" > "$inv/ledger-source-after.json"
inventory "$REG" "${parts[@]}" > "$inv/registry-source-after.json"
for pair in ledger-source-before:ledger-copy ledger-source-before:ledger-source-after \
    registry-source-before:registry-copy registry-source-before:registry-source-after; do
  cmp "$inv/${pair%%:*}.json" "$inv/${pair##*:}.json" || echo "REJECT: $pair differs"
done
```

Every `cmp` must print nothing; on `REJECT`, the driver must fail. A difference means a
mode, owner or byte changed, or a writer ran during the copy. Building the inventories
reads the registered artifacts, 7.4 GiB, three times, and the manifest and the
verifier read them again.

### Driver changes

The driver is the operator's to extend. As of 2026-11-10 it hardcodes `Conveyance_KG`,
requires the reading-desk collections, sends its preflight to `Conveyance_KG`, and has
no step for the copy above. It must:

- take the database name as an argument, and send its preflight and readiness checks
  to that database;
- tolerate a database without the reading-desk collections, since the first managed
  backup is taken before the reading desk moves in;
- run the copy block above after the dump and before computing the manifest, and fail
  when any comparison differs;
- list every file under `setup-ledger/`, `model-registry/` and `inventory/` in `files`
  with its SHA-256, as it already does for the dump, configuration, units, binaries
  and transcripts;
- record `model-registry/sessions/` and the registry `.lock` among its exclusions;
- keep writing its backup directory to `.local-install/latest-workspace-backup.txt`,
  which the runbook reads.

### Verify

Run `scripts/verify_workspace_backup.py` as in
[Validate in isolation](#validate-in-isolation). It needs the reading-desk
collections, so it cannot pass on a managed backup taken before the reading desk
moves in; the runbook verifies its second backup only. It stages every manifested file
in a new `/tmp/hades-recovery-*` directory, and `/tmp` on this laptop is a 32 GiB
tmpfs shared with other work, and the registered artifacts alone take 7.4 GiB of it
while it runs. Require at least 15 GiB available first
(`test "$(df --output=avail -B1G /tmp | tail -1)" -ge 15`), and remove the directory
once its `result.json` is recorded.

The verifier compares content, not modes or owners; the inventory comparison above is
that evidence. A pass proves the records restore in isolation. It does not reactivate
the managed workspace: a restored database has a new incarnation that the copied
receipt cannot authorize, and ownership rebinding is not supported.

Registered artifacts are frozen read-only and keep those modes in the copy. To delete
an old backup, run `chmod -R u+w` on it first.
