# ThinkPad workspace backup and recovery

This procedure was validated on 2026-10-18 with ArangoDB 3.12.12-1 and HADES
commit `15e2c9565a9c051ca8a2b17c625cc6005b3aa0fa`. It covers the local
`Conveyance_KG` workspace. Adapt paths from the
[service configuration](../installation-thinkpad-services.md).

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
