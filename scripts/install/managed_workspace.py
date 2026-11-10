#!/usr/bin/env python3
"""JSON checks and the managed backup copy for thinkpad-managed-workspace.sh (#225).

The installer calls one subcommand at a time with explicit paths. Each prints a short
result on stdout, or one line on stderr and exit 1 when an expectation fails. Nothing
here contacts a service: the installer runs `hades`, `curl` and the backup driver and
hands their output files to this helper. Standard library only, Python 3.11 or later.

`add-managed` copies the setup ledger and the model registry's records and referenced
artifacts into a backup the private driver made, checks modes, owners and bytes
against inventories of the source taken before and after, and adds every copied file
to the backup's manifest. It is the reference copy of the managed workspace backup
(docs/operations/thinkpad-backup-restore.md, "Managed workspace").
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

MOVING = re.compile(r"hades_reading_.+|reading_.+_(nodes|edges)")
VERIFIER_DIR = re.compile(r"/tmp/hades-recovery-[A-Za-z0-9_]+")


class Refused(Exception):
    """An expectation failed; the message is the one line the operator sees."""


def load(path):
    """Parse a JSON file, refusing an empty or malformed one by name."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"{Path(path).name} is not readable JSON: {exc}") from None


def data(path):
    """Return the `data` of a HADES CLI envelope."""
    value = load(path)
    if not isinstance(value, dict) or "data" not in value:
        raise Refused(f"{Path(path).name} is not a HADES envelope")
    return value["data"]


def require(condition, message):
    if not condition:
        raise Refused(message)


# Backup copy -------------------------------------------------------------------------

def inventory(root, parts=None):
    """List mode, owner, type and SHA-256 of `root` and each part beneath it, sorted.

    The ancestors of every part are listed too, so a mode change on an intermediate
    directory (artifacts/, for instance) is caught as well.
    """
    root = Path(root)
    rows = {}

    def add(path):
        info = os.lstat(path)
        kind = ("dir" if stat.S_ISDIR(info.st_mode)
                else "file" if stat.S_ISREG(info.st_mode) else "other")
        row = {"path": os.path.relpath(path, root), "mode": oct(stat.S_IMODE(info.st_mode)),
               "uid": info.st_uid, "gid": info.st_gid, "type": kind}
        if kind == "file":
            with open(path, "rb") as stream:
                row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        rows[row["path"]] = row

    add(root)
    for part in parts or ["."]:
        parent = os.path.dirname(os.path.normpath(part))
        while parent:
            add(root / parent)
            parent = os.path.dirname(parent)
        for top, _dirs, files in os.walk(root / part):
            add(top)
            for name in files:
                add(os.path.join(top, name))
    return [rows[key] for key in sorted(rows)]


def copy_tree(source, target):
    """Copy with `cp -a`, which keeps modes, owners and times, frozen directories too."""
    subprocess.run(["cp", "-a", "--", str(source), str(target)], check=True)


def write_private(path, text):
    """Replace a file atomically with a private one."""
    temp = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)
    os.replace(temp, path)


def add_managed(backup, database, ledger, registry, copy=copy_tree):
    """Copy the ledger and registry parts into `backup` and extend its manifest.

    Refuses a backup of another database or one that already holds a copy. On any
    inventory difference it raises before the manifest changes, so the copied files
    stay unmanifested and the verifier rejects the backup.
    """
    backup, ledger, registry = Path(backup), Path(ledger), Path(registry)
    manifest_path = backup / "manifest.json"
    manifest = load(manifest_path)
    require(manifest.get("database") == database,
            f"backup {backup.name} backs up {manifest.get('database')}, not {database}")
    targets = [backup / "setup-ledger", backup / "model-registry", backup / "inventory"]
    require("managed_workspace" not in manifest and not any(p.exists() for p in targets),
            f"backup {backup.name} already holds a managed workspace copy")
    # Checked before anything is created, so a wrong path leaves the backup untouched.
    require(ledger.is_dir(), f"the setup ledger {ledger} is not a directory")
    records = sorted((registry / "records").glob("*.json"))
    require(records, "the model registry holds no registration record")
    snapshots = sorted({load(p)["snapshot"] for p in records})
    parts = ["records", *(f"artifacts/{s}" for s in snapshots)]
    inv = backup / "inventory"
    for directory in (inv, backup / "model-registry", backup / "model-registry/artifacts"):
        directory.mkdir(mode=0o700)

    def record(name, rows):
        write_private(inv / f"{name}.json", json.dumps(rows, indent=1) + "\n")
        return rows

    ledger_before = record("ledger-source-before", inventory(ledger))
    registry_before = record("registry-source-before", inventory(registry, parts))
    copy(ledger, backup / "setup-ledger")
    copy(registry / "records", backup / "model-registry/records")
    for snapshot in snapshots:
        copy(registry / "artifacts" / snapshot, backup / "model-registry/artifacts" / snapshot)
    compared = {
        "ledger-source-before:ledger-copy":
            (ledger_before, record("ledger-copy", inventory(backup / "setup-ledger"))),
        "ledger-source-before:ledger-source-after":
            (ledger_before, record("ledger-source-after", inventory(ledger))),
        "registry-source-before:registry-copy":
            (registry_before,
             record("registry-copy", inventory(backup / "model-registry", parts))),
        "registry-source-before:registry-source-after":
            (registry_before, record("registry-source-after", inventory(registry, parts))),
    }
    for pair, (left, right) in compared.items():
        # A difference is a mode, owner or byte that changed, or a writer during the copy.
        require(left == right, f"REJECT: {pair} differs; this backup is not certified")
    for part in targets:
        for path in sorted(part.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest["files"][str(path.relative_to(backup))] = digest
    manifest["managed_workspace"] = {
        "ledger": "setup-ledger", "registry": "model-registry",
        "records": [p.name for p in records], "snapshots": snapshots,
        "inventory_compared": sorted(compared)}
    manifest.setdefault("excluded", []).extend([
        "model-registry/sessions/ (disposable; restore recreates it empty)",
        "model-registry/.lock (the registry recreates it on first use)"])
    write_private(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return {"backup": backup.name, "snapshots": snapshots,
            "files_added": sum(1 for name in manifest["files"]
                               if name.startswith(("setup-ledger/", "model-registry/",
                                                   "inventory/")))}


def check_manifest(backup, database):
    """Run the verifier's own file check, and require the managed copy."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verify_workspace_backup import verified_files

    backup = Path(backup).resolve()
    manifest = load(backup / "manifest.json")
    require(manifest.get("database") == database,
            f"backup backs up {manifest.get('database')}, not {database}")
    require("managed_workspace" in manifest, "backup has no managed workspace copy")
    try:
        files = verified_files(backup, manifest)
    except ValueError as exc:
        raise Refused(str(exc)) from None
    return len(files)


# Subcommands ------------------------------------------------------------------------

def cmd_json_field(a):
    value = load(a.file)
    require(isinstance(value, dict) and a.key in value, f"{Path(a.file).name} has no {a.key}")
    return str(value[a.key])


def cmd_register_state(a):
    path = Path(a.registry) / "records" / f"{a.provider}.json"
    if not path.exists():
        return "absent"
    return "match" if load(path).get("manifest_md5") == a.md5 else "differ"


def cmd_check_register(a):
    record = load(a.file)
    require(record.get("manifest_md5") == a.md5,
            f"manifest MD5 {record.get('manifest_md5')} is not the qualified {a.md5}")
    require(record.get("loaded_verified") is False, "registration unexpectedly loaded a model")
    return f"manifest MD5 matches; {record.get('files')} files"


def cmd_text_state(a):
    """absent: no named section; match: the exact block is present; differ: otherwise."""
    text, block = Path(a.file).read_text(), Path(a.block).read_text()
    if block.strip("\n") in text:
        return "match"
    headers = re.compile(r"^(%s):" % "|".join(map(re.escape, a.headers)), re.MULTILINE)
    return "differ" if headers.search(text) else "absent"


def cmd_check_status(a):
    status = load(a.file)
    require(status.get("model") == a.model and status.get("space") == a.space
            and status.get("loaded") is True, f"provider status is {status}")
    return "status ok"


def cmd_check_providers(a):
    """`hades status` lists the provider as serving the late space with a loaded model."""
    entries = data(a.file).get("verified_providers")
    require(isinstance(entries, list), "hades status has no verified_providers")
    entry = next((e for e in entries if e.get("id") == a.provider), None)
    require(entry is not None, f"verified_providers does not list {a.provider}")
    require(entry.get("space") == "late" and entry.get("loaded") is True,
            f"verified_providers reports {entry}")
    return f"verified_providers: {a.provider} late loaded"


def cmd_check_verify(a):
    report = load(a.file)
    model, caps = report["model"], report["capabilities"]
    require(model.get("revision") == a.revision, f"verify reports revision {model}")
    require(model.get("manifest_md5") == a.md5, f"verify reports manifest {model}")
    require(caps.get("late_chunking") is True and caps.get("tokenizer_offsets") is True,
            f"verify reports capabilities {caps}")
    require(caps.get("max_input_tokens") == a.max_tokens,
            f"verify reports max_input_tokens {caps.get('max_input_tokens')}")
    return f"verified; space {report['receipt']['space_id']}"


def receipt_of(path, operation, database):
    try:
        value = data(path)
    except Refused:
        return "error", None
    receipt = value.get("receipt")
    if receipt is None:
        return "none", None
    require(receipt.get("operation_id") == operation and receipt.get("database") == database,
            f"operation {receipt.get('operation_id')} names database {receipt.get('database')}")
    return ("ready" if receipt.get("status") == "ready" and receipt.get("step") == "complete"
            else receipt.get("status")), receipt


def cmd_receipt_state(a):
    return receipt_of(a.file, a.operation, a.database)[0]


def cmd_check_receipt(a):
    state, run = receipt_of(a.run, a.operation, a.database)
    require(state == "ready", f"setup run receipt is {state}")
    state, inspected = receipt_of(a.inspect, a.operation, a.database)
    require(state == "ready", f"inspect receipt is {state}")
    require(inspected["request_sha256"] == run["request_sha256"],
            "inspect reports a different request_sha256 from run")
    action = data(a.diagnose).get("action")
    require(action == "none", f"diagnose recommends {action}")
    return f"ready; request_sha256 {run['request_sha256']}"


def cmd_diagnose_action(a):
    return data(a.file).get("action")


def aql_results(path):
    results = data(path).get("results")
    require(isinstance(results, list), f"{Path(path).name} has no AQL results")
    return results


def cmd_moving_aql(a):
    names = aql_results(a.file)[0]
    return "\n".join(sorted(n for n in names if MOVING.fullmatch(n)))


def cmd_moving_dump(a):
    names = {load(p)["parameters"]["name"] for p in Path(a.dump).glob("*.structure.json")}
    return "\n".join(sorted(n for n in names if MOVING.fullmatch(n)))


def cmd_staying(a):
    moving = set(Path(a.moving).read_text().split())
    names = aql_results(a.file)[0]
    return json.dumps([n for n in names if n not in moving and not n.startswith("_")])


def cmd_manifest_counts(a):
    """The backup's counts for the moving collections, in the live counts query's format.

    Compared with the live source before the restore, it proves the final backup still
    matches the quiet database, however long ago it was taken (#225).
    """
    collections = load(a.manifest)["collections"]
    names = Path(a.moving).read_text().split()
    missing = [n for n in names if n not in collections]
    require(not missing, f"the backup manifest lacks {missing}")
    return json.dumps([[n, collections[n]["count"]] for n in names])


def cmd_names_bind(a):
    return json.dumps({"names": Path(a.moving).read_text().split()})


def cmd_check_graphs(a):
    """Every graph is a reading graph, and every collection it names moves with it."""
    moving = set(Path(a.moving).read_text().split())
    graphs = aql_results(a.file)
    for graph in graphs:
        require(graph["_key"].startswith("reading_"), f"graph {graph['_key']} is not a reading")
        names = set(graph.get("orphanCollections", []))
        for definition in graph["edgeDefinitions"]:
            names |= {definition["collection"], *definition["from"], *definition["to"]}
        require(names <= moving,
                f"graph {graph['_key']} names unlisted {sorted(names - moving)}")
    return f"graphs ok: {len(graphs)}"


def cmd_results(a):
    return json.dumps(aql_results(a.file))


def cmd_desk(a):
    value = data(a.file)
    value.pop("database", None)
    return json.dumps(value, sort_keys=True)


def cmd_verifier_dir(a):
    for line in Path(a.file).read_text().splitlines():
        if line.startswith("Private recovery artifacts: "):
            directory = line.split(": ", 1)[1]
            require(VERIFIER_DIR.fullmatch(directory),
                    "verifier printed an unexpected artifacts directory")
            return directory
    raise Refused("verifier printed no artifacts directory")


def cmd_check_ingest(a):
    envelope = load(a.file)
    value = envelope.get("data", {})
    require(envelope.get("success") is True and value.get("failed") == 0,
            f"ingest reports failures: {value.get('failed')}")
    require(value.get("completed") == a.count,
            f"ingest completed {value.get('completed')} of {a.count} files")
    require(value.get("embedding_profile_state") == "ready",
            f"embedding profile is {value.get('embedding_profile_state')}")
    return f"ingested {a.count} documents"


def cmd_query_sources(a):
    """Print the ranked sources; pass when an expected source is among them."""
    sources = [hit.get("source_id", "?") for hit in aql_results(a.file)]
    lines = [f"{rank}. {source}" for rank, source in enumerate(sources, 1)]
    # Managed search names a root-relative file `relative-file:<path>`
    # (profiled_documents/files.rs); match the path after the scheme.
    paths = [s.split(":", 1)[1] if s.startswith("relative-file:") else s for s in sources]
    hit = any(p == e or p.endswith("/" + e) for p in paths for e in a.expected)
    print("\n".join(lines))
    if not hit:
        print("FAIL: none of " + ", ".join(a.expected))
        raise SystemExit(1)
    return "PASS"


def cmd_add_managed(a):
    return json.dumps(add_managed(a.backup, a.database, a.ledger, a.registry))


def cmd_check_manifest(a):
    return f"{check_manifest(a.backup, a.database)} files verified"


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, func, *args, **typed):
        command = sub.add_parser(name)
        for arg in args:
            if arg.endswith("..."):
                command.add_argument(arg[:-3], nargs="+")
            else:
                command.add_argument(arg, type=typed.get(arg, str))
        command.set_defaults(func=func)

    add("json-field", cmd_json_field, "file", "key")
    add("register-state", cmd_register_state, "registry", "provider", "md5")
    add("check-register", cmd_check_register, "file", "md5")
    add("text-state", cmd_text_state, "file", "block", "headers...")
    add("check-status", cmd_check_status, "file", "model", "space")
    add("check-providers", cmd_check_providers, "file", "provider")
    add("check-verify", cmd_check_verify, "file", "revision", "md5", "max_tokens",
        max_tokens=int)
    add("receipt-state", cmd_receipt_state, "file", "operation", "database")
    add("check-receipt", cmd_check_receipt, "run", "inspect", "diagnose", "operation",
        "database")
    add("diagnose-action", cmd_diagnose_action, "file")
    add("moving-aql", cmd_moving_aql, "file")
    add("moving-dump", cmd_moving_dump, "dump")
    add("staying", cmd_staying, "file", "moving")
    add("names-bind", cmd_names_bind, "moving")
    add("manifest-counts", cmd_manifest_counts, "manifest", "moving")
    add("check-graphs", cmd_check_graphs, "file", "moving")
    add("results", cmd_results, "file")
    add("desk", cmd_desk, "file")
    add("verifier-dir", cmd_verifier_dir, "file")
    add("check-ingest", cmd_check_ingest, "file", "count", count=int)
    add("query-sources", cmd_query_sources, "file", "expected...")
    add("add-managed", cmd_add_managed, "backup", "database", "ledger", "registry")
    add("check-manifest", cmd_check_manifest, "backup", "database")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    try:
        result = args.func(args)
    except (Refused, KeyError, TypeError, OSError) as exc:
        if isinstance(exc, Refused):
            message = str(exc)
        elif isinstance(exc, OSError):
            # strerror, not str(exc): the latter carries the full path.
            message = f"{exc.strerror}: {Path(exc.filename or '?').name}"
        else:
            message = f"unexpected shape: {exc!r}"
        print(f"managed_workspace {args.command}: {message}", file=sys.stderr)
        return 1
    if result:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
