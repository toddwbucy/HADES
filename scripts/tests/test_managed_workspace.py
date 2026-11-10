"""Tests for scripts/install/managed_workspace.py, the installer's JSON and backup helper.

Everything runs on synthetic files: no ArangoDB, provider, systemd unit or live path.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "install"))
import managed_workspace as mw  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(data, success=True):
    return json.dumps({"success": success, "command": "x", "data": data,
                       "timestamp": "2026-11-10T00:00:00Z"})


class Scratch(unittest.TestCase):
    """A private scratch directory that is removed even when it holds frozen files."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mw-test-"))
        os.chmod(self.root, 0o700)

    def tearDown(self):
        subprocess.run(["chmod", "-R", "u+w", str(self.root)], check=True)
        shutil.rmtree(self.root)

    def write(self, relative, text, mode=0o600):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(mode)
        return path

    def call(self, *argv):
        """Run the helper's main() and return (exit code, stdout and stderr)."""
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            try:
                code = mw.main([str(a) for a in argv])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue()


class ManagedBackup(Scratch):
    """The ledger and registry copy, its inventory comparison and the manifest."""

    def setUp(self):
        super().setUp()
        self.ledger = self.root / "data/setup-ledger"
        self.ledger.mkdir(parents=True, mode=0o700)
        self.write("data/setup-ledger/.lock", "")
        self.write("data/setup-ledger/operations/op.json", '{"op": 1}')
        self.registry = self.root / "data/model-registry"
        for name in ("records", "artifacts", "sessions"):
            (self.registry / name).mkdir(parents=True, mode=0o700)
        self.registry.chmod(0o700)
        self.write("data/model-registry/.lock", "")
        self.write("data/model-registry/records/laptop-jina.json",
                   json.dumps({"snapshot": "model_aa", "manifest_md5": "m"}))
        frozen = self.write("data/model-registry/artifacts/model_aa/model.bin", "w", 0o400)
        frozen.parent.chmod(0o500)
        self.write("data/model-registry/artifacts/model_unreferenced/x", "u")
        self.write("data/model-registry/sessions/s1/x", "s")
        self.backup = self.root / "backups/workspace-1"
        dump = self.write("backups/workspace-1/dump/dump.json", '{"database": "DB2"}')
        self.write("backups/workspace-1/manifest.json", json.dumps({
            "database": "DB2", "collections": {}, "provider_transcripts": [],
            "excluded": ["raw live RocksDB files"],
            "files": {"dump/dump.json": sha(dump)}}))

    def add(self, **kwargs):
        return mw.add_managed(self.backup, "DB2", self.ledger, self.registry, **kwargs)

    def test_copies_ledger_and_referenced_artifacts_only(self):
        self.add()
        copied = {str(p.relative_to(self.backup)) for p in self.backup.rglob("*")
                  if p.is_file()}
        self.assertIn("setup-ledger/.lock", copied)
        self.assertIn("setup-ledger/operations/op.json", copied)
        self.assertIn("model-registry/records/laptop-jina.json", copied)
        self.assertIn("model-registry/artifacts/model_aa/model.bin", copied)
        self.assertFalse(any(p.startswith(("model-registry/sessions",
                                           "model-registry/artifacts/model_unref"))
                             for p in copied), copied)
        self.assertNotIn("model-registry/.lock", copied)
        mode = (self.backup / "model-registry/artifacts/model_aa/model.bin").stat().st_mode
        self.assertEqual(stat.S_IMODE(mode), 0o400)

    def test_manifest_lists_every_file_and_passes_the_verifier_file_check(self):
        summary = self.add()
        manifest = json.loads((self.backup / "manifest.json").read_text())
        self.assertEqual(manifest["managed_workspace"]["snapshots"], ["model_aa"])
        self.assertIn("model-registry/sessions/", " ".join(manifest["excluded"]))
        self.assertEqual(summary["snapshots"], ["model_aa"])
        code, out = self.call("check-manifest", self.backup, "DB2")
        self.assertEqual(code, 0, out)
        self.assertIn("files verified", out)

    def test_wrong_database_is_refused_before_copying(self):
        manifest = self.backup / "manifest.json"
        manifest.write_text(manifest.read_text().replace('"DB2"', '"DB1"', 1))
        with self.assertRaisesRegex(mw.Refused, "backs up DB1, not DB2"):
            self.add()
        self.assertFalse((self.backup / "setup-ledger").exists())

    def test_a_second_add_is_refused(self):
        self.add()
        with self.assertRaisesRegex(mw.Refused, "already holds"):
            self.add()

    def test_a_mode_changed_in_the_copy_is_rejected(self):
        def drifting_copy(source, target):
            mw.copy_tree(source, target)
            if target.name == "setup-ledger":
                (target / "operations/op.json").chmod(0o644)

        with self.assertRaisesRegex(mw.Refused, "REJECT: ledger-source-before:ledger-copy"):
            self.add(copy=drifting_copy)
        manifest = json.loads((self.backup / "manifest.json").read_text())
        self.assertNotIn("managed_workspace", manifest)

    def test_unmanifested_file_fails_the_manifest_check(self):
        self.add()
        self.write("backups/workspace-1/stray.txt", "x")
        code, out = self.call("check-manifest", self.backup, "DB2")
        self.assertEqual(code, 1)
        self.assertIn("unmanifested backup files", out)


class StateChecks(Scratch):
    """The small JSON checks the installer runs between commands."""

    def test_json_field(self):
        dump = self.write("dump.json", json.dumps({"database": "DB1"}))
        self.assertEqual(self.call("json-field", dump, "database"), (0, "DB1\n"))
        self.assertEqual(self.call("json-field", dump, "other")[0], 1)

    def test_register_state(self):
        registry = self.root / "registry"
        self.assertEqual(self.call("register-state", registry, "p", "abc")[1], "absent\n")
        self.write("registry/records/p.json", json.dumps({"manifest_md5": "abc"}))
        self.assertEqual(self.call("register-state", registry, "p", "abc")[1], "match\n")
        self.assertEqual(self.call("register-state", registry, "p", "def")[1], "differ\n")

    def test_check_register(self):
        record = self.write("r.json", json.dumps({"manifest_md5": "abc", "files": 9,
                                                  "loaded_verified": False}))
        self.assertEqual(self.call("check-register", record, "abc")[0], 0)
        code, out = self.call("check-register", record, "def")
        self.assertEqual(code, 1)
        self.assertIn("manifest MD5", out)

    def test_text_state(self):
        block = self.write("block", "\nsetup:\n  ledger_root: /x\n")
        yaml = self.write("hades.yaml", "database:\n  name: x\n")
        self.assertEqual(self.call("text-state", yaml, block, "setup", "registered_providers")[1],
                         "absent\n")
        yaml.write_text("database:\n  name: x\n\nsetup:\n  ledger_root: /x\n")
        self.assertEqual(self.call("text-state", yaml, block, "setup")[1], "match\n")
        yaml.write_text("database:\n  name: x\nsetup:\n  ledger_root: /y\n")
        self.assertEqual(self.call("text-state", yaml, block, "setup")[1], "differ\n")

    def test_status_and_verify(self):
        status = self.write("s.json", json.dumps({"model": "jinaai/jina-embeddings-v4",
                                                  "space": "late", "loaded": True}))
        self.assertEqual(self.call("check-status", status, "jinaai/jina-embeddings-v4",
                                   "late")[0], 0)
        status.write_text(json.dumps({"model": "jinaai/jina-embeddings-v4", "space": "late",
                                      "loaded": False}))
        self.assertEqual(self.call("check-status", status, "jinaai/jina-embeddings-v4",
                                   "late")[0], 1)
        verify = self.write("v.json", json.dumps({
            "model": {"revision": "r", "manifest_md5": "m"},
            "capabilities": {"late_chunking": True, "tokenizer_offsets": True,
                             "max_input_tokens": 20000},
            "receipt": {"space_id": "space-1"}}))
        code, out = self.call("check-verify", verify, "r", "m", 20000)
        self.assertEqual((code, out), (0, "verified; space space-1\n"))
        self.assertEqual(self.call("check-verify", verify, "r", "m", 4096)[0], 1)

    def receipt(self, status="ready", step="complete", sha="h"):
        return {"operation_id": "op", "database": "DB2", "request_sha256": sha,
                "status": status, "step": step, "diagnostic": None}

    def test_receipt_state(self):
        missing = self.write("i0.json", envelope({"operation_id": "op", "receipt": None},
                                                 success=False))
        self.assertEqual(self.call("receipt-state", missing, "op", "DB2")[1], "none\n")
        ready = self.write("i1.json", envelope({"operation_id": "op",
                                                "receipt": self.receipt()}))
        self.assertEqual(self.call("receipt-state", ready, "op", "DB2")[1], "ready\n")
        held = self.write("i2.json", envelope({"operation_id": "op", "receipt": self.receipt(
            "needs_reconciliation", "marker")}, success=False))
        self.assertEqual(self.call("receipt-state", held, "op", "DB2")[1],
                         "needs_reconciliation\n")
        other = self.write("i3.json", envelope({"operation_id": "op", "receipt": dict(
            self.receipt(), database="Other")}))
        self.assertEqual(self.call("receipt-state", other, "op", "DB2")[0], 1)
        self.assertEqual(self.call("receipt-state", self.write("e.json", ""), "op",
                                   "DB2")[1], "error\n")

    def test_check_receipt(self):
        run = self.write("run.json", envelope({"receipt": self.receipt()}))
        inspect = self.write("inspect.json", envelope({"operation_id": "op",
                                                       "receipt": self.receipt()}))
        diagnose = self.write("diagnose.json", envelope({"action": "none",
                                                         "receipt": self.receipt()}))
        self.assertEqual(self.call("check-receipt", run, inspect, diagnose, "op", "DB2")[0], 0)
        inspect.write_text(envelope({"operation_id": "op", "receipt": self.receipt(sha="x")}))
        code, out = self.call("check-receipt", run, inspect, diagnose, "op", "DB2")
        self.assertEqual(code, 1)
        self.assertIn("request_sha256", out)
        self.assertEqual(self.call("diagnose-action", diagnose)[1], "none\n")

    def test_moving_lists(self):
        names = ["hades_reading_sessions", "reading_ab_nodes", "reading_ab_edges",
                 "documents", "_graphs", "hades_schema", "reading_notes"]
        aql = self.write("names.json", envelope({"results": [names]}))
        code, out = self.call("moving-aql", aql)
        self.assertEqual(out, "hades_reading_sessions\nreading_ab_edges\nreading_ab_nodes\n")
        dump = self.root / "dump"
        for name in names:
            self.write(f"dump/{name}_x.structure.json",
                       json.dumps({"parameters": {"name": name}}))
        self.assertEqual(self.call("moving-dump", dump)[1], out)
        moving = self.write("moving.txt", out)
        self.assertEqual(self.call("staying", aql, moving)[1],
                         '["documents", "hades_schema", "reading_notes"]\n')
        self.assertEqual(json.loads(self.call("names-bind", moving)[1]),
                         {"names": out.split()})

    def test_check_graphs(self):
        moving = self.write("moving.txt", "reading_ab_edges\nreading_ab_nodes\n")
        good = {"_key": "reading_ab", "orphanCollections": [], "edgeDefinitions": [
            {"collection": "reading_ab_edges", "from": ["reading_ab_nodes"],
             "to": ["reading_ab_nodes"]}]}
        graphs = self.write("g.json", envelope({"results": [good]}))
        self.assertEqual(self.call("check-graphs", graphs, moving)[1], "graphs ok: 1\n")
        stray = dict(good, orphanCollections=["documents"])
        graphs.write_text(envelope({"results": [stray]}))
        self.assertEqual(self.call("check-graphs", graphs, moving)[0], 1)
        graphs.write_text(envelope({"results": [dict(good, _key="other")]}))
        self.assertEqual(self.call("check-graphs", graphs, moving)[0], 1)

    def test_results_and_desk(self):
        rows = self.write("r.json", envelope({"results": [["a", 1]]}))
        self.assertEqual(self.call("results", rows)[1], '[["a", 1]]\n')
        desk = self.write("d.json", envelope({"database": "DB", "b": 1, "a": 2}))
        self.assertEqual(self.call("desk", desk)[1], '{"a": 2, "b": 1}\n')

    def test_check_providers(self):
        entry = {"id": "laptop-jina", "model": "m", "space": "late", "loaded": True}
        status = self.write("st.json", envelope({"verified_providers": [entry]}))
        self.assertEqual(self.call("check-providers", status, "laptop-jina"),
                         (0, "verified_providers: laptop-jina late loaded\n"))
        status.write_text(envelope({"verified_providers": [
            {"id": "laptop-jina", "error": "connection refused"}]}))
        self.assertEqual(self.call("check-providers", status, "laptop-jina")[0], 1)
        status.write_text(envelope({"embedder": {}}))
        self.assertEqual(self.call("check-providers", status, "laptop-jina")[0], 1)

    def test_manifest_counts_match_the_live_counts_format(self):
        manifest = self.write("manifest.json", json.dumps({"collections": {
            "reading_ab_nodes": {"count": 3}, "hades_reading_sessions": {"count": 1},
            "documents": {"count": 9}}}))
        moving = self.write("moving.txt", "hades_reading_sessions\nreading_ab_nodes\n")
        live = self.write("live.json", envelope({"results": [["hades_reading_sessions", 1],
                                                             ["reading_ab_nodes", 3]]}))
        self.assertEqual(self.call("manifest-counts", manifest, moving)[1],
                         self.call("results", live)[1])

    def test_a_filesystem_error_is_a_one_line_refusal(self):
        backup = self.root / "b"
        self.write("b/manifest.json", json.dumps({"database": "DB2", "files": {}}))
        self.write("registry/records/p.json", json.dumps({"snapshot": "s"}))
        code, out = self.call("add-managed", backup, "DB2", self.root / "no-ledger",
                              self.root / "registry")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", out)

    def test_verifier_dir(self):
        log = self.write("v.log", "Private recovery artifacts: /tmp/hades-recovery-ab12\n{}\n")
        self.assertEqual(self.call("verifier-dir", log)[1], "/tmp/hades-recovery-ab12\n")
        log.write_text("Private recovery artifacts: /var/x\n")
        self.assertEqual(self.call("verifier-dir", log)[0], 1)

    def test_check_ingest(self):
        ok = self.write("i.json", envelope({"completed": 3, "failed": 0,
                                            "embedding_profile_state": "ready"}))
        self.assertEqual(self.call("check-ingest", ok, 3), (0, "ingested 3 documents\n"))
        self.assertEqual(self.call("check-ingest", ok, 4)[0], 1)

    def test_query_sources(self):
        # Managed search names a root-relative file as relative-file:<path>
        # (profiled_documents/files.rs), so the match must look past the scheme.
        hits = [{"source_id": "relative-file:operations/x.md"},
                {"source_id": "relative-file:specs/db-setup-door.md"}]
        result = self.write("q.json", envelope({"results": hits}))
        code, out = self.call("query-sources", result, "specs/db-setup-door.md")
        self.assertEqual(code, 0, out)
        self.assertEqual(out, "1. relative-file:operations/x.md\n"
                              "2. relative-file:specs/db-setup-door.md\nPASS\n")
        code, out = self.call("query-sources", result, "specs/other.md")
        self.assertEqual(code, 1)
        self.assertTrue(out.endswith("FAIL: none of specs/other.md\n"))


if __name__ == "__main__":
    unittest.main()
