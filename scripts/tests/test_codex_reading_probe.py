"""Service-free checks for the issue #214 proof-of-concept outbox."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "codex_reading_probe", Path(__file__).resolve().parents[1] / "probe_codex_reading.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "journal.sqlite"
        self.journal = probe.Journal(self.path)
        self.addCleanup(self.journal.close)

    def test_selection_change_does_not_change_prepared_snapshot(self):
        context = {"passage": "first", "references": [{"text": "original"}]}
        snapshot = self.journal.prepare("request", context, "Explain")
        context["passage"] = "second"
        context["references"][0]["text"] = "changed"
        stored = self.journal.db.execute("SELECT body FROM exchange").fetchone()[0]
        self.assertEqual(snapshot, stored)
        self.assertEqual(json.loads(stored)["context"]["references"][0]["text"], "original")

    def test_changed_payload_cannot_reuse_request_id(self):
        self.journal.prepare("request", {"revision": 1}, "Explain")
        with self.assertRaisesRegex(ValueError, "different snapshot"):
            self.journal.prepare("request", {"revision": 2}, "Explain")

    def test_claim_survives_restart_and_blocks_ambiguous_resend(self):
        self.journal.prepare("request", {}, "Explain")
        self.journal.claim("request")
        self.journal.close()
        reopened = probe.Journal(self.path)
        self.addCleanup(reopened.close)
        reopened.prepare("request", {}, "Explain")
        with self.assertRaisesRegex(ValueError, "never resend"):
            reopened.claim("request")

    def test_only_one_connection_can_claim_request(self):
        self.journal.prepare("request", {}, "Explain")
        another = probe.Journal(self.path)
        self.addCleanup(another.close)
        self.journal.claim("request")
        with self.assertRaises(ValueError):
            another.claim("request")

    def test_reconcile_requires_exactly_one_matching_turn(self):
        self.journal.prepare("request", {}, "Explain")
        self.journal.claim("request")
        turn = {"id": "turn", "status": "completed", "items": [
            {"type": "userMessage", "clientId": "request"}]}
        for history in ([], [turn, {**turn, "id": "duplicate"}]):
            with self.assertRaisesRegex(ValueError, "explicit recovery"):
                self.journal.reconcile("request", history)
        self.assertEqual(self.journal.reconcile("request", [turn]), turn)
        self.assertEqual(self.journal.db.execute(
            "SELECT state, provider_turn FROM exchange").fetchone(), ("completed", "turn"))

    def test_context_limit_counts_utf8_bytes(self):
        with self.assertRaisesRegex(ValueError, "12,000-byte"):
            self.journal.prepare("request", {"passage": "é" * 6000}, "Explain")


if __name__ == "__main__":
    unittest.main()
