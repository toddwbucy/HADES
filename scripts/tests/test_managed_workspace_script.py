"""End-to-end tests for scripts/install/thinkpad-managed-workspace.sh.

Each test runs the script against scratch git repositories, a scratch data directory
and stub commands: a fake `hades` that answers from files, and a `systemctl` that
fails, first on PATH, so no run can reach a live unit, database or provider.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "install" / "thinkpad-managed-workspace.sh"
HELPER = SCRIPT.with_name("managed_workspace.py")

FAKE_HADES = """#!/usr/bin/env python3
\"\"\"Answer `ingest` and `db query` as the CLI would, logging every call.\"\"\"
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_HADES_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
ingested = Path(os.environ["FAKE_HADES_INGESTED"])
if "ingest" in args:
    files = args[args.index("--root") + 2:]
    if ingested.exists():
        # The managed path refuses an existing source without --force.
        data = {"completed": 0, "failed": len(files), "skipped": 0}
        print(json.dumps({"success": False, "command": "ingest", "data": data}))
        sys.exit(1)
    ingested.write_text("yes")
    data = {"completed": len(files), "failed": 0, "skipped": 0,
            "embedding_profile_state": "ready"}
    print(json.dumps({"success": True, "command": "ingest", "data": data}))
elif "query" in args:
    hits = json.loads(Path(os.environ["FAKE_HADES_HITS"]).read_text())
    print(json.dumps({"success": True, "command": "db.query", "data": {"results": hits}}))
else:
    sys.exit(2)
"""

STUB_SYSTEMCTL = """#!/bin/sh
echo "systemctl $*" >> "$STUB_SYSTEMCTL_LOG"
exit 1
"""

# Sources that answer both first-ingest questions, and sources that answer neither.
GOOD_HITS = [{"source_id": "relative-file:specs/db-setup-door.md"},
             {"source_id": "relative-file:installation-thinkpad.md"}]
BAD_HITS = [{"source_id": "relative-file:unrelated.md"}]


class ScriptRun(unittest.TestCase):
    """A scratch home for one script run: data, config, stub binaries and git identity."""

    def setUp(self):
        """Create the scratch directories and the environment every run shares."""
        self.root = Path(tempfile.mkdtemp(prefix="mw-script-"))
        os.chmod(self.root, 0o700)
        self.data = self.root / "data"
        self.state = self.data / "managed-workspace" / "state"
        self.state.mkdir(parents=True)
        self.config = self.root / "config"
        self.config.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        stub = self.root / "stub"
        stub.mkdir()
        self.executable(stub / "systemctl", STUB_SYSTEMCTL)
        self.env = {
            "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.root),
            "STUB_SYSTEMCTL_LOG": str(self.root / "systemctl.log"),
            "HADES_MW_DATA": str(self.data),
            "HADES_MW_CONFIG_DIR": str(self.config),
            "HADES_MW_UNIT_DIR": str(self.root / "units"),
            "HADES_MW_BIN_DIR": str(self.bin),
            "HADES_MW_YAML": str(self.config / "hades.yaml"),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }

    def tearDown(self):
        """Remove the scratch tree."""
        shutil.rmtree(self.root)

    def executable(self, path, text):
        """Write an executable file and return its path."""
        path.write_text(text)
        path.chmod(0o755)
        return path

    def git(self, repo, *argv):
        """Run git in REPO with the scratch identity and return its stdout."""
        return subprocess.run(["git", "-C", str(repo), *argv], env=self.env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, repo, files):
        """Write FILES (relative path to text) into REPO, commit them, return the SHA."""
        for relative, text in files.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-q", "-m", "fixture")
        return self.git(repo, "rev-parse", "HEAD")

    def run_script(self, *argv, script=SCRIPT, **env):
        """Run the installer with the scratch environment; return (exit code, output)."""
        result = subprocess.run(["bash", str(script), *argv], env={**self.env, **env},
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=60)
        return result.returncode, result.stdout + result.stderr


class CheckoutPin(ScriptRun):
    """install-main holds the checkout's services tree at the installer's own commit."""

    def setUp(self):
        """Build a repository whose installer commit changes services/ after main."""
        super().setUp()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git(self.repo, "init", "-q", "-b", "main")
        self.main_commit = self.commit(self.repo, {
            "services/embedding/verified_server.py": "main\n",
            "scripts/check_daemon_ready.py": "ready\n",
            "scripts/install/managed_workspace.py": HELPER.read_text(),
            "scripts/install/thinkpad-managed-workspace.sh": SCRIPT.read_text(),
        })
        self.installer_commit = self.commit(self.repo, {
            "services/embedding/verified_server.py": "main plus the SIGTERM fix\n",
        })
        self.script = self.repo / "scripts/install/thinkpad-managed-workspace.sh"
        self.checkout = self.root / "checkout"
        self.git(self.root, "clone", "-q", str(self.repo), str(self.checkout))
        # The executables count as installed, so only the checkout decides the state.
        (self.state / "install-main.checked").write_text("backup\n")
        self.env.update(HADES_MW_CHECKOUT=str(self.checkout),
                        HADES_MW_MAIN_COMMIT=self.main_commit)

    def head(self):
        """Return the checkout's HEAD commit."""
        return self.git(self.checkout, "rev-parse", "HEAD")

    def install_main(self, *flags):
        """Run install-main from the scratch repository's copy of the script."""
        return self.run_script(*flags, "install-main", script=self.script)

    def test_a_checkout_at_main_is_brought_to_the_installer_commit(self):
        """A checkout at main lacks this PR's services/ and is fast-forwarded to it."""
        self.git(self.checkout, "reset", "-q", "--hard", self.main_commit)
        code, out = self.install_main("--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn("== install-main (", out)
        self.assertNotIn("already done", out)
        code, out = self.install_main()
        self.assertEqual(code, 0, out)
        self.assertIn("install-main: done", out)
        self.assertEqual(self.head(), self.installer_commit)

    def test_a_checkout_at_the_installer_commit_is_done(self):
        """A checkout equal to the installer's commit, with main as ancestor, passes."""
        code, out = self.install_main()
        self.assertEqual(code, 0, out)
        self.assertIn("install-main: already done", out)

    def test_local_changes_outside_the_pinned_paths_do_not_block(self):
        """An uncommitted installer copy in the checkout does not stop the merge."""
        self.git(self.checkout, "reset", "-q", "--hard", self.main_commit)
        (self.checkout / "scripts/install/managed_workspace.py").write_text("# edited\n")
        code, out = self.install_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.head(), self.installer_commit)

    def test_local_changes_under_services_are_refused(self):
        """An uncommitted change to the checkout's services/ is a refusal, not a merge."""
        self.git(self.checkout, "reset", "-q", "--hard", self.main_commit)
        (self.checkout / "services/embedding/verified_server.py").write_text("local\n")
        code, out = self.install_main()
        self.assertEqual(code, 1, out)
        self.assertIn("local changes under services/", out)
        self.assertEqual(self.head(), self.main_commit)

    def test_an_installer_without_main_is_blocked(self):
        """An installer commit that lacks main's commit cannot pin services/."""
        self.git(self.repo, "checkout", "-q", "-b", "side", self.main_commit)
        side = self.commit(self.repo, {"unrelated.txt": "x\n"})
        self.git(self.repo, "checkout", "-q", "main")
        code, out = self.run_script("install-main", script=self.script,
                                    HADES_MW_MAIN_COMMIT=side)
        self.assertEqual(code, 1, out)
        self.assertIn("does not contain", out)

    def test_local_changes_in_the_installer_services_are_blocked(self):
        """The installer's services/ must be committed, since its commit is the pin."""
        (self.repo / "services/embedding/verified_server.py").write_text("uncommitted\n")
        code, out = self.install_main()
        self.assertEqual(code, 1, out)
        self.assertIn("local changes", out)


class IngestGates(ScriptRun):
    """ingest is done only once both query gates pass, and a re-run never re-ingests."""

    def setUp(self):
        """Stage a finished desk move, a clean corpus and the fake hades."""
        super().setUp()
        (self.state / "move-desk.done").write_text("record\n")
        (self.config / "daemon.env").write_text("")
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.git(self.corpus, "init", "-q", "-b", "main")
        self.commit(self.corpus, {"specs/db-setup-door.md": "door\n",
                                  "installation-thinkpad.md": "laptop\n"})
        self.hits = self.root / "hits.json"
        self.log = self.root / "hades.log"
        self.executable(self.bin / "hades", FAKE_HADES)
        self.env.update(HADES_MW_CHECKOUT=str(self.root),
                        HADES_MW_CORPUS=str(self.corpus),
                        FAKE_HADES_LOG=str(self.log), FAKE_HADES_HITS=str(self.hits),
                        FAKE_HADES_INGESTED=str(self.root / "ingested"))

    def calls(self, verb):
        """Count the fake hades calls whose arguments include VERB."""
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return sum(verb in line.split() for line in lines)

    def test_a_failed_gate_leaves_ingest_undone_and_a_rerun_only_requeries(self):
        """A missed question keeps ingest pending, and the re-run only asks again."""
        self.hits.write_text(json.dumps(BAD_HITS))
        code, out = self.run_script("ingest")
        self.assertEqual(code, 1, out)
        self.assertFalse((self.state / "ingest.done").exists(), out)
        self.assertEqual(self.calls("ingest"), 1)

        code, out = self.run_script("--dry-run", "ingest")
        self.assertIn("== ingest (ingested, query gates not passed)", out)

        self.hits.write_text(json.dumps(GOOD_HITS))
        code, out = self.run_script("ingest")
        self.assertEqual(code, 0, out)
        self.assertIn("ingest: done", out)
        self.assertTrue((self.state / "ingest.done").exists())
        self.assertEqual(self.calls("ingest"), 1)
        self.assertEqual(self.calls("query"), 4)


if __name__ == "__main__":
    sys.exit(unittest.main())
