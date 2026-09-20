"""Selector tests with fake systemctl/curl; no systemd, GPU, or network access."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/systemd/hades-embedder-profile"
UNIT = "hades-embedder@{}.service"
MOCK = r'''#!/usr/bin/env python3
import json, os, pathlib, signal, sys
path = pathlib.Path(os.environ["HADES_PROFILE_TEST_STATE"])
state = json.loads(path.read_text())
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
state["log"].append([name, *args])
code = 0
if name == "curl":
    profile = state.get("metadata_profile") or (state["active"][0].split("@")[1].split(".")[0] if state["active"] else "absent")
    print(json.dumps({"data": [{"profile": profile, "dimension": 2048, "max_seq_length": 11900}]}))
else:
    args.remove("--user")
    verb = args[0]
    unit = args[-1]
    if verb == "list-units":
        for value in state["active"]:
            print(value, "loaded active running Embedder")
    elif verb == "list-unit-files":
        for value, status in sorted(state["enabled"].items()):
            print(value, status, "enabled")
    elif verb == "is-active":
        code = 0 if unit in state["active"] else 3
    elif verb == "start":
        if unit in state.get("fail_starts", []):
            code = 1
        elif unit not in state["active"]:
            state["active"].append(unit)
    elif verb == "stop":
        if unit in state.get("fail_stops", []):
            code = 1
        elif unit in state["active"]:
            state["active"].remove(unit)
    elif verb == "enable":
        state["enabled"][unit] = "enabled-runtime" if "--runtime" in args else "enabled"
    elif verb == "disable":
        if "--runtime" not in args or state["enabled"].get(unit) == "enabled-runtime":
            state["enabled"].pop(unit, None)
    elif verb != "reset-failed":
        raise AssertionError(args)
interrupt = name == "systemctl" and args[0] == "start" and state.get("interrupt_start") == args[-1]
if interrupt:
    state.pop("interrupt_start")
path.write_text(json.dumps(state))
if interrupt:
    os.kill(os.getppid(), signal.SIGTERM)
sys.exit(code)
'''


class ProfileSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hades-profile-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        profiles = self.root / ".config/hades/embedder-profiles"
        profiles.mkdir(parents=True)
        for profile in ("gpu1", "gpu2"):
            (profiles / f"{profile}.conf").write_text(
                f"HADES_EMBEDDER_PROFILE={profile}\nCUDA_VISIBLE_DEVICES=2\n"
            )
        binaries = self.root / "bin"
        binaries.mkdir()
        for name in ("systemctl", "curl"):
            path = binaries / name
            path.write_text(MOCK)
            path.chmod(0o755)
        self.state_path = self.root / "state.json"
        self.env = dict(os.environ, HOME=str(self.root),
                        XDG_RUNTIME_DIR=str(self.root / "runtime"),
                        HADES_PROFILE_TEST_STATE=str(self.state_path),
                        HADES_PROFILE_READY_TIMEOUT="1",
                        PATH=str(binaries) + os.pathsep + os.environ["PATH"])
        self.set_state(["gpu2"], {"gpu2": "enabled"})

    def set_state(self, active, enabled, **extra):
        self.state_path.write_text(json.dumps({
            "active": [UNIT.format(p) for p in active],
            "enabled": {UNIT.format(p): state for p, state in enabled.items()},
            "log": [], **extra,
        }))

    def state(self):
        return json.loads(self.state_path.read_text())

    def run_selector(self, target):
        return subprocess.run(["bash", str(SCRIPT), target], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def assert_selected(self, profile):
        state = self.state()
        self.assertEqual(state["active"], [UNIT.format(profile)])
        self.assertEqual(state["enabled"], {UNIT.format(profile): "enabled"})

    def test_reconciles_failed_enabled_instance_without_restarting_selected(self):
        self.set_state(["gpu2"], {"gpu1": "enabled", "gpu2": "enabled"})
        result = self.run_selector("gpu2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_selected("gpu2")
        self.assertFalse(any(entry[2] in ("start", "stop") for entry in self.state()["log"] if entry[0] == "systemctl"))
        self.assertIn("inference has not been tested", result.stdout)

    def test_repeated_switches_and_simulated_reboot_have_one_persistent_profile(self):
        for target in ("gpu1", "gpu1", "gpu2"):
            result = self.run_selector(target)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_selected(target)
            state = self.state()
            state["active"] = [unit for unit, mode in state["enabled"].items() if mode == "enabled"]
            self.state_path.write_text(json.dumps(state))
            self.assert_selected(target)

    def test_runtime_only_selection_becomes_persistent(self):
        self.set_state(["gpu2"], {"gpu1": "enabled", "gpu2": "enabled-runtime"})
        result = self.run_selector("gpu2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_selected("gpu2")

    def test_start_failure_restores_old_profile_and_boot_selection(self):
        self.set_state(["gpu2"], {"gpu2": "enabled"}, fail_starts=[UNIT.format("gpu1")])
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assert_selected("gpu2")
        self.assertIn("prior unit/enablement state restored", result.stderr)

    def test_wrong_metadata_rolls_back(self):
        self.set_state(["gpu2"], {"gpu2": "enabled"}, metadata_profile="gpu2")
        self.env["PYTHONOPTIMIZE"] = "1"
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assert_selected("gpu2")
        self.assertIn("matching metadata", result.stderr)

    def test_ambiguous_active_state_and_invalid_names_do_not_mutate_services(self):
        for target in ("../gpu1", "missing", "gpu1"):
            with self.subTest(target=target):
                self.set_state(["gpu1", "gpu2"], {"gpu1": "enabled", "gpu2": "enabled"})
                result = self.run_selector(target)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(all(entry[2].startswith("list-") for entry in self.state()["log"]))

    def test_initial_selection_without_previous_active_profile(self):
        self.set_state([], {})
        result = self.run_selector("gpu1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_selected("gpu1")

    def test_signal_during_switch_restores_prior_state(self):
        self.set_state(["gpu2"], {"gpu2": "enabled"}, interrupt_start=UNIT.format("gpu1"))
        result = self.run_selector("gpu1")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assert_selected("gpu2")

    def test_failed_rollback_stop_does_not_start_a_competing_profile(self):
        self.set_state(["gpu2"], {"gpu2": "enabled"}, metadata_profile="gpu2",
                       fail_stops=[UNIT.format("gpu1")])
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ROLLBACK INCOMPLETE", result.stderr)
        self.assertEqual(self.state()["active"], [UNIT.format("gpu1")])
        self.assertEqual(self.state()["enabled"], {UNIT.format("gpu2"): "enabled"})
        self.assertNotIn(["systemctl", "--user", "start", UNIT.format("gpu2")], self.state()["log"])

    def test_concurrent_selector_is_rejected_before_systemctl(self):
        lock = self.root / "runtime/hades/embedder-profile.lock"
        lock.parent.mkdir(parents=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("in progress", result.stderr)
        self.assertEqual(self.state()["log"], [])


if __name__ == "__main__":
    unittest.main()
