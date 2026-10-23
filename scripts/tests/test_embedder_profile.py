"""Selector tests with fake systemctl/busctl/curl and real temporary enablement links."""
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
home = pathlib.Path(os.environ["HOME"])
persistent = home / ".config/systemd/user"
runtime = pathlib.Path(os.environ["XDG_RUNTIME_DIR"]) / "systemd/user"
roots = {persistent, runtime, *(pathlib.Path(p) for p in state.get("unit_paths", []))}
if name == "curl":
    profile = state.get("metadata_profile") or (state["active"][0].split("@")[1].split(".")[0] if state["active"] else "absent")
    print(json.dumps({"data": [{"profile": profile, "dimension": 2048, "max_seq_length": 11900}]}))
elif name == "busctl":
    assert args == ["--user", "--json=short", "get-property", "org.freedesktop.systemd1",
                    "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "UnitPath"]
    if state.get("unit_path_failure"):
        print("Failed to connect to bus", file=sys.stderr)
        code = 1
    elif "unit_path_response" in state:
        print(json.dumps(state["unit_path_response"]))
    else:
        print(json.dumps({"type": "as", "data": state.get("unit_paths", [str(persistent), str(runtime)])}))
else:
    args.remove("--user")
    verb = args[0]
    unit = args[-1]
    if verb == "list-units":
        for value in state["active"]:
            print(value, "loaded active running Embedder")
    elif verb == "list-unit-files":
        # Reproduce the original failure: a glob can show only the template,
        # even with enabled instances. This mock must not rescue old discovery.
        print("hades-embedder@.service indirect enabled")
        code = state.get("list_unit_files_exit", 1)
    elif verb == "is-active":
        code = 0 if unit in state["active"] else 3
    elif verb == "is-enabled":
        count = state["is_enabled_calls"] = state.get("is_enabled_calls", 0) + 1
        fault = state.get("is_enabled_faults", {}).get(str(count))
        if fault:
            print(fault.get("output", ""), end="")
            print("fixture enablement query failure", file=sys.stderr)
            code = fault["status"]
        else:
            statuses = [state["enabled"].get(value, "disabled") for value in args[1:]]
            print("\n".join(statuses))
            code = 0 if any(s in ("enabled", "enabled-runtime", "alias", "static", "indirect", "generated", "transient") for s in statuses) else 1
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
        link = (runtime if "--runtime" in args else persistent) / "default.target.wants" / unit
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to("../hades-embedder@.service")
    elif verb == "disable":
        mode = "enabled-runtime" if "--runtime" in args else "enabled"
        if state["enabled"].get(unit) == mode and unit not in state.get("sticky_enabled", []):
            state["enabled"].pop(unit, None)
        if unit not in state.get("sticky_enabled", []):
            for root in roots:
                if (root == runtime) != ("--runtime" in args):
                    continue
                for pattern in ("*.wants", "*.requires"):
                    for target in root.glob(pattern):
                        link = target / unit
                        if link.is_symlink():
                            link.unlink()
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
        self.profiles = self.root / ".config/hades/embedder-profiles"
        self.profiles.mkdir(parents=True)
        for profile in ("gpu1", "gpu2"):
            (self.profiles / f"{profile}.conf").write_text(
                f"HADES_EMBEDDER_PROFILE={profile}\nCUDA_VISIBLE_DEVICES=2\n"
            )
        binaries = self.root / "bin"
        binaries.mkdir()
        for name in ("systemctl", "busctl", "curl"):
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
        for link in self.root.rglob("hades-embedder@*.service"):
            if link.is_symlink():
                link.unlink()
        for profile, mode in enabled.items():
            self.add_link(profile, mode)
        self.state_path.write_text(json.dumps({
            "active": [UNIT.format(p) for p in active],
            "enabled": {UNIT.format(p): state for p, state in enabled.items()},
            "log": [], **extra,
        }))

    def add_link(self, profile, mode="enabled", root=None, dependency="default.target.wants"):
        if root is None:
            root = self.root / ("runtime/systemd/user" if mode == "enabled-runtime" else ".config/systemd/user")
        link = root / dependency / UNIT.format(profile)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to("../hades-embedder@.service")
        return link

    def assert_no_mutations(self):
        self.assertTrue(all(entry[2] in ("list-units", "is-enabled")
                            for entry in self.state()["log"] if entry[0] == "systemctl"))

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
                self.assert_no_mutations()

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

    def test_template_only_listing_does_not_hide_enabled_instances(self):
        for status in (0, 1):
            with self.subTest(list_unit_files_status=status):
                self.set_state(["gpu2"], {"gpu2": "enabled"}, list_unit_files_exit=status)
                result = self.run_selector("gpu1")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_selected("gpu1")
                self.assertFalse(any(e[:3] == ["systemctl", "--user", "list-unit-files"]
                                     for e in self.state()["log"]))

    def test_orphan_persistent_and_runtime_instances_are_reconciled(self):
        for mode in ("enabled", "enabled-runtime"):
            with self.subTest(mode=mode):
                self.set_state(["gpu2"], {"gpu0": mode, "gpu2": "enabled"})
                result = self.run_selector("gpu1")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_selected("gpu1")
                self.assertFalse(any(self.root.rglob(UNIT.format("gpu0"))))

    def test_removed_configuration_does_not_hide_enabled_instance(self):
        self.set_state(["gpu2"], {"gpu1": "enabled", "gpu2": "enabled"})
        (self.profiles / "gpu1.conf").unlink()
        result = self.run_selector("gpu2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_selected("gpu2")

    def test_requires_links_and_manager_paths_with_spaces_are_discovered(self):
        custom = self.root / "custom unit path/systemd/user"
        self.set_state([], {"retired": "enabled"}, unit_paths=[str(custom)])
        self.add_link("retired", root=custom, dependency="custom.target.requires")
        result = self.run_selector("show")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{UNIT.format('retired')} enabled", result.stdout)
        self.assert_no_mutations()

    def test_duplicate_links_and_profile_names_are_queried_once_per_snapshot(self):
        self.add_link("gpu2", "enabled-runtime", dependency="extra.target.requires")
        result = self.run_selector("gpu2")
        self.assertEqual(result.returncode, 0, result.stderr)
        queries = [e[3:] for e in self.state()["log"] if e[:3] == ["systemctl", "--user", "is-enabled"]]
        self.assertEqual(queries, [[UNIT.format("gpu1"), UNIT.format("gpu2")]] * 2)
        self.assert_selected("gpu2")

    def test_invalid_profile_filenames_are_excluded_from_list_and_queries(self):
        (self.profiles / "bad name.conf").write_text("CUDA_VISIBLE_DEVICES=0\n")
        result = self.run_selector("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line.split()[0] for line in result.stdout.splitlines()], ["gpu1", "gpu2"])
        result = self.run_selector("show")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("bad name" in arg for entry in self.state()["log"] for arg in entry))

    def test_initial_query_failure_or_malformed_snapshot_never_mutates(self):
        for output, status in [("", 1), ("", 0), ("enabled\n", 0),
                               ("disabled\nmystery\n", 1), ("disabled\ndisabled\n", 0),
                               ("disabled\nenabled\n", 1), ("disabled\ndisabled\n", 124),
                               ("disabled\nnot-found\n", 4)]:
            with self.subTest(output=output, status=status):
                self.set_state(["gpu2"], {"gpu2": "enabled"},
                               is_enabled_faults={"1": {"output": output, "status": status}})
                result = self.run_selector("gpu1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot read", result.stderr)
                self.assert_selected("gpu2")
                self.assert_no_mutations()

    def test_failed_final_snapshot_rolls_back_and_does_not_report_selection(self):
        self.set_state(["gpu2"], {"gpu2": "enabled"},
                       is_enabled_faults={"2": {"output": "", "status": 1}})
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assert_selected("gpu2")
        self.assertIn("prior unit/enablement state restored", result.stderr)
        self.assertNotIn("selected profile:", result.stdout)

    def test_query_failure_during_rollback_is_reported_as_incomplete(self):
        for query in ("2", "3"):
            with self.subTest(query=query):
                self.set_state(["gpu2"], {"gpu2": "enabled"}, fail_starts=[UNIT.format("gpu1")],
                               is_enabled_faults={query: {"output": "", "status": 1}})
                result = self.run_selector("gpu1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ROLLBACK INCOMPLETE", result.stderr)
                self.assertNotIn("prior unit/enablement state restored", result.stderr)

    def test_orphan_enablement_is_restored_after_start_failure(self):
        self.set_state(["gpu2"], {"gpu0": "enabled-runtime", "gpu2": "enabled"},
                       fail_starts=[UNIT.format("gpu1")])
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.state()["enabled"], {UNIT.format("gpu0"): "enabled-runtime", UNIT.format("gpu2"): "enabled"})
        self.assertIn("prior unit/enablement state restored", result.stderr)

    def test_manager_path_failure_never_produces_an_empty_success_snapshot(self):
        for extra in ({"unit_path_failure": True}, {"unit_path_response": {"type": "s", "data": "bad"}},
                      {"unit_path_response": {"type": "as", "data": []}}):
            with self.subTest(extra=extra):
                self.set_state(["gpu2"], {"gpu2": "enabled"}, **extra)
                result = self.run_selector("gpu1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot discover", result.stderr)
                self.assert_selected("gpu2")
                self.assert_no_mutations()

    def test_enablement_that_cannot_be_removed_never_reports_convergence(self):
        self.set_state(["gpu2"], {"gpu0": "enabled", "gpu2": "enabled"},
                       sticky_enabled=[UNIT.format("gpu0")])
        result = self.run_selector("gpu1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not converge", result.stderr)
        self.assertNotIn("selected profile:", result.stdout)


if __name__ == "__main__":
    unittest.main()
