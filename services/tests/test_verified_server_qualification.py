"""Pure helpers and refusals of the verified-server qualification harness (#225)."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qualify_verified_server.py"


def script():
    """Import the harness without installing the scripts directory."""
    spec = importlib.util.spec_from_file_location("qualify_verified_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = script()
MODEL = {"id": "jinaai/jina-embeddings-v4"}


def test_check_description_matches_space():
    """The harness accepts a description only for the space it advertises."""
    late = {"model": MODEL, "capabilities": {"late_chunking": True, "tokenizer_offsets": True}}
    conv = {"model": MODEL, "capabilities": {"late_chunking": False, "tokenizer_offsets": False}}
    harness.check_description(late, "late")
    harness.check_description(conv, "conventional")
    with pytest.raises(ValueError):
        harness.check_description(late, "conventional")
    with pytest.raises(ValueError):
        harness.check_description(conv, "late")


def test_check_description_refuses_mixed_capabilities_and_other_models():
    """Half-late capabilities and another model's description are refused."""
    mixed = {"model": MODEL, "capabilities": {"late_chunking": True, "tokenizer_offsets": False}}
    with pytest.raises(ValueError):
        harness.check_description(mixed, "late")
    other = {"model": {"id": "other/model"},
             "capabilities": {"late_chunking": True, "tokenizer_offsets": True}}
    with pytest.raises(ValueError):
        harness.check_description(other, "late")


def test_planning_request_shapes():
    """Planning requests carry each space's route, policy and receipt fields."""
    path, body = harness.planning_request("late", "text", {"r": 1}, 20000)
    assert path == "/v1/late-passages"
    assert body["context_cap_tokens"] == 16384 and body["passage_tokens"] == 500
    assert body["overlap_tokens"] == 200 and body["minimum_window_overlap_tokens"] == 1000
    assert body["expected_receipt"] == {"r": 1} and body["max_input_tokens"] == 20000
    path, body = harness.planning_request("late", "text", {"r": 1}, 4096)
    assert body["context_cap_tokens"] == 4096
    path, body = harness.planning_request("conventional", "text", {"r": 1}, 20000)
    assert path == "/v1/passages" and "context_cap_tokens" not in body
    assert body["task"] == "retrieval.passage" and body["input"] == "text"


def test_planning_request_fields_match_the_verified_app():
    """Planning request fields match the verified app's request models exactly."""
    from embedding.verified_http import LatePassageRequest, PassageRequest
    _, late = harness.planning_request("late", "text", {"r": 1}, 20000)
    _, conv = harness.planning_request("conventional", "text", {"r": 1}, 20000)
    assert set(late) == set(LatePassageRequest.model_fields)
    assert set(conv) == set(PassageRequest.model_fields)


def test_cli_refuses_existing_output_dir(tmp_path):
    """The qualification CLI refuses an existing output dir and writes nothing."""
    (tmp_path / "out").mkdir()
    r = subprocess.run([sys.executable, str(SCRIPT), "--registry", str(tmp_path), "--provider",
                        "p", "--space", "late", "--device", "cpu", "--output-dir",
                        str(tmp_path / "out")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert list((tmp_path / "out").iterdir()) == []
    # One line naming the refusal, not a FileExistsError traceback (#225).
    assert "Traceback" not in r.stderr
    assert r.stderr.strip().splitlines() == [
        "qualify_verified_server: the output directory already exists; name a new one"]


def test_relative_registry_reaches_the_child_absolute(tmp_path, monkeypatch):
    """A relative --registry is resolved against the caller's cwd, not services/."""
    (tmp_path / "registry").mkdir()
    monkeypatch.chdir(tmp_path)
    launched = {}
    real = subprocess.Popen

    def capture(command, **kwargs):
        """Capture the server launch and stop; pass other commands through."""
        if "embedding.verified_server" not in command:
            return real(command, **kwargs)
        launched["command"], launched["cwd"] = command, kwargs.get("cwd")
        raise RuntimeError("launch captured")

    monkeypatch.setattr(harness.subprocess, "Popen", capture)
    # This test is about the launch; a developer's uncommitted edits must not stop it.
    monkeypatch.setattr(harness, "tracked_changes", lambda repo: "")
    # main() sets a process-wide umask; restore it so later tests' modes are unchanged.
    umask = os.umask(0o022)
    os.umask(umask)
    try:
        code = harness.main(["--registry", "registry", "--provider", "p",
                             "--space", "late", "--device", "cpu",
                             "--output-dir", str(tmp_path / "out")])
    finally:
        os.umask(umask)
    assert code == 1
    command = launched["command"]
    registry = command[command.index("--registry") + 1]
    assert launched["cwd"] != tmp_path
    assert registry == str((tmp_path / "registry").resolve())


def test_status_after_must_still_report_the_space_loaded():
    """The status after planning must report the selected space, still loaded."""
    harness.check_status_after({"space": "late", "loaded": True}, "late")
    with pytest.raises(RuntimeError, match="unloaded"):
        harness.check_status_after({"space": "late", "loaded": False}, "late")
    with pytest.raises(RuntimeError, match="different space"):
        harness.check_status_after({"space": "conventional", "loaded": True}, "late")


def test_plan_must_not_be_empty():
    """A 200 with no passages or windows is not a qualifying plan."""
    harness.check_plan("late", {"passages": [{}], "windows": [{}]})
    harness.check_plan("conventional", {"spans": [{}]})
    with pytest.raises(RuntimeError, match="no passages"):
        harness.check_plan("late", {"passages": [], "windows": [{}]})
    with pytest.raises(RuntimeError, match="no windows"):
        harness.check_plan("late", {"passages": [{}], "windows": []})
    with pytest.raises(RuntimeError, match="no spans"):
        harness.check_plan("conventional", {"spans": []})


def git(repo, *args):
    """Run git in a scratch repository with a fixed identity."""
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   cwd=repo, check=True, capture_output=True)


def test_tracked_changes_ignores_untracked_files(tmp_path):
    """A modified tracked file is a change; an untracked one is not."""
    git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-q", "-m", "a")
    (tmp_path / "untracked").write_text("")
    assert harness.tracked_changes(tmp_path) == ""
    (tmp_path / "a.py").write_text("x = 2\n")
    assert "a.py" in harness.tracked_changes(tmp_path)


def test_cli_refuses_a_dirty_tree_before_writing(tmp_path, monkeypatch, capsys):
    """A tree with tracked changes is refused in one line; no evidence is written.

    The report records HEAD as the implementation commit, which would misdescribe a
    run of uncommitted code (#225).
    """
    monkeypatch.setattr(harness, "tracked_changes", lambda repo: " M services/x.py\n")
    code = harness.main(["--registry", str(tmp_path), "--provider", "p", "--space", "late",
                         "--device", "cpu", "--output-dir", str(tmp_path / "out")])
    assert code == 1 and not (tmp_path / "out").exists()
    assert capsys.readouterr().err.strip().splitlines() == [
        "qualify_verified_server: the checkout has uncommitted changes to tracked files;"
        " commit or stash them, so the report's commit describes the code that ran"]
