"""Pure helpers and refusals of the verified-server qualification harness (#225)."""
import importlib.util
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
    assert r.returncode != 0
    assert list((tmp_path / "out").iterdir()) == []
